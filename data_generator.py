
from landmarks import DlibLandmarksDetector, add_boundary_points
import numpy as np
import tensorflow as tf
from landmarks import compute_displacements_interpolation
from transformation import deform
from plot import plot_normalized_sequence, plot_landmarks, plot_triangles, save_gif
from pathlib import Path
from settings.facial import DEFAULT_TRIANGULATION

class FrameDataGenerator():
    
    def __init__(self, videos_path, batch_size=1):
        self.videos_path = videos_path
        self.n = len(videos_path)
        self._restart()
        self.batch_size = batch_size
    
    def _load_current_video(self):
        self.current_video = np.load(self.videos_path[self.current_video_index])
    
    def _load_next_video(self):
        self.current_frame_index = 1
        self.current_video_index += 1
        self._load_current_video()
    
    def _restart(self):
        self.current_video_index = 0
        self.current_frame_index = 0
        self._load_current_video()
        self.height = self.current_video.shape[1]
        self.width = self.current_video.shape[2]
    

    def _current_frame(self):
        return self.current_video[self.current_frame_index]
    
    def _next_frame(self):
        return self.current_video[self.current_frame_index+1]
    
    def _previous_frame(self):
        return self.current_video[self.current_frame_index-1]
    
    def _current_video_has_next_frame(self):
        return self.current_frame_index+1 < self.current_video.shape[0]
    
    def _has_current_video(self):
        return self.current_video_index < self.n
    
    def _has_next_video(self):
        return self.current_video_index+1 < self.n
    
    def _end_of_videos(self):
            raise StopIteration
    
    def _get_current_frames(self):
        return self._previous_frame(), self._current_frame(), self._next_frame()
    
    def _get_next_frames(self):
        self.current_frame_index += 1 # Frame index start at one because we have to return the previous frame
        if not self._has_current_video():
            self._end_of_videos()
        while not self._current_video_has_next_frame():
            if self._has_next_video():
                self._load_next_video()
            else:
                self._end_of_videos()
        return self._get_current_frames()
    
    def _get_batch(self):
        batch_previous_frame = []
        batch_current_frame = []
        batch_next_frame = []
        for i in range(self.batch_size):
            previous_frame, current_frame, next_frame = self._get_next_frames()
            batch_previous_frame.append(previous_frame)
            batch_current_frame.append(current_frame)
            batch_next_frame.append(next_frame)
        return np.array(batch_previous_frame), np.array(batch_current_frame), np.array(batch_next_frame)


    def __next__(self):
        return self._get_batch()
    
    def __call__(self):
        return next(self)
    
    def take(self, n):
        for i in range(n+1): # Frame index start at one
            try:
                yield self()
            except StopIteration:
                self._restart()
    
    def __iter__(self):
        return self
    
    def repeat(self):
        while True:
            try:
                yield self()
            except StopIteration:
                self._restart()


class PreprocessedFrameDataGenerator(FrameDataGenerator):

    def __init__(self, videos_path, generator, discriminator, generator_loss_function, discriminator_loss_function, landmark_detector=None, repeat=False):
        if landmark_detector:
            self.landmark_detector = landmark_detector
        else:
            self.landmark_detector = DlibLandmarksDetector()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss_function = generator_loss_function
        self.discriminator_loss_function = discriminator_loss_function
        self.repeat = repeat
        super().__init__(videos_path, 1)
    
    def _load_current_video(self):
        super()._load_current_video()
        self._deformed_frame = self._first_frame() # first frame as input
        self._current_source_video = self.current_video # TODO: change this to the most similar video
        self._landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self.current_video)
        self._current_source_landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self._current_source_video)
        self._deformed_landmarks = self._landmarks[0]
        self.previously_generated = self._first_frame() # first frame as input
    
    def _load_next_video(self):
        super()._load_next_video()
        self.height = self.current_video.shape[1]
        self.width = self.current_video.shape[2]
    
    def _current_frame(self):
        return self.current_video[self.current_frame_index:self.current_frame_index+1]
    
    def _next_frame(self):
        return self.current_video[self.current_frame_index+1:self.current_frame_index+2]
    
    def _previous_frame(self):
        return self.current_video[self.current_frame_index-1:self.current_frame_index]

    def _first_frame(self):
        return self.current_video[0:1]
    
    def _current_landmarks(self):
        return self._landmarks[self.current_frame_index:self.current_frame_index+1]
    
    def _next_landmarks(self):
        return self._landmarks[self.current_frame_index+1:self.current_frame_index+2]
    
    def _previous_landmarks(self):
        return self._landmarks[self.current_frame_index-1:self.current_frame_index]

    def _deform_current_frame(self): # TODO: put in next the change of _deformed_landmarks and _deformed_frame
        self._deformed_frame, self._deformed_landmarks = deform(
            self._deformed_frame[0], # only one frame
            self._deformed_landmarks,
            self._current_source_landmarks[self.current_frame_index-1],
            self._current_source_landmarks[self.current_frame_index]
        )
        self._deformed_frame = self._deformed_frame[None, :, :, :] # add batch dimension
    
    def _get_displacements(self):
        displacements = compute_displacements_interpolation(
            self._current_source_landmarks[self.current_frame_index-1],
            self._current_source_landmarks[self.current_frame_index],
            self.width,
            self.height,
            1)
        return displacements[None, :, :, :] # add batch dimension

    def _get_current_frames(self):
        return None # TODO: get generated frame?
    
    def generate_next_frame(self):
        try:
            _ = self._get_next_frames()
        except StopIteration:
            if self.repeat:
                self._restart()
                _ = self._get_next_frames()
            else:
                raise StopIteration
        previously_generated = self.previously_generated
        deformed_frame = self._deformed_frame 
        self._deform_current_frame()
        l = [self._first_frame(), previously_generated, deformed_frame, self._get_displacements()]
        generated_frame = self.generator(l)
        self.previously_generated = generated_frame
        return previously_generated, generated_frame, self._current_frame()
    
    def next_loss(self):
        previously_generated, generated_frame, current_frame = self.generate_next_frame()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previously_generated)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frame)
        previous_frame = self._previous_frame()
        disc_real_output = self.discriminator([previous_frame, current_frame], training=True)
        disc_generated_output = self.discriminator([previous_frame, generated_frame], training=True)
        disc_loss_value = self.discriminator_loss_function(disc_real_output, disc_generated_output)
        total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value = self.generator_loss_function(
            disc_generated_output,
            previously_generated,
            generated_frame,
            previous_frame,
            current_frame,
            self._previous_landmarks(),
            self._current_landmarks(),
            previous_gen_landmarks,
            current_gen_landmarks
        )        
        return total_gen_loss, disc_loss_value, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value
    
    def restart(self):
        return self._restart()
    
    def __next__(self):
        yield self.generate_next_frame()
    


class MultipleVideoDataGenerator():
    
    def __init__(self,
                 videos_path, 
                 generator,
                 discriminator,
                 generator_loss_function,
                 discriminator_loss_function,
                 batch_size,
                 height,
                 width,
                 landmark_detector=None):
        self.n = len(videos_path)
        self.batch_size = batch_size
        if self.n < self.batch_size:
            raise ValueError("There are less videos than the batch size")
        self.videos_path = videos_path
        self.landmark_detector = landmark_detector if landmark_detector else DlibLandmarksDetector()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss_function = generator_loss_function
        self.discriminator_loss_function = discriminator_loss_function
        self.height = height
        self.width = width
        self.num_landmarks = self.landmark_detector.num_landmarks
        self._restart()

    def _load_batch_video(self, batch_index):
        self._batch_videos[batch_index] = np.load(self.videos_path[self._batch_videos_index[batch_index]])
        if self._video_has_next_frames(batch_index):
            self._batch_landmarks[batch_index] = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self._batch_videos[batch_index])
            self._batch_deformed_frames[batch_index] = self._first_frame(batch_index) # first frame as input
            self._batch_deformed_landmarks[batch_index] = self._first_landmarks(batch_index)
            self._batch_source_videos[batch_index] = self._batch_videos[batch_index] # TODO: change this to the most similar video
            self._batch_source_landmarks[batch_index] = self._batch_landmarks[batch_index] # TODO: change this to the most similar video
            self._batch_previously_generated_frames[batch_index] = self._batch_deformed_frames[batch_index] # first frame as input
            self._batch_videos_frame_index[batch_index] = 1 # Frame index start at one to always return the previous frame
            return True
        else: # video has less than three frames
            return False

    def _load_batch_videos(self):
        for batch_index in range(self.batch_size):
            self._load_next_video(batch_index)
    
    def _batch_image_zeros(self):
        return np.zeros((self.batch_size, self.height, self.width, 3))
    
    def _batch_landmarks_zeros(self):
        return np.zeros((self.batch_size, self.num_landmarks, 2))
    
    def _batch_displacements_zeros(self):
        return np.zeros((self.batch_size, self.height, self.width, 1))
    
    def empty_list_batch(self):
        return [None] * self.batch_size
    
    def _restart(self):
        self._batch_videos_index = np.zeros(self.batch_size, dtype=int)
        self._batch_next_video_index = 0
        self._batch_videos_frame_index = np.ones(self.batch_size, dtype=int)
        self._batch_videos = self.empty_list_batch()
        self._batch_landmarks = self.empty_list_batch()
        self._batch_source_videos = self.empty_list_batch()
        self._batch_source_landmarks = self.empty_list_batch()
        self._batch_deformed_frames = self._batch_image_zeros()
        self._batch_deformed_landmarks = self._batch_landmarks_zeros()
        self._batch_previously_generated_frames = self._batch_image_zeros()
        self._batch_displacements = self._batch_displacements_zeros()
        self._load_batch_videos()
    
    def _frame_index(self, batch_index, frame_shift=0):
        return self._batch_videos_frame_index[batch_index]+frame_shift

    def _get_item(self, array, batch_index, frame_shift):
        frame_index = self._frame_index(batch_index, frame_shift)
        return array[batch_index][frame_index]
    
    def _get_frame(self, batch_index, frame_shift=0):
        return self._get_item(self._batch_videos, batch_index, frame_shift)
    
    def _get_frames(self, frame_shift=0):
        frames = self._batch_image_zeros()
        for batch_index in range(self.batch_size):
            frames[batch_index] = self._get_frame(batch_index, frame_shift)
        return frames
    
    def _get_landmark(self, batch_index, frame_shift=0):
        return self._get_item(self._batch_landmarks, batch_index, frame_shift)
    
    def _get_landmarks(self, frame_shift=0):
        landmarks = self._batch_landmarks_zeros()
        for batch_index in range(self.batch_size):
            landmarks[batch_index] = self._get_landmark(batch_index, frame_shift)
        return landmarks
    
    def _first_frame(self, batch_index):
        return self._batch_videos[batch_index][0]
    
    def _first_landmarks(self, batch_index):
        return self._batch_landmarks[batch_index][0]
    
    def _video_has_next_frames(self, batch_index):
        frame_index = self._frame_index(batch_index,1)
        return frame_index < self._batch_videos[batch_index].shape[0]
    
    def _load_next_video(self, batch_index):
        while (True): # load next video with frames
            self._batch_videos_index[batch_index] = self._batch_next_video_index
            self._batch_next_video_index = (self._batch_next_video_index + 1) % self.n
            if self._load_batch_video(batch_index):
                break
        
    def _load_next_frames(self, batch_index, previously_generated_frame):
        if self._video_has_next_frames(batch_index):
            self._batch_videos_frame_index[batch_index] += 1
            self._batch_previously_generated_frames[batch_index] = previously_generated_frame
        else: 
            self._load_next_video(batch_index)
    
    def _deform_current_frames(self, batch_index):
        frame_index = self._frame_index(batch_index)
        deformed_frame, deformed_landmarks = deform(
            self._batch_deformed_frames[batch_index],
            self._batch_deformed_landmarks[batch_index],
            self._batch_source_landmarks[batch_index][frame_index-1],
            self._batch_source_landmarks[batch_index][frame_index]
        )
        self._batch_deformed_frames[batch_index] = deformed_frame
        self._batch_deformed_landmarks[batch_index] = deformed_landmarks
    
    def _compute_current_displacements(self, batch_index):
        frame_index = self._frame_index(batch_index)
        displacements = compute_displacements_interpolation(
            self._batch_source_landmarks[batch_index][frame_index-1],
            self._batch_source_landmarks[batch_index][frame_index],
            self.width,
            self.height,
            1)
        self._batch_displacements[batch_index] = displacements
    
    def _get_all_inputs(self):
        generated_frames = self._batch_image_zeros()
        current_frames = self._batch_image_zeros()
        previously_generated_frames = self._batch_image_zeros()
        first_frames = self._batch_image_zeros()
        deformed_frames = self._batch_image_zeros()
        displacements = self._batch_displacements_zeros()

        for batch_index in range(self.batch_size):
            self._deform_current_frames(batch_index)
            self._compute_current_displacements(batch_index)
            first_frames[batch_index] = self._first_frame(batch_index)
            deformed_frames[batch_index] = self._batch_deformed_frames[batch_index]
            previously_generated_frames[batch_index] = self._batch_previously_generated_frames[batch_index]
            displacements[batch_index] = self._batch_displacements[batch_index]
            previously_generated_frames[batch_index] = self._batch_previously_generated_frames[batch_index]
            current_frames[batch_index] = self._get_frame(batch_index)
        
        generated_frames = self.generator([
            first_frames,
            previously_generated_frames,
            deformed_frames,
            displacements
        ])
        
        for batch_index in range(self.batch_size):
            self._load_next_frames(batch_index, generated_frames[batch_index])
        
        return previously_generated_frames, generated_frames, current_frames, first_frames, deformed_frames, displacements
    
    def generate_next_frames(self):
        previously_generated_frames, generated_frames, current_frames, _, _, _ = self._get_all_inputs()
        return previously_generated_frames, generated_frames, current_frames
    
    def next_loss(self):
        # get frames and landmarks before generating next frames
        previous_frames = self._get_frames(-1)
        previous_landmarks = self._get_landmarks(-1)
        current_landmarks = self._get_landmarks()
        # generate_next_frames loads the next frames
        previously_generated_frames, generated_frames, current_frames = self.generate_next_frames()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previously_generated_frames)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames)
        frames_diff = current_frames - generated_frames
        generated_diff = generated_frames - previously_generated_frames
        disc_real_output = self.discriminator([previous_frames, current_frames, frames_diff], training=True)
        disc_generated_output = self.discriminator([previously_generated_frames, generated_frames, generated_diff], training=True)
        disc_loss_value = self.discriminator_loss_function(disc_real_output, disc_generated_output)
        total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value = self.generator_loss_function(
            disc_generated_output,
            previously_generated_frames,
            generated_frames,
            previous_frames,
            current_frames,
            previous_landmarks,
            current_landmarks,
            previous_gen_landmarks,
            current_gen_landmarks
        )        
        return total_gen_loss, disc_loss_value, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value
    
    def next_plot(self):
        # get frames and landmarks before generating next frames
        previous_frames = self._get_frames(-1)
        previous_landmarks = self._get_landmarks(-1)
        current_landmarks = self._get_landmarks()
        # _get_all_inputs loads the next frames
        previously_generated_frames, generated_frames, current_frames, first_frames, deformed_frames, displacements = self._get_all_inputs()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previously_generated_frames)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames)
        tf.print("previously_generated_frames")
        plot_normalized_sequence(previously_generated_frames)
        tf.print("generated_frames")
        plot_normalized_sequence(generated_frames)
        tf.print("current_frames")
        plot_normalized_sequence(current_frames)
        tf.print("previous_frames")
        plot_normalized_sequence(previous_frames)
        tf.print("first_frames")
        plot_normalized_sequence(first_frames)
        tf.print("deformed_frames")
        plot_normalized_sequence(deformed_frames)
        tf.print("displacements")
        plot_normalized_sequence(displacements)
        tf.print("previous_gen_landmarks")
        plot_landmarks(previously_generated_frames, previous_gen_landmarks)
        tf.print("current_gen_landmarks")
        plot_landmarks(generated_frames, current_gen_landmarks)
        tf.print("previous_landmarks")
        plot_landmarks(previous_frames, previous_landmarks)
        tf.print("current_landmarks")
        plot_landmarks(current_frames, current_landmarks)


# Without previous frame
class PreprocessedDataGenerator():

    def __init__(self,
                videos_path,
                generator,
                discriminator,
                generator_loss_function,
                discriminator_loss_function,
                batch_size=1,
                landmark_detector=None,
                repeat=False,
                save_path=None):
        if landmark_detector:
            self.landmark_detector = landmark_detector
        else:
            self.landmark_detector = DlibLandmarksDetector()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss_function = generator_loss_function
        self.discriminator_loss_function = discriminator_loss_function
        self.videos_path = videos_path
        self.n = len(videos_path)
        self.batch_size = batch_size
        self.stop_iteration = False
        self.repeat = repeat
        self.save_path = save_path
        self._restart()
    
    def _restart(self):
        self.current_video_index = 0
        self.current_frame_index = 1
        self._load_current_video()
        self.last_generated_frame = self.current_video[0:1]
        # repeated first frame
        self._first_batch_frames =  None
        self._first_batch_first_frames =  None
        self._first_batch_deformed_frames =  None
        self._first_batch_displacements  =  None
    
    def _load_current_video(self):
        while True:
            self.current_video = np.load(self.videos_path[self.current_video_index])
            if self._current_video_has_two_frames():
                break
            else:
                self.current_video_index += 1
        self.height = self.current_video.shape[1]
        self.width = self.current_video.shape[2]
        self._current_source_video = self.current_video # TODO: change this to the most similar video
        self._landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self.current_video)
        # self._current_source_landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self._current_source_video)
        self._current_source_landmarks = self._landmarks # TODO: change this to the most similar video
        self._deform_current_video()
        self._compute_current_displacements()
    
    @property
    def name(self):
        # return self.videos_path[self.current_video_index].split("/")[-1].split(".")[0]
        return f"{self.current_video_index}_{self.current_frame_index}"

    def _load_next_video(self):
        self.current_frame_index = 1
        self.current_video_index += 1
        self._load_current_video()

    def _first_frame(self):
        return self.current_video[0]

    def _first_landmarks(self):
        return self._landmarks[0]
    
    def _range_current_video(self):
        return range(len(self._current_source_video))

    def _deform_current_video(self):
        self._deformed_frames = np.zeros_like(self.current_video)
        for i in self._range_current_video():
             self._deformed_frames[i], _ = deform(
                self._first_frame(),
                self._first_landmarks(),
                self._current_source_landmarks[0],
                self._current_source_landmarks[i]
            )
    
    def _compute_current_displacements(self):
        self.displacements = compute_displacements_interpolation(
            self._current_source_landmarks[:-1],
            self._current_source_landmarks[1:],
            self.width,
            self.height)
        first_displacement = np.zeros([1, self.height, self.width]) # to match video frames, first displacement is zero
        self.displacements = np.concatenate([first_displacement, self.displacements])
        
    def n_first_frames(self, n):
        return np.array([self._first_frame() for _ in range(n)])
    
    def _has_next_video(self):
        return self.current_video_index < self.n - 1
    
    def _current_video_has_two_frames(self):
        return len(self.current_video) > 2

    def _current_video_has_next_frame(self):
        return self.current_frame_index < len(self.current_video)
    
    def _get_all_inputs(self):
        if self.stop_iteration:
            raise StopIteration
        
        if not self._current_video_has_next_frame() and self._has_next_video():
            self._load_next_video()
        
        begin_batch = self.current_frame_index
        end_batch = begin_batch+self.batch_size
        begin_previous = begin_batch-1
        batch_frames = self.current_video[begin_batch:end_batch]
        batch_landmarks = self._landmarks[begin_batch:end_batch]
        batch_first_frames = self.n_first_frames(len(batch_frames))
        batch_deformed_frames = self._deformed_frames[begin_batch:end_batch]
        batch_displacements = self.displacements[begin_batch:end_batch]
        end_previous = begin_previous+len(batch_frames)
        batch_previous_frames = self.current_video[begin_previous:end_previous]
        batch_previous_landmarks = self._landmarks[begin_previous:end_previous]
        self.current_frame_index += len(batch_frames)
        while len(batch_frames) < self.batch_size:
            if self._has_next_video():
                self._load_next_video()
                if self._current_video_has_two_frames():
                    n_frames = len(batch_frames)
                    n_remaning = self.batch_size - n_frames
                    begin_batch = self.current_frame_index
                    end_batch = begin_batch+n_remaning
                    begin_previous = begin_batch-1
                    batch_frames = np.concatenate([batch_frames, self.current_video[begin_batch:end_batch]])
                    batch_landmarks = np.concatenate([batch_landmarks, self._landmarks[begin_batch:end_batch]])
                    batch_deformed_frames = np.concatenate([batch_deformed_frames, self._deformed_frames[begin_batch:end_batch]])
                    batch_displacements = np.concatenate([batch_displacements, self.displacements[begin_batch:end_batch]])
                    n_added_frames = len(batch_frames) - n_frames
                    batch_first_frames = np.concatenate([batch_first_frames, self.n_first_frames(n_added_frames)])
                    end_previous = begin_previous+n_added_frames
                    batch_previous_frames = np.concatenate([batch_previous_frames, self.current_video[begin_previous:end_previous]])
                    batch_previous_landmarks = np.concatenate([batch_previous_landmarks, self._landmarks[begin_previous:end_previous]])
                    self.current_frame_index += n_added_frames
            else:
                self._restart() # don't stop iteration in the middle of a batch
                if not self.repeat:
                    self.stop_iteration = True
        
        generated_frames = self.generator([batch_first_frames, batch_deformed_frames, batch_displacements])
        previous_generated_frames = np.concatenate([self.last_generated_frame, generated_frames[:-1]])
        self.last_generated_frame = generated_frames[-1:]

        return (batch_previous_frames,
                batch_frames,
                batch_previous_landmarks,
                batch_landmarks,
                previous_generated_frames,
                generated_frames,
                batch_first_frames,
                batch_deformed_frames,
                batch_displacements)

    def generate_next_batch(self):
        _, current_frames, _, _, previous_generated_frames, generated_frames, _, _, _ = self._get_all_inputs()
        return previous_generated_frames, generated_frames, current_frames
    
    def generate_diff_batch(self):
        previous_frames, current_frames, _, _, previous_generated_frames, generated_frames, _, _, _ = self._get_all_inputs()
        return previous_generated_frames, generated_frames - previous_generated_frames, current_frames - previous_frames
    
    def get_deformed_diff_batch(self):
        if self.stop_iteration:
            raise StopIteration
        
        if not self._current_video_has_next_frame() and self._has_next_video():
            self._load_next_video()
        
        begin_batch = self.current_frame_index
        end_batch = begin_batch+self.batch_size
        begin_previous = begin_batch-1
        batch_frames = self.current_video[begin_batch:end_batch]
        batch_deformed_frames = self._deformed_frames[begin_batch:end_batch]
        end_previous = begin_previous+len(batch_frames)
        batch_previous_frames = self.current_video[begin_previous:end_previous]
        batch_previous_deformed_frames = self._deformed_frames[begin_previous:end_previous]
        self.current_frame_index += len(batch_frames)
        while len(batch_frames) < self.batch_size:
            if self._has_next_video():
                self._load_next_video()
                if self._current_video_has_two_frames():
                    n_frames = len(batch_frames)
                    n_remaning = self.batch_size - n_frames
                    begin_batch = self.current_frame_index
                    end_batch = begin_batch+n_remaning
                    begin_previous = begin_batch-1
                    batch_frames = np.concatenate([batch_frames, self.current_video[begin_batch:end_batch]])
                    batch_deformed_frames = np.concatenate([batch_deformed_frames, self._deformed_frames[begin_batch:end_batch]])
                    n_added_frames = len(batch_frames) - n_frames
                    end_previous = begin_previous+n_added_frames
                    batch_previous_frames = np.concatenate([batch_previous_frames, self.current_video[begin_previous:end_previous]])
                    batch_previous_deformed_frames = np.concatenate([batch_previous_deformed_frames, self._deformed_frames[begin_previous:end_previous]])
                    self.current_frame_index += n_added_frames
            else:
                self._restart() # don't stop iteration in the middle of a batch
                if not self.repeat:
                    self.stop_iteration = True
        return (batch_frames - batch_previous_frames), (batch_deformed_frames - batch_previous_deformed_frames)


    def generate_first_batch(self):
        if self._first_batch_frames is None:
            # self._restart()
            (_,
            self._first_batch_frames,
            _,
            _,
            previous_generated_frames,
            generated_frames,
            self._first_batch_first_frames,
            self._first_batch_deformed_frames,
            self._first_batch_displacements) = self._get_all_inputs()
            return previous_generated_frames, generated_frames, self._first_batch_frames
        else:
            generated_frames = self.generator([self._first_batch_first_frames, self._first_batch_deformed_frames, self._first_batch_displacements])
            previous_generated_frames = np.concatenate([self._first_batch_first_frames[0:1], generated_frames[:-1]])
            return previous_generated_frames, generated_frames, self._first_batch_frames
    
    def next_loss(self):
        (previous_frames,
        current_frames,
        previous_landmarks,
        current_landmarks,
        previous_generated_frames,
        generated_frames,
        _, _, _) = self._get_all_inputs()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previous_generated_frames)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames)
        frames_diff = current_frames - previous_frames
        generated_diff = generated_frames - previous_generated_frames
        disc_real_output = self.discriminator([previous_frames, current_frames, frames_diff], training=True)
        disc_generated_output = self.discriminator([previous_generated_frames, generated_frames, generated_diff], training=True)
        disc_loss_value = self.discriminator_loss_function(disc_real_output, disc_generated_output)
        total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value = self.generator_loss_function(
            disc_generated_output,
            previous_generated_frames,
            generated_frames,
            previous_frames,
            current_frames,
            previous_landmarks,
            current_landmarks,
            previous_gen_landmarks,
            current_gen_landmarks
        )        
        return total_gen_loss, disc_loss_value, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value

    def next_plot(self):
        (previous_frames,
        current_frames,
        previous_landmarks,
        current_landmarks,
        previous_generated_frames,
        generated_frames,
        first_frames,
        deformed_frames,
        displacements) = self._get_all_inputs()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previous_generated_frames).numpy()
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames).numpy()
        previous_landmarks = add_boundary_points(previous_landmarks, self.height, self.width)
        current_landmarks = add_boundary_points(current_landmarks, self.height, self.width)
        previous_gen_landmarks = add_boundary_points(previous_gen_landmarks, self.height, self.width)
        current_gen_landmarks = add_boundary_points(current_gen_landmarks, self.height, self.width)
        previous_triangles = previous_landmarks[:, DEFAULT_TRIANGULATION]
        current_triangles = current_landmarks[:, DEFAULT_TRIANGULATION]
        previous_gen_triangles = previous_gen_landmarks[:, DEFAULT_TRIANGULATION]
        current_gen_triangles = current_gen_landmarks[:, DEFAULT_TRIANGULATION]
        frames_diff = current_frames - previous_frames
        generated_diff = generated_frames - previous_generated_frames
        diff_diff = frames_diff - generated_diff
        if self.save_path:
            output_folder = Path(self.save_path) / self.name
            output_folder.mkdir(parents=True, exist_ok=True)
            save_gif(generated_frames, output_folder / "generated.gif")
            save_gif(current_frames, output_folder / "groundtruth.gif")
            save_gif(deformed_frames, output_folder / "deformed.gif")
            save_gif(frames_diff, output_folder / "frames_diff.gif")
            save_gif(generated_diff, output_folder / "generated_diff.gif")
            save_gif(diff_diff, output_folder / "diff_diff.gif")
            np.save(output_folder / "generated.npy", generated_frames)
            np.save(output_folder / "groundtruth.npy", current_frames)
            np.save(output_folder / "deformed.npy", deformed_frames)
            np.save(output_folder / "frames_diff.npy", frames_diff)
            np.save(output_folder / "generated_diff.npy", generated_diff)
            np.save(output_folder / "diff_diff.npy", diff_diff)
            previous_generated_frames_path = output_folder / "previous_generated_frames.pdf"
            generated_frames_path = output_folder / "generated_frames.pdf"
            current_frames_path = output_folder / "current_frames.pdf"
            previous_frames_path = output_folder / "previous_frames.pdf"
            first_frames_path = output_folder / "first_frames.pdf"
            deformed_frames_path = output_folder / "deformed_frames.pdf"
            displacements_path = output_folder / "displacements.pdf"
            previous_gen_landmarks_path = output_folder / "previous_gen_landmarks.pdf"
            current_gen_landmarks_path = output_folder / "current_gen_landmarks.pdf"
            previous_landmarks_path = output_folder / "previous_landmarks.pdf"
            current_landmarks_path = output_folder / "current_landmarks.pdf"
            previous_gen_triangles_path = output_folder / "previous_gen_triangles.pdf"
            current_gen_triangles_path = output_folder / "current_gen_triangles.pdf"
            previous_triangles_path = output_folder / "previous_triangles.pdf"
            current_triangles_path = output_folder / "current_triangles.pdf"
        else:
            previous_generated_frames_path = None
            generated_frames_path = None
            current_frames_path = None
            previous_frames_path = None
            first_frames_path = None
            deformed_frames_path = None
            displacements_path = None
            previous_gen_landmarks_path = None
            current_gen_landmarks_path = None
            previous_landmarks_path = None
            current_landmarks_path = None
            previous_gen_triangles_path = None
            current_gen_triangles_path = None
            previous_triangles_path = None
            current_triangles_path = None
        tf.print("previous_generated_frames")
        plot_normalized_sequence(previous_generated_frames, previous_generated_frames_path)
        tf.print("generated_frames")
        plot_normalized_sequence(generated_frames, generated_frames_path)
        tf.print("current_frames")
        plot_normalized_sequence(current_frames, current_frames_path)
        tf.print("previous_frames")
        plot_normalized_sequence(previous_frames, previous_frames_path)
        tf.print("first_frames")
        plot_normalized_sequence(first_frames, first_frames_path)
        tf.print("deformed_frames")
        plot_normalized_sequence(deformed_frames, deformed_frames_path)
        tf.print("displacements")
        plot_normalized_sequence(displacements, displacements_path)
        tf.print("previous_gen_landmarks")
        plot_landmarks(previous_generated_frames, previous_gen_landmarks, previous_gen_landmarks_path)
        tf.print("current_gen_landmarks")
        plot_landmarks(generated_frames, current_gen_landmarks, current_gen_landmarks_path)
        tf.print("previous_landmarks")
        plot_landmarks(previous_frames, previous_landmarks, previous_landmarks_path)
        tf.print("current_landmarks")
        plot_landmarks(current_frames, current_landmarks, current_landmarks_path)
        tf.print("current_triangles")
        plot_triangles(current_frames, current_triangles, current_triangles_path)
        tf.print("previous_triangles")
        plot_triangles(previous_frames, previous_triangles, previous_triangles_path)
        tf.print("current_gen_triangles")
        plot_triangles(generated_frames, current_gen_triangles, current_gen_triangles_path)
        tf.print("previous_gen_triangles")
        plot_triangles(previous_generated_frames, previous_gen_triangles, previous_gen_triangles_path)

    def next_interpolation_plot(self):
        (previous_frames,
        current_frames,
        _,
        _,
        previous_generated_frames,
        generated_frames,
        _,
        _,
        _) = self._get_all_inputs()
        mid_gen = (previous_generated_frames + generated_frames)/2
        mid_target = (previous_frames + current_frames)/2
        mid_diff = mid_target - mid_gen
        if self.save_path:
            output_folder = Path(self.save_path) / self.name
            output_folder.mkdir(parents=True, exist_ok=True)
            # np.save(output_folder / "mid_gen.npy", mid_gen)
            # np.save(output_folder / "mid_target.npy", mid_target)
            # np.save(output_folder / "mid_diff.npy", mid_diff)
            save_gif(mid_gen, output_folder / "mid_gen.gif")
            save_gif(mid_target, output_folder / "mid_target.gif")
            save_gif(mid_diff, output_folder / "mid_diff.gif")
            mid_gen_path = output_folder / "mid_gen.pdf"
            mid_target_path = output_folder / "mid_target.pdf"
            mid_diff_path = output_folder / "mid_diff.pdf"
        else:
            mid_gen_path = None
            mid_target_path = None
            mid_diff_path = None
        tf.print("mid_gen")
        plot_normalized_sequence(mid_gen, mid_gen_path)
        tf.print("mid_target")
        plot_normalized_sequence(mid_target, mid_target_path)
        tf.print("mid_diff")
        plot_normalized_sequence(mid_diff, mid_diff_path)




class PreprocessedNeutralDataGenerator(PreprocessedDataGenerator):

   
    def _load_current_video(self):
        while True:
            self.current_video = np.load(self.videos_path[self.current_video_index])
            if self._current_video_has_two_frames():
                break
            else:
                self.current_video_index += 1
        self.height = self.current_video.shape[1]
        self.width = self.current_video.shape[2]
        self._current_source_video = self.current_video # TODO: change this to the most similar video
        self._landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self.current_video)
        # self._current_source_landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self._current_source_video)
        self._current_source_landmarks = self._landmarks # TODO: change this to the most similar video
        self._compute_current_displacements()
    
    def _get_all_inputs(self):
        if self.stop_iteration:
            raise StopIteration
        
        if not self._current_video_has_next_frame() and self._has_next_video():
            self._load_next_video()
        
        begin_batch = self.current_frame_index
        end_batch = begin_batch+self.batch_size
        begin_previous = begin_batch-1
        batch_frames = self.current_video[begin_batch:end_batch]
        batch_landmarks = self._landmarks[begin_batch:end_batch]
        batch_first_frames = self.n_first_frames(len(batch_frames))
        batch_displacements = self.displacements[begin_batch:end_batch]
        end_previous = begin_previous+len(batch_frames)
        batch_previous_frames = self.current_video[begin_previous:end_previous]
        batch_previous_landmarks = self._landmarks[begin_previous:end_previous]
        self.current_frame_index += len(batch_frames)
        while len(batch_frames) < self.batch_size:
            if self._has_next_video():
                self._load_next_video()
                if self._current_video_has_two_frames():
                    n_frames = len(batch_frames)
                    n_remaning = self.batch_size - n_frames
                    begin_batch = self.current_frame_index
                    end_batch = begin_batch+n_remaning
                    begin_previous = begin_batch-1
                    batch_frames = np.concatenate([batch_frames, self.current_video[begin_batch:end_batch]])
                    batch_landmarks = np.concatenate([batch_landmarks, self._landmarks[begin_batch:end_batch]])
                    batch_displacements = np.concatenate([batch_displacements, self.displacements[begin_batch:end_batch]])
                    n_added_frames = len(batch_frames) - n_frames
                    batch_first_frames = np.concatenate([batch_first_frames, self.n_first_frames(n_added_frames)])
                    end_previous = begin_previous+n_added_frames
                    batch_previous_frames = np.concatenate([batch_previous_frames, self.current_video[begin_previous:end_previous]])
                    batch_previous_landmarks = np.concatenate([batch_previous_landmarks, self._landmarks[begin_previous:end_previous]])
                    self.current_frame_index += n_added_frames
            else:
                self._restart() # don't stop iteration in the middle of a batch
                if not self.repeat:
                    self.stop_iteration = True
        
        generated_frames = self.generator([batch_first_frames, batch_displacements])
        previous_generated_frames = np.concatenate([self.last_generated_frame, generated_frames[:-1]])
        self.last_generated_frame = generated_frames[-1:]

        return (batch_previous_frames,
                batch_frames,
                batch_previous_landmarks,
                batch_landmarks,
                previous_generated_frames,
                generated_frames,
                batch_first_frames,
                batch_displacements)

    def generate_next_batch(self):
        _, current_frames, _, _, previous_generated_frames, generated_frames, _, _ = self._get_all_inputs()
        return previous_generated_frames, generated_frames, current_frames
    
    def generate_first_batch(self):
        if self._first_batch_frames is None:
            # self._restart()
            (_,
            self._first_batch_frames,
            _,
            _,
            previous_generated_frames,
            generated_frames,
            self._first_batch_first_frames,
            self._first_batch_displacements) = self._get_all_inputs()
            return previous_generated_frames, generated_frames, self._first_batch_frames
        else:
            generated_frames = self.generator([self._first_batch_first_frames, self._first_batch_displacements])
            previous_generated_frames = np.concatenate([self._first_batch_first_frames[0:1], generated_frames[:-1]])
            return previous_generated_frames, generated_frames, self._first_batch_frames
    
    def next_loss(self):
        (previous_frames,
        current_frames,
        previous_landmarks,
        current_landmarks,
        previous_generated_frames,
        generated_frames,
        _, _) = self._get_all_inputs()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previous_generated_frames)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames)
        disc_real_output = self.discriminator([previous_frames, current_frames], training=True)
        disc_generated_output = self.discriminator([previous_generated_frames, generated_frames], training=True)
        disc_loss_value = self.discriminator_loss_function(disc_real_output, disc_generated_output)
        total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value = self.generator_loss_function(
            disc_generated_output,
            previous_generated_frames,
            generated_frames,
            previous_frames,
            current_frames,
            previous_landmarks,
            current_landmarks,
            previous_gen_landmarks,
            current_gen_landmarks
        )        
        return total_gen_loss, disc_loss_value, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss, perceptual_loss_value, interpolation_frame_loss_value, interpolation_landmarks_loss_value

    def next_plot(self):
        (previous_frames,
        current_frames,
        previous_landmarks,
        current_landmarks,
        previous_generated_frames,
        generated_frames,
        first_frames,
        displacements) = self._get_all_inputs()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previous_generated_frames)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames)
        if self.save_path:
            output_folder = Path(self.save_path) / self.name
            output_folder.mkdir(parents=True, exist_ok=True)
            save_gif(generated_frames, output_folder / "generated.gif")
            save_gif(current_frames, output_folder / "groundtruth.gif")
            save_gif(deformed_frames, output_folder / "deformed.gif")
            np.save(output_folder / "generated.npy", generated_frames)
            np.save(output_folder / "groundtruth.npy", current_frames)
            np.save(output_folder / "deformed.npy", deformed_frames)
            previous_generated_frames_path = output_folder / "previous_generated_frames.pdf"
            generated_frames_path = output_folder / "generated_frames.pdf"
            current_frames_path = output_folder / "current_frames.pdf"
            previous_frames_path = output_folder / "previous_frames.pdf"
            first_frames_path = output_folder / "first_frames.pdf"
            deformed_frames_path = output_folder / "deformed_frames.pdf"
            displacements_path = output_folder / "displacements.pdf"
            previous_gen_landmarks_path = output_folder / "previous_gen_landmarks.pdf"
            current_gen_landmarks_path = output_folder / "current_gen_landmarks.pdf"
            previous_landmarks_path = output_folder / "previous_landmarks.pdf"
            current_landmarks_path = output_folder / "current_landmarks.pdf"
        else:
            previous_generated_frames_path = None
            generated_frames_path = None
            current_frames_path = None
            previous_frames_path = None
            first_frames_path = None
            deformed_frames_path = None
            displacements_path = None
            previous_gen_landmarks_path = None
            current_gen_landmarks_path = None
            previous_landmarks_path = None
            current_landmarks_path = None
        tf.print("previous_generated_frames")
        plot_normalized_sequence(previous_generated_frames, previous_generated_frames_path)
        tf.print("generated_frames")
        plot_normalized_sequence(generated_frames, generated_frames_path)
        tf.print("current_frames")
        plot_normalized_sequence(current_frames, current_frames_path)
        tf.print("previous_frames")
        plot_normalized_sequence(previous_frames, previous_frames_path)
        tf.print("first_frames")
        plot_normalized_sequence(first_frames, first_frames_path)
        tf.print("displacements")
        plot_normalized_sequence(displacements, displacements_path)
        tf.print("previous_gen_landmarks")
        plot_landmarks(previous_generated_frames, previous_gen_landmarks, previous_gen_landmarks_path)
        tf.print("current_gen_landmarks")
        plot_landmarks(generated_frames, current_gen_landmarks, current_gen_landmarks_path)
        tf.print("previous_landmarks")
        plot_landmarks(previous_frames, previous_landmarks, previous_landmarks_path)
        tf.print("current_landmarks")
        plot_landmarks(current_frames, current_landmarks, current_landmarks_path)
