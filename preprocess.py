
from landmarks import DlibLandmarksDetector
import numpy as np
from landmarks import compute_displacements_interpolation
from transformation import deform

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


class PreprocessedFrameDataGenerator(FrameDataGenerator): # TODO: adapat to only 1 frame input

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
            import tensorflow as tf
            tf.print("End of videos")
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

        total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss = self.generator_loss_function(
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
        return total_gen_loss, disc_loss_value, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss
    
    def restart(self):
        return self._restart()
    
    def __next__(self):
        yield self.generate_next_frame()
    
