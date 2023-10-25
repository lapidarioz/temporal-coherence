
from landmarks import DlibLandmarksDetector, add_boundary_points
import numpy as np
import tensorflow as tf
from landmarks import compute_displacements_interpolation
from transformation import deform
from plot import plot_normalized_sequence, plot_landmarks, plot_triangles, save_gif
from pathlib import Path
from settings.facial import DEFAULT_TRIANGULATION
from video_processing import blend_frames, increase_frame_rate

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
                save_path=None,
                blend_deformation=False):
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
        self.blend_deformation = blend_deformation
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
        self._first_previous_generated_frames =  None
        self._generated_frames_to_save = self.current_video[0:0]
        self._current_video_to_save = self.current_video[0:0]
        self._deformed_frames_to_save = self.current_video[0:0]
        self._blended_frames_to_save = self.current_video[0:0]
        self._previous_generated_frames = self.n_first_frames(self.batch_size)
    
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
        return f"{self.current_video_index}_{self.current_frame_index}"
    
    def _get_path_unique_parts(self):
        video_path = Path(self.videos_path[self.current_video_index])
        return video_path.parts[3:-1]
    
    def current_video_relative_path(self):
        return Path(*self._get_path_unique_parts())
    
    def path_id(self):
        return "_".join(self._get_path_unique_parts())

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
                self._current_source_video[i],
                self._first_landmarks(),
                self._current_source_landmarks[0],
                self._current_source_landmarks[i]
            )
        if self.blend_deformation:
            previous_deformed_frames = np.concatenate((self._deformed_frames[0:1], self._deformed_frames[:-1]))
            next_deformed_frames = np.concatenate((self._deformed_frames[1:], self._deformed_frames[-1:]))
            self._deformed_frames = blend_frames(previous_deformed_frames, self._deformed_frames, next_deformed_frames)
    
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
    
    # TODO: only one method for all types of inputs
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
        
        generated_frames = self.generator([batch_first_frames, self._previous_generated_frames, batch_deformed_frames, batch_displacements])
        previous_generated_frames = np.concatenate([self.last_generated_frame, generated_frames[:-1]])
        self._previous_generated_frames = previous_generated_frames
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
    
     # TODO: only one method for all types of inputs
    def _get_all_inputs_deformed(self):
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
                    batch_landmarks = np.concatenate([batch_landmarks, self._landmarks[begin_batch:end_batch]])
                    batch_deformed_frames = np.concatenate([batch_deformed_frames, self._deformed_frames[begin_batch:end_batch]])
                    batch_displacements = np.concatenate([batch_displacements, self.displacements[begin_batch:end_batch]])
                    n_added_frames = len(batch_frames) - n_frames
                    batch_first_frames = np.concatenate([batch_first_frames, self.n_first_frames(n_added_frames)])
                    end_previous = begin_previous+n_added_frames
                    batch_previous_frames = np.concatenate([batch_previous_frames, self.current_video[begin_previous:end_previous]])
                    batch_previous_landmarks = np.concatenate([batch_previous_landmarks, self._landmarks[begin_previous:end_previous]])
                    batch_previous_deformed_frames = np.concatenate([batch_previous_deformed_frames, self._deformed_frames[begin_previous:end_previous]])
                    self.current_frame_index += n_added_frames
            else:
                self._restart() # don't stop iteration in the middle of a batch
                if not self.repeat:
                    self.stop_iteration = True
        
        generated_frames = self.generator([batch_first_frames, batch_previous_deformed_frames, batch_deformed_frames, batch_displacements])
        previous_generated_frames = np.concatenate([self.last_generated_frame, generated_frames[:-1]])
        self._previous_generated_frames = previous_generated_frames
        self.last_generated_frame = generated_frames[-1:]

        return (batch_previous_frames,
                batch_frames,
                batch_previous_landmarks,
                batch_landmarks,
                previous_generated_frames,
                generated_frames,
                batch_first_frames,
                batch_deformed_frames,
                batch_displacements,
                batch_previous_deformed_frames)

    def generate_next_batch(self):
        _, current_frames, _, _, previous_generated_frames, generated_frames, _, _, _ = self._get_all_inputs()
        return generated_frames, current_frames

    
    def generate_diff_batch(self):
        previous_frames, current_frames, _, _, previous_generated_frames, generated_frames, _, _, _ = self._get_all_inputs()
        return previous_generated_frames, generated_frames - previous_generated_frames, current_frames - previous_frames
    
    def blended_diff_batch(self):
        previous_frames, current_frames, _, _, _, _, _, _, _, blended_frames, previous_blended_frames = self._get_all_inputs_blended()
        return previous_blended_frames, blended_frames - previous_blended_frames, current_frames - previous_frames
    
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
            self._first_previous_generated_frames,
            generated_frames,
            self._first_batch_first_frames,
            self._first_batch_deformed_frames,
            self._first_batch_displacements) = self._get_all_inputs()
            return self._first_previous_generated_frames, generated_frames, self._first_batch_frames
        else:
            generated_frames = self.generator([self._first_batch_first_frames, self._first_previous_generated_frames, self._first_batch_deformed_frames, self._first_batch_displacements])
            self._first_previous_generated_frames = np.concatenate([self._first_batch_first_frames[0:1], generated_frames[:-1]])
            return self._first_previous_generated_frames, generated_frames, self._first_batch_frames
    
    def generate_first_batch_deformed(self):
        if self._first_batch_frames is None:
            # self._restart()
            (_,
            self._first_batch_frames,
            _,
            _,
            self._first_previous_generated_frames,
            generated_frames,
            self._first_batch_first_frames,
            self._first_batch_deformed_frames,
            self._first_batch_displacements,
            self._first_batch_previous_deformed_frames) = self._get_all_inputs_deformed()
            return self._first_previous_generated_frames, generated_frames, self._first_batch_frames
        else:
            generated_frames = self.generator([self._first_batch_first_frames, self._first_batch_previous_deformed_frames, self._first_batch_deformed_frames, self._first_batch_displacements])
            self._first_previous_generated_frames = np.concatenate([self._first_batch_first_frames[0:1], generated_frames[:-1]])
            return self._first_previous_generated_frames, generated_frames, self._first_batch_frames
    
    # TODO: only one next loss method
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
        total_gen_loss, other_loss = self.generator_loss_function(
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
        return total_gen_loss, disc_loss_value, other_loss
    
    # TODO: only one next loss method
    def next_loss_deformed(self):
        (previous_frames,
        current_frames,
        previous_landmarks,
        current_landmarks,
        previous_generated_frames,
        generated_frames,
        _, _, _, _) = self._get_all_inputs_deformed()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previous_generated_frames)
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames)
        frames_diff = current_frames - previous_frames
        generated_diff = generated_frames - previous_generated_frames
        disc_real_output = self.discriminator([previous_frames, current_frames, frames_diff], training=True)
        disc_generated_output = self.discriminator([previous_generated_frames, generated_frames, generated_diff], training=True)
        disc_loss_value = self.discriminator_loss_function(disc_real_output, disc_generated_output)
        total_gen_loss, other_loss = self.generator_loss_function(
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
        return total_gen_loss, disc_loss_value, other_loss

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
    
    def save_gifs(self):
        (_,
        current_frames,
        previous_landmarks,
        current_landmarks,
        previous_generated_frames,
        generated_frames,
        _,
        _,
        _) = self._get_all_inputs()
        previous_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(previous_generated_frames).numpy()
        current_gen_landmarks = self.landmark_detector.preprocess_and_detect_landmarks(generated_frames).numpy()
        previous_landmarks = add_boundary_points(previous_landmarks, self.height, self.width)
        current_landmarks = add_boundary_points(current_landmarks, self.height, self.width)
        previous_gen_landmarks = add_boundary_points(previous_gen_landmarks, self.height, self.width)
        current_gen_landmarks = add_boundary_points(current_gen_landmarks, self.height, self.width)
        if self.save_path:
            output_folder = Path(self.save_path) / self.name
        else:
            raise ValueError("save_path must be set")
        output_folder.mkdir(parents=True, exist_ok=True)
        save_gif(generated_frames, output_folder / "generated.gif")
        save_gif(current_frames, output_folder / "groundtruth.gif")

    # TODO: refactor to make only one generate video method
    def save_next_video(self):
        previous_video_index = self.current_video_index
        previous_video_path = self.current_video_relative_path()
        (_,
        current_frames,
        _,
        _,
        _,
        generated_frames,
        _,
        deformed_frames,
        _) = self._get_all_inputs()

        if self.current_video_index == previous_video_index:
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_frames])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames])
            self._deformed_frames_to_save = np.concatenate([self._deformed_frames_to_save, deformed_frames])
        else:
            # TODO: fix when video has less than batch_size frames
            split_position = self.batch_size - self.current_frame_index + 1
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_frames[:split_position]])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames[:split_position]])
            self._deformed_frames_to_save = np.concatenate([self._deformed_frames_to_save, deformed_frames[:split_position]])
            # save previous videos
            if self.save_path:
                output_folder = Path(self.save_path) / previous_video_path
            else:
                raise ValueError("save_path must be set")
            output_folder.mkdir(parents=True, exist_ok=True)
            save_gif(self._generated_frames_to_save, output_folder / f"{previous_video_index}_generated.gif")
            save_gif(self._current_video_to_save, output_folder / f"{previous_video_index}_groundtruth.gif")
            save_gif(self._deformed_frames_to_save, output_folder / f"{previous_video_index}_deformed.gif")
            # store current video frames
            self._generated_frames_to_save = generated_frames[split_position:]
            self._current_video_to_save = current_frames[split_position:]
            self._deformed_frames_to_save = deformed_frames[split_position:]
        
    # TODO: refactor to make only one generate video method
    def save_next_video_more_frames(self):
        previous_video_index = self.current_video_index
        previous_video_path = self.current_video_relative_path()
        (_,
        current_frames,
        _,
        _,
        _,
        generated_frames,
        _,
        deformed_frames,
        _) = self._get_all_inputs()

        if self.current_video_index == previous_video_index:
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_frames])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames])
            self._deformed_frames_to_save = np.concatenate([self._deformed_frames_to_save, deformed_frames])
        else:
            # TODO: fix when video has less than batch_size frames
            split_position = self.batch_size - self.current_frame_index + 1
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_frames[:split_position]])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames[:split_position]])
            self._deformed_frames_to_save = np.concatenate([self._deformed_frames_to_save, deformed_frames[:split_position]])
            # save previous videos
            if self.save_path:
                output_folder = Path(self.save_path) / previous_video_path
            else:
                raise ValueError("save_path must be set")
            output_folder.mkdir(parents=True, exist_ok=True)
            save_gif(self._generated_frames_to_save, output_folder / f"{previous_video_index}_generated.gif")
            more_frames = increase_frame_rate(self._generated_frames_to_save)
            save_gif(more_frames, output_folder / f"{previous_video_index}_generated_more_frames.gif")
            save_gif(self._current_video_to_save, output_folder / f"{previous_video_index}_groundtruth.gif")
            save_gif(self._deformed_frames_to_save, output_folder / f"{previous_video_index}_deformed.gif")
            # store current video frames
            self._generated_frames_to_save = generated_frames[split_position:]
            self._current_video_to_save = current_frames[split_position:]
            self._deformed_frames_to_save = deformed_frames[split_position:]
    
    # TODO: refactor to make only one generate video method
    def save_next_blended_video(self):
        previous_video_index = self.current_video_index
        previous_video_path = self.current_video_relative_path()
        (_,
        current_frames,
        _,
        _,
        _,
        generated_frames,
        _,
        deformed_frames,
        _,
        blend_frames,
        _) = self._get_all_inputs_blended()

        if self.current_video_index == previous_video_index:
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_frames])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames])
            self._deformed_frames_to_save = np.concatenate([self._deformed_frames_to_save, deformed_frames])
            self._blended_frames_to_save = np.concatenate([self._blended_frames_to_save, blend_frames])
        else:
            # TODO: fix when video has less than batch_size frames
            split_position = self.batch_size - self.current_frame_index + 1
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_frames[:split_position]])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames[:split_position]])
            self._deformed_frames_to_save = np.concatenate([self._deformed_frames_to_save, deformed_frames[:split_position]])
            self._blended_frames_to_save = np.concatenate([self._blended_frames_to_save, blend_frames[:split_position]])
            # save previous videos
            if self.save_path:
                output_folder = Path(self.save_path) / previous_video_path
            else:
                raise ValueError("save_path must be set")
            output_folder.mkdir(parents=True, exist_ok=True)
            save_gif(self._generated_frames_to_save, output_folder / f"{previous_video_index}_generated.gif")
            save_gif(self._current_video_to_save, output_folder / f"{previous_video_index}_groundtruth.gif")
            save_gif(self._deformed_frames_to_save, output_folder / f"{previous_video_index}_deformed.gif")
            save_gif(self._blended_frames_to_save, output_folder / f"{previous_video_index}_blended.gif")
            # store current video frames
            self._generated_frames_to_save = generated_frames[split_position:]
            self._current_video_to_save = current_frames[split_position:]
            self._deformed_frames_to_save = deformed_frames[split_position:]
            self._blended_frames_to_save = blend_frames[split_position:]
    
    # TODO: refactor to make only one generate video method
    def generate_next_video(self):
        previous_video_index = self.current_video_index
        previous_video_path_id = self.path_id()
        (_,
        current_frames,
        _,
        _,
        _,
        _,
        _,
        _,
        _) = self._get_all_inputs()

        if self.current_video_index == previous_video_index:
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames])
            return None, None, None
        else:
            # TODO: fix when video has less than batch_size frames
            split_position = self.batch_size - self.current_frame_index + 1
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames[:split_position]])
            # store current video frames
            self._current_video_to_save = current_frames[split_position:]
            return self._current_video_to_save, self._generated_frames_to_save, previous_video_path_id
    

    # TODO: refactor to make only one generate video method
    def generate_next_blended_video(self):
        previous_video_index = self.current_video_index
        previous_video_path_id = self.path_id()
        (_,
        current_frames,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        blend_frames,
        _) = self._get_all_inputs_blended()

        if self.current_video_index == previous_video_index:
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames])
            self._blended_frames_to_save = np.concatenate([self._blended_frames_to_save, blend_frames])
            return None, None, None
        else:
            # TODO: fix when video has less than batch_size frames
            split_position = self.batch_size - self.current_frame_index + 1
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_frames[:split_position]])
            self._blended_frames_to_save = np.concatenate([self._blended_frames_to_save, blend_frames[:split_position]])
            # store current video frames
            self._current_video_to_save = current_frames[split_position:]
            self._blended_frames_to_save = blend_frames[split_position:]
            return self._current_video_to_save, self._blended_frames_to_save, previous_video_path_id
    
    # TODO: refactor to make only one generate video method
    def generate_diff_video(self):
        previous_video_index = self.current_video_index
        previous_video_path_id = self.path_id()
        previous_frames, current_frames, _, _, previous_generated_frames, generated_frames, _, _, _ = self._get_all_inputs()

        if self.current_video_index == previous_video_index:
            generated_diff = generated_frames - previous_generated_frames
            current_diff = current_frames - previous_frames
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_diff])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_diff])
            return None, None, None
        else:
            # TODO: fix when video has less than batch_size frames
            split_position = self.batch_size - self.current_frame_index + 1
            generated_diff = generated_frames[:split_position] - previous_generated_frames[:split_position]
            current_diff = current_frames[:split_position] - previous_frames[:split_position]
            self._generated_frames_to_save = np.concatenate([self._generated_frames_to_save, generated_diff])
            self._current_video_to_save = np.concatenate([self._current_video_to_save, current_diff])
            # store current video frames
            self._generated_frames_to_save = generated_frames[split_position:] - previous_generated_frames[split_position:]
            self._current_video_to_save = current_frames[split_position:] - previous_frames[split_position:]
        return self._current_video_to_save, self._generated_frames_to_save, previous_video_path_id
    
    def _get_all_inputs_blended(self):
        (batch_previous_frames,
        batch_frames,
        batch_previous_landmarks,
        batch_landmarks,
        previous_generated_frames,
        generated_frames,
        batch_first_frames,
        batch_deformed_frames,
        batch_displacements) = self._get_all_inputs()

        next_generated_frames = np.concatenate([generated_frames[1:], generated_frames[-1:]])
        previous_previous_generated_frames = np.concatenate([previous_generated_frames[0:1], previous_generated_frames[:-1]])
        blended_frames = blend_frames(previous_generated_frames, generated_frames, next_generated_frames)
        previous_blended_frames = blend_frames(previous_previous_generated_frames, previous_generated_frames, generated_frames)

        return (batch_previous_frames,
        batch_frames,
        batch_previous_landmarks,
        batch_landmarks,
        previous_generated_frames,
        generated_frames,
        batch_first_frames,
        batch_deformed_frames,
        batch_displacements,
        blended_frames,
        previous_blended_frames)

    
    def blended_next_batch(self):
        _, current_frames, _, _, _, _, _, _, _, blended_frames, _ = self._get_all_inputs_blended()
        return blended_frames, current_frames
    
    def deformed_next_batch(self):
        (_,
        batch_frames,
        _,
        _,
        _,
        _,
        _,
        batch_deformed_frames,
        _) = self._get_all_inputs()
        return batch_deformed_frames, batch_frames
