
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
        for i in range(n):
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

    def __init__(self, videos_path, batch_size=1):
        super().__init__(videos_path, batch_size)
        self.landmark_detector = DlibLandmarksDetector(batch_size)
    
    def _load_current_video(self):
        super()._load_current_video()
        self._deformed_frame = self.current_video[0] # first frame as input
        self._current_source_video = self.current_video # TODO: change this to the most similar video
        self._landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self.current_video)
        self._current_source_landmarks = self.landmark_detector.preprocess_and_detect_landmarks_numpy(self._current_source_video)
        self._deformed_landmarks = self._landmarks[0]
    
    def _load_next_video(self):
        super()._load_next_video()
        self.height = self.current_video.shape[1]
        self.width = self.current_video.shape[2]
    
    def _current_landmarks(self):
        return self._landmarks[self.current_frame_index]
    
    def _next_landmarks(self):
        return self._landmarks[self.current_frame_index+1]
    
    def _previous_landmarks(self):
        return self._landmarks[self.current_frame_index-1]

    def _deform_current_frame(self): # TODO: put in next the change of _deformed_landmarks and _deformed_frame
        self._deformed_frame, self._deformed_landmarks = deform(
            self._deformed_frame,
            self._deformed_landmarks,
            self._current_source_landmarks[self.current_frame_index-1],
            self._current_source_landmarks[self.current_frame_index]
        )
    
    def _get_displacements(self):
        return compute_displacements_interpolation(
            self._current_source_landmarks[self.current_frame_index-1],
            self._current_source_landmarks[self.current_frame_index],
            self.width,
            self.height,
            1)

    def _get_current_frames(self):
        deformed_frame = self._deformed_frame 
        self._deform_current_frame() # TODO: put in next
        return (
            self._previous_frame(),
            self._current_frame(),
            self._next_frame(),
            deformed_frame,
            self._get_displacements(),
            self._previous_landmarks,
            self._landmarks,
            self._next_landmarks
        )

    def _get_batch(self):
        batch_previous_frame = []
        batch_current_frame = []
        batch_next_frame = []
        batch_deformed_frame = []
        batch_displacements = []
        batch_previous_landmarks = []
        batch_current_landmarks = []
        batch_next_landmarks = []
        for i in range(self.batch_size):
            previous_frame,current_frame, next_frame = self._get_next_frames()
            batch_previous_frame.append(previous_frame)
            batch_current_frame.append(current_frame)
            batch_next_frame.append(next_frame)
            batch_deformed_frame.append(self._deformed_frame)
            batch_displacements.append(self._get_displacements())
            batch_previous_landmarks.append(self._previous_landmarks)
            batch_current_landmarks.append(self._landmarks)
            batch_next_landmarks.append(self._next_landmarks)
        return (
            np.array(batch_previous_frame),
            np.array(batch_current_frame),
            np.array(batch_next_frame),
            np.array(batch_deformed_frame),
            np.array(batch_displacements),
            np.array(batch_previous_landmarks),
            np.array(batch_current_landmarks),
            np.array(batch_next_landmarks)
        )
