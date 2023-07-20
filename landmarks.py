from settings.facial import LOW_DEFAUlT_LANDMARKS, HIGH_DEFAULT_LANDMARKS
import numpy as np
import tensorflow as tf
import mediapipe as mp
import face_recognition
from scipy.interpolate import LinearNDInterpolator
from normalize import images_from_normalized


# TODO: change to worst case landmarks
# e.g. maximize the distance between landmarks for odd and even frames
def get_default_landmarks(i=None):
    landmarks = np.array(LOW_DEFAUlT_LANDMARKS, dtype=np.float32)
    if (i is not None) and (i % 2 == 1):
        landmarks = np.array(HIGH_DEFAULT_LANDMARKS, dtype=np.float32)
    return landmarks

class LandmarkDetector():

    def __init__(self, num_landmarks=68):
        self.holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        self.num_landmarks = num_landmarks
    
    def preprocess_and_detect_landmarks_numpy(self, images):
        raise NotImplementedError

    def preprocess_and_detect_landmarks(self, images):
        batch_size = images.shape[0]
        results = tf.numpy_function(self.preprocess_and_detect_landmarks_numpy, [images], Tout=[tf.float32])
        results = tf.convert_to_tensor(results, dtype=tf.float32)
        results.set_shape((batch_size, self.num_landmarks, 2))
        return results 


class MediapipeLandmarkDetector(LandmarkDetector):

    def preprocess_and_detect_landmarks_numpy(self, images):
        images =  images_from_normalized(images)
        images = np.array(images, dtype=np.uint8)
        all_landmarks = []
        for i, image in enumerate(images):
            landmarks = []
            results = self.holistic_model.process(image)
            if results.face_landmarks is None:
                landmarks = get_default_landmarks(i)
            else:
                for landmark in results.face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks, dtype=np.float32)
                landmarks = landmarks[self.landmark_points_68]
            all_landmarks.append(landmarks)
        return np.array(all_landmarks, dtype=np.float32)


def dlib_landmarks_to_array(dlib_landmarks):
    landmarks = []
    for current_landmarks in dlib_landmarks:
        landmarks.append([current_landmarks.x, current_landmarks.y])
    return np.array(landmarks)

class DlibLandmarksDetector(LandmarkDetector):

    def preprocess_and_detect_landmarks_numpy(self, images):
        clip_landmarks = []
        for i, frame in enumerate(images):
            frame =  images_from_normalized(frame)
            frame = np.array(frame, dtype=np.uint8)
            # Only considering the first face
            faces = face_recognition.api._raw_face_landmarks(face_image=frame)
            if len(faces) > 0:
                landmarks_array = dlib_landmarks_to_array(faces[0].parts())
            else:
                height, width = frame.shape[:2]
                landmarks_array = get_default_landmarks(i)
                landmarks_array = landmarks_array * np.array([width, height])
            clip_landmarks.append(landmarks_array)
        clip_landmarks = np.array(clip_landmarks)
        return np.ndarray.astype(clip_landmarks, dtype=np.float32)


def compute_displacements_interpolation(points_a, points_b, image_width, image_height, fill_value=0):
    all_displacements_map = []
    for pa, pb in zip(points_a, points_b):
        displacements_values = np.linalg.norm(pb - pa, axis=1)
        interpolator_a = LinearNDInterpolator(pa, displacements_values, fill_value=fill_value)
        interpolator_b = LinearNDInterpolator(pb, displacements_values, fill_value=fill_value)
        X = np.arange(0, image_height)
        Y = np.arange(0, image_width)
        X, Y = np.meshgrid(X, Y)
        displacements_map_a = interpolator_a(X, Y)
        displacements_map_b = interpolator_b(X, Y)
        displacements_map = np.mean([displacements_map_a, displacements_map_b], axis=0)
        all_displacements_map.append(displacements_map)
    return np.array(all_displacements_map, dtype=np.float32)

@tf.function
def get_tensors_displacements(points_a, points_b, image_width, image_height, batch_size):
    [results,] = tf.numpy_function(compute_displacements_interpolation, [points_a, points_b, image_height, image_width, batch_size], Tout=[tf.float32])
    results = tf.convert_to_tensor(results, dtype=tf.float32)
    results.set_shape((batch_size, image_height, image_width))
    # add batch and channel dimensions
    # results = tf.expand_dims(results, axis=0)
    # results = tf.expand_dims(results, axis=-1)
    return results

def get_tensor_display_displacements_from_images(images_a, images_b, landmark_detector, image_width, image_height, batch_size):
    points_a = landmark_detector.preprocess_and_detect_landmarks(images_a)
    points_b = landmark_detector.preprocess_and_detect_landmarks(images_b)
    return get_tensors_displacements(points_a, points_b, image_width, image_height, batch_size)

def add_boundary_points(points, height, width):
    bp = np.array(
        [(0, 0),
         (width / 6, 0),
         (width / 3, 0),
         (width / 2, 0),
         (2 * width / 3, 0),
         (5 * width / 6, 0),
         (width - 1, 0),
         (width - 1, height / 6),
         (width - 1, height / 3),
         (width - 1, height / 2),
         (width - 1, 2 * height / 3),
         (width - 1, 5 * height / 6),
         (width - 1, height - 1),
         (width / 6, height - 1),
         (width / 3, height - 1),
         (width / 2, height - 1),
         (2 * width / 3, height - 1),
         (5 * width / 6, height - 1),
         (0, height - 1),
         (0, height / 6),
         (0, height / 3),
         (0, height / 2),
         (0, 2 * height / 3),
         (0, 5 * height / 6),
         ]).astype(int)
    if len(points.shape) == 2:
        all_points = np.concatenate((points, bp), axis=0)
        return np.ndarray.astype(all_points, dtype=np.float32)
    else:
        all_points  = []
        for i in range(points.shape[0]):
            p = np.concatenate((points[i], bp), axis=0)
            all_points.append(p)
        return np.array(all_points, dtype=np.float32)
