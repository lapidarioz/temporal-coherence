import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model

import os
from pathlib import Path
import time
import datetime
import numpy as np
import cv2 
from itertools import combinations

from matplotlib import pyplot as plt
from IPython.display import display, clear_output, update_display
from tqdm.notebook import tqdm

from joblib import Memory
from joblib import Parallel, delayed
import face_recognition
import random
from scipy.interpolate import LinearNDInterpolator
from scipy import stats

import mediapipe as mp

from fid import get_frechet_inception_distance
import pandas as pd

# Normalizing the images to [-1, 1]
def normalize(sequence):
  return (sequence / 127.5) - 1

# Undo normalizing the images to [0, 255]
@tf.function
def normalized_sequence_to_images(sequence):
  return tf.cast((sequence + 1) * 127.5, tf.uint8)


def plot_sample_sequence(sample_sequence):
    plot_sequence = list(sample_sequence)
    n_frames = len(plot_sequence)
    if n_frames > 20:
        plot_sequence = plot_sequence[:20]
        n_frames = 20
    rows = ((n_frames-1) // 10)+1
    cols = max(n_frames % 10, 10)
    fig = plt.figure(figsize=(20, 4))
    for i, im in enumerate(plot_sequence):
        ax = fig.add_subplot(rows,cols,i+1)
        ax.imshow(im) 
        ax.axis('off')
    plt.show()
    plt.close()

def plot_frame(frame, title=None):
    plt.imshow(frame)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()

def plot_normalized_frame(frame, title=None):
    plot_frame(normalized_sequence_to_images(frame), title)

def sequence_generator(sample_sequence):
    for i in range(sample_sequence.shape[0]):
        yield sample_sequence[i, ...]

def plot_sequence(sample_sequence):
    plot_sample_sequence(sequence_generator(sample_sequence))

def plot_normalized_sequence(sample_sequence):
    plot_sequence(normalized_sequence_to_images(sample_sequence))

def inverse_sequence_generator(sample_sequence):
    for i in range(sample_sequence.shape[-1]):
        yield sample_sequence[..., i]

def plot_inverse_sequence(sample_sequence):
    plot_sample_sequence(inverse_sequence_generator(sample_sequence))

def plot_normalized_inverse_sequence(sample_sequence):
    plot_inverse_sequence(normalized_sequence_to_images(sample_sequence))

def plot_normalized_sequence(sample_sequence):
    plot_sequence(normalized_sequence_to_images(sample_sequence))

def plot_sequence_from_tensor(sample_sequence):
    if len(sample_sequence.shape) == 5 and sample_sequence.shape[0] == 1:
        plot_sequence(sample_sequence[0])
    else:
        raise ValueError("The tensor must be of shape (1, frames, height, width, channels)")

def plot_sequence_from_normalized_tensor(sample_sequence):
    plot_sequence_from_tensor(normalized_sequence_to_images(sample_sequence))

class FrameDataGenerator():
    
    def __init__(self, videos_path, batch_size=1):
        self.videos_path = videos_path
        self.n = len(videos_path)
        self._restart()
        self.batch_size = batch_size
    
    def _load_current_video(self):
        self.current_video = np.load(self.videos_path[self.current_video_index])
    
    def __iter__(self):
        return self
    
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
    
    def _get_batch(self):
        batch_previous_frame = []
        batch_current_frame = []
        batch_next_frame = []
        for i in range(self.batch_size):
            previous_frame, current_frame, next_frame = self._get_current_frames()
            batch_previous_frame.append(previous_frame)
            batch_current_frame.append(current_frame)
            batch_next_frame.append(next_frame)
        return np.array(batch_previous_frame), np.array(batch_current_frame), np.array(batch_next_frame)


    def __next__(self): # TODO: fix to consider batch size
        self.current_frame_index += 1 # Frame index start at one because we have to return the previous frame
        if not self._has_current_video():
            self._end_of_videos()
        while not self._current_video_has_next_frame():
            if self._has_next_video():
                self._load_next_video()
            else:
                self._end_of_videos()
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



def downsample(filters, size, strides=2, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, strides=2, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def Discriminator(img_width, img_height, n_channels):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], name='current_frame')
  tar = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], name='folowing_frame')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, 2, apply_batchnorm=False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4, 2)(down1)  # (batch_size, 64, 64, 128)
  # down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 2, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


def Generator(img_width, img_height, n_channels):
    previous = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='previous_frame')
    # warped = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='warped_frame')
    # neutral = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='neutral_frame')
    # distances = tf.keras.layers.Input(shape=[img_width, img_height, 1], dtype=tf.float64, name='distances')


    down_stack = [
        downsample(128, 4, 2, apply_batchnorm=False),  # (batch_size, frames_group_size, 32, 32, 128)
        downsample(128, 4, 2),  # (batch_size, frames_group_size, 16, 16, 128)
        downsample(256, 4, 2),  # (batch_size, frames_group_size, 8, 8, 256)
        downsample(512, 4, 2),  # (batch_size, frames_group_size, 4, 4, 512)
        downsample(512, 4, 2),  # (batch_size, frames_group_size, 2, 2, 512)
    ]

    up_stack = [
        upsample(512, 4, 2, apply_dropout=False),  # (batch_size, frames_group_size, 4, 4, 512)
        # upsample(512, 4, 2, apply_dropout=True),  # (batch_size, frames_group_size, 4, 4, 512)
        upsample(256, 4, 2),  # (batch_size, frames_group_size, 8, 8, 256)
        upsample(128, 4, 2),  # (batch_size, frames_group_size, 16, 16, 256)
        upsample(128, 4, 2),  # (batch_size, frames_group_size, 32, 32, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(n_channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, , frames_group_size, 64, 64, 3)

    # x = tf.keras.layers.concatenate([warped, neutral, distances, previous])
    x = tf.keras.layers.concatenate([previous])

    # Downsampling through the model
    skips = []
    for down in down_stack:
    x = down(x)
    skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[
        # warped,
        # neutral,
        # distances,
        previous
        ], outputs=x)

# TODO: change to worst case landmarks
# e.g. maximize the distance between landmarks for odd and even frames
def get_default_landmarks():
    landmarks = np.array([
        [0.29012445, 0.46152872],
        [0.2892933 , 0.5601992 ],
        [0.2931674 , 0.60794365],
        [0.3160373 , 0.7112384 ],
        [0.33694965, 0.75203294],
        [0.3611762 , 0.7809247 ],
        [0.41771138, 0.82153064],
        [0.47449082, 0.84470284],
        [0.5121861 , 0.84691584],
        [0.54917306, 0.8418585 ],
        [0.600678  , 0.81461644],
        [0.64821094, 0.77014446],
        [0.6676827 , 0.74018675],
        [0.6844234 , 0.69848907],
        [0.7026569 , 0.5953291 ],
        [0.70517784, 0.54816395],
        [0.7053476 , 0.4507432 ],
        [0.31499407, 0.4301809 ],
        [0.3496217 , 0.41683587],
        [0.3800801 , 0.4043182 ],
        [0.41894913, 0.40356678],
        [0.46308845, 0.4068857 ],
        [0.55364776, 0.40440607],
        [0.59623   , 0.39843333],
        [0.63251084, 0.39682683],
        [0.6602349 , 0.40741068],
        [0.6877083 , 0.419833  ],
        [0.5091903 , 0.4663502 ],
        [0.51167095, 0.5154015 ],
        [0.5140266 , 0.55779254],
        [0.51499516, 0.5852076 ],
        [0.47420424, 0.62131006],
        [0.49072832, 0.6327596 ],
        [0.51303864, 0.63169634],
        [0.53469867, 0.6317114 ],
        [0.55023676, 0.61953837],
        [0.36748627, 0.49156916],
        [0.389509  , 0.47794613],
        [0.42158607, 0.47474214],
        [0.45398962, 0.49241695],
        [0.4244432 , 0.50182253],
        [0.3926721 , 0.5021153 ],
        [0.56044996, 0.48936403],
        [0.5910041 , 0.46954608],
        [0.6231319 , 0.47053385],
        [0.6443183 , 0.48294783],
        [0.62151766, 0.49532306],
        [0.58998615, 0.49713737],
        [0.43043962, 0.7109368 ],
        [0.46534333, 0.6881037 ],
        [0.48942912, 0.6805593 ],
        [0.51225406, 0.6830868 ],
        [0.5341855 , 0.6793185 ],
        [0.55653214, 0.6858657 ],
        [0.5862155 , 0.7075139 ],
        [0.5538744 , 0.7318629 ],
        [0.5344313 , 0.73881537],
        [0.5121652 , 0.74083835],
        [0.48921126, 0.7400408 ],
        [0.46820667, 0.734128  ],
        [0.44096255, 0.7079656 ],
        [0.4936156 , 0.7052665 ],
        [0.5114745 , 0.70541674],
        [0.52808297, 0.70446116],
        [0.5756917 , 0.7052983 ],
        [0.52984923, 0.708476  ],
        [0.51168627, 0.7092654 ],
        [0.49262676, 0.7093172 ]], dtype=np.float32)
    landmarks = landmarks + tf.random.uniform(landmarks.shape, -0.1, 0.1)
    return np.moveaxis(landmarks, 0, 1)

class LandmarkDetector():

    def __init__(self, batch_size, num_landmarks=68):
        self.holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        self.batch_size = batch_size
        self.num_landmarks = num_landmarks
    
    def preprocess_and_detect_landmarks_numpy(self, images):
        raise NotImplementedError

    def preprocess_and_detect_landmarks(self, images):
        results = tf.numpy_function(self.preprocess_and_detect_landmarks_numpy, [images], Tout=[tf.float32])
        results = tf.convert_to_tensor(results, dtype=tf.float32)
        results.set_shape((self.batch_size, self.num_landmarks, 2)) # TODO: fix to consider batch size
        results = tf.expand_dims(results, axis=0)
        return results 


class MediapipeLandmarkDetector(LandmarkDetector):

    def preprocess_and_detect_landmarks_numpy(self, images):
        images =  normalized_sequence_to_images(images)
        images = np.array(images, dtype=np.uint8)
        all_landmarks = []
        for image in images:
            landmarks = []
            results = self.holistic_model.process(image)
            if results.face_landmarks is None:
                landmarks = get_default_landmarks()
            else:
                for landmark in results.face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks, dtype=np.float32)
                landmarks = landmarks[self.landmark_points_68]
            all_landmarks.append(landmarks)
        return np.array(all_landmarks, dtype=np.float32)


def landmarks_to_array_dlib(landmarks):
    x = []
    y = []
    for current_landmarks in landmarks:
        x.append(current_landmarks.x)
        y.append(current_landmarks.y)
    return np.array([x, y])

class DlibLandmarksDetector(LandmarkDetector):

    def preprocess_and_detect_landmarks_numpy(self, images):
        clip_landmarks = []
        for frame in images:
            frame =  normalized_sequence_to_images(frame)
            frame = np.array(frame, dtype=np.uint8)
            # Only considering the first face
            faces = face_recognition.api._raw_face_landmarks(face_image=frame)
            if len(faces) > 0:
                landmarks_array = landmarks_to_array_dlib(faces[0].parts())
                clip_landmarks.append(landmarks_array) # TODO: fix to work with batch size > 1
        if len(clip_landmarks) > 0:
            clip_landmarks = np.array(clip_landmarks)
            clip_landmarks = np.moveaxis(clip_landmarks, 1, 2)
        else:
            clip_landmarks = get_default_landmarks()
            clip_landmarks = np.expand_dims(clip_landmarks, axis=0)
        return np.ndarray.astype(clip_landmarks, dtype=np.float32)


def compute_displacements_interpolation(points_a, points_b, image_width, image_height, batch_size, fill_value=0):
    points_a = np.asarray(points_a)
    points_b = np.asarray(points_b)
    points_a = np.reshape(points_a, (batch_size, -1, 2))
    points_b = np.reshape(points_b, (batch_size, -1, 2))

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

def get_default_triangulation():
    triangles = [
        [78, 14, 77],
        [2, 90, 89],
        [85, 79, 80],
        [79, 12, 78],
        [12, 79, 85],
        [13, 93, 14],
        [78, 13, 14],
        [12, 13, 78],
        [12, 11, 104],
        [84, 11, 85],
        [11, 12, 85],
        [91, 81, 86],
        [4, 91, 90],
        [91, 4, 81],
        [82, 7, 83],
        [6, 7, 82],
        [107, 106, 71],
        [106, 70, 71],
        [72, 107, 71],
        [1, 2, 89],
        [13, 92, 93],
        [92, 13, 12],
        [75, 73, 74],
        [16, 76, 77],
        [25, 44, 24],
        [45, 44, 25],
        [3, 90, 2],
        [3, 4, 90],
        [9, 84, 83],
        [22, 21, 107],
        [21, 106, 107],
        [108, 72, 73],
        [72, 108, 107],
        [108, 25, 24],
        [107, 108, 24],
        [69, 87, 68],
        [105, 69, 70],
        [105, 106, 19],
        [106, 105, 70],
        [88, 0, 89],
        [0, 1, 89],
        [87, 0, 88],
        [5, 6, 82],
        [81, 5, 82],
        [4, 5, 81],
        [92, 54, 93],
        [54, 35, 93],
        [35, 54, 53],
        [54, 12, 104],
        [54, 92, 12],
        [93, 15, 14],
        [14, 15, 77],
        [15, 16, 77],
        [95, 45, 25],
        [95, 15, 45],
        [15, 95, 16],
        [44, 23, 24],
        [23, 107, 24],
        [23, 22, 107],
        [3, 96, 4],
        [96, 5, 4],
        [7, 8, 83],
        [8, 9, 83],
        [10, 11, 84],
        [9, 10, 84],
        [11, 10, 104],
        [18, 105, 19],
        [17, 87, 69],
        [17, 0, 87],
        [105, 17, 69],
        [18, 17, 105],
        [0, 99, 1],
        [17, 99, 0],
        [99, 17, 18],
        [35, 34, 30],
        [32, 31, 30],
        [31, 49, 48],
        [33, 32, 30],
        [34, 33, 30],
        [5, 100, 6],
        [100, 96, 48],
        [96, 100, 5],
        [64, 54, 104],
        [26, 95, 25],
        [95, 26, 16],
        [75, 26, 73],
        [26, 75, 76],
        [16, 26, 76],
        [26, 108, 73],
        [108, 26, 25],
        [23, 43, 22],
        [43, 23, 44],
        [43, 42, 22],
        [94, 15, 93],
        [15, 94, 45],
        [35, 94, 93],
        [94, 35, 30],
        [38, 20, 21],
        [106, 20, 19],
        [21, 20, 106],
        [41, 40, 98],
        [99, 36, 1],
        [36, 41, 98],
        [1, 36, 98],
        [36, 99, 18],
        [96, 97, 48],
        [97, 31, 48],
        [31, 97, 98],
        [1, 97, 2],
        [97, 1, 98],
        [97, 3, 2],
        [97, 96, 3],
        [60, 100, 48],
        [46, 44, 45],
        [94, 46, 45],
        [102, 8, 7],
        [8, 102, 9],
        [60, 59, 100],
        [37, 18, 19],
        [37, 36, 18],
        [20, 37, 19],
        [37, 20, 38],
        [36, 37, 41],
        [37, 40, 41],
        [40, 37, 38],
        [43, 47, 42],
        [47, 94, 42],
        [47, 46, 94],
        [103, 10, 9],
        [102, 103, 9],
        [56, 103, 102],
        [10, 103, 104],
        [102, 101, 58],
        [101, 59, 58],
        [101, 102, 7],
        [101, 7, 6],
        [100, 101, 6],
        [59, 101, 100],
        [29, 28, 42],
        [29, 94, 30],
        [94, 29, 42],
        [39, 40, 38],
        [39, 38, 21],
        [29, 39, 28],
        [40, 39, 98],
        [39, 29, 98],
        [55, 64, 104],
        [103, 55, 104],
        [55, 103, 56],
        [57, 102, 58],
        [57, 56, 102],
        [52, 35, 53],
        [52, 34, 35],
        [52, 33, 34],
        [52, 51, 33],
        [50, 31, 32],
        [31, 50, 49],
        [33, 50, 32],
        [51, 50, 33],
        [27, 39, 21],
        [39, 27, 28],
        [27, 21, 22],
        [42, 27, 22],
        [28, 27, 42],
        [46, 47, 44],
        [47, 43, 44],
        [48, 60, 59],  # lower_lip
        [60, 59, 67],
        [59, 67, 58],
        [67, 58, 66],
        [58, 66, 57],
        [66, 57, 65],
        [57, 65, 56],
        [56, 65, 55],
        [65, 55, 64],
        [55, 64, 54],
        [48, 60, 61],  # mouth_open
        [48, 60, 67],
        [60, 61, 67],
        [61, 62, 67],
        [62, 66, 67],
        [62, 66, 63],
        [63, 66, 65],
        [63, 64, 65],
        [63, 64, 54],
        [64, 54, 65],
        [48, 60, 49],  # upper_lip
        [60, 49, 61],
        [49, 50, 61],
        [50, 61, 51],
        [51, 61, 62],
        [52, 62, 63],
        [52, 51, 62],
        [52, 63, 53],
        [64, 53, 63],
        [53, 54, 64],
        [30, 98, 29],  # nose
        [30, 31, 98]
    ]
    return [line for line in triangles if all(num <= 67 for num in line)]

@tf.function
def euclidean_distance(a, b):
  return tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(a-b, 2), axis=-1))

@tf.function
def all_pairs_distances(x):
  x = tf.reshape(x, (-1, x.shape[-1]))
  indices = tf.stack(list(combinations(range(len(x)), 2))) # TODO: fix to consider batch size
  a_index = indices[:,0]
  b_index = indices[:,1]
  a = tf.gather(x, a_index)
  b = tf.gather(x, b_index)
  return euclidean_distance(a,b)

@tf.function
def l1_loss(previous, current):
  return  tf.math.abs(current-previous)

@tf.function
def pairwise_loss(y_true, y_pred):
  y_true = all_pairs_distances(y_true)
  y_pred = all_pairs_distances(y_pred)
  return tf.math.reduce_mean(euclidean_distance(y_true, y_pred))

@tf.function
def get_luminance(image):
  output = normalized_sequence_to_images(image)
  output = tfio.experimental.color.rgb_to_ycbcr(output)
  output = output[...,0] # get y from ycbcr
  return tf.cast(output, tf.float32)

@tf.function
def apply_luminance_loss(previous, current, loss_function):
  previous_luminance = get_luminance(previous)
  current_luminance = get_luminance(current)
  return loss_function(previous_luminance, current_luminance)

@tf.function
def mean_loss(y_true, y_pred, loss_function):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.math.reduce_mean(loss_function(y_true, y_pred))

@tf.function
def coherence_mean_loss(previous_y_true, y_true, previous_y_pred, y_pred, loss_function):
  loss_true = apply_luminance_loss(previous_y_true, y_true, loss_function)
  loss_pred = apply_luminance_loss(previous_y_pred, y_pred, loss_function)
  return tf.math.reduce_mean(tf.abs(loss_true - loss_pred))

@tf.function
def coherence_mean_landmarks_loss(previous_y_true_landmarks, y_true_landmarks, previous_y_pred_landmarks, y_pred_landmarks, loss_function):
  previous_y_true_landmarks = tf.cast(previous_y_true_landmarks, tf.float32)
  y_true_landmarks = tf.cast(y_true_landmarks, tf.float32)
  previous_y_pred_landmarks = tf.cast(previous_y_pred_landmarks, tf.float32)
  y_pred_landmarks = tf.cast(y_pred_landmarks, tf.float32)
  loss_true = loss_function(previous_y_true_landmarks, y_true_landmarks)
  loss_pred = loss_function(previous_y_pred_landmarks, y_pred_landmarks)
  return tf.math.reduce_mean(tf.abs(loss_true - loss_pred))

class GeneratorLoss(object):

    def __init__(self,
                landmarks_detector,
                main_loss_function,
                lambda_main_loss=1,
                coherence_loss_function=None,
                lambda_coherence_loss=1,
                landmarks_loss_function=None,
                lambda_landmarks_loss=1,
                landmarks_coherence_loss_function=None,
                lambda_landmarks_coherence_loss=1):
        self.main_loss_function = main_loss_function
        self.lambda_main_loss = lambda_main_loss
        self.coherence_loss_function = coherence_loss_function
        self.lambda_coherence_loss = lambda_coherence_loss
        self.landmarks_loss_function = landmarks_loss_function
        self.lambda_landmarks_loss = lambda_landmarks_loss
        self.landmarks_coherence_loss_function = landmarks_coherence_loss_function
        self.lambda_landmarks_coherence_loss = lambda_landmarks_coherence_loss
        self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.landmarks_detector = landmarks_detector

    def __call__(self, disc_generated_output, previous_gen, current_gen, previous_target, current_target):
        gan_loss = self.cross_entropy_loss(tf.ones_like(disc_generated_output), disc_generated_output)

        if self.main_loss_function:
            main_loss = mean_loss(current_target, current_gen, self.main_loss_function) * self.lambda_main_loss
        else:
            main_loss = 0

        if self.coherence_loss_function:
            coherence_loss = coherence_mean_loss(previous_target, current_target, previous_gen, current_gen, self.coherence_loss_function) * self.lambda_coherence_loss
        else:
            coherence_loss = 0

        landmarks_loss = 0
        landmarks_coherence_loss = 0

        if self.landmarks_loss_function or self.landmarks_coherence_loss_function:
            # TODO: compute landmarks only once per image
            current_target_landmarks = self.landmarks_detector.preprocess_and_detect_landmarks(current_target)[0] # TODO: fix to work with batch > 1
            current_gen_landmarks = self.landmarks_detector.preprocess_and_detect_landmarks(current_gen)[0] # TODO: fix to work with batch > 1
            if self.lambda_landmarks_loss:
                landmarks_loss = mean_loss(current_target_landmarks, current_gen_landmarks, self.landmarks_loss_function) * self.lambda_landmarks_loss
            if self.landmarks_coherence_loss_function:
                previous_target_landmarks = self.landmarks_detector.preprocess_and_detect_landmarks(previous_target)[0] # TODO: fix to work with batch > 1
                previous_gen_landmarks = self.landmarks_detector.preprocess_and_detect_landmarks(previous_gen)[0] # TODO: fix to work with batch > 1
                landmarks_coherence_loss = coherence_mean_landmarks_loss(previous_target_landmarks, current_target_landmarks, previous_gen_landmarks, current_gen_landmarks, self.landmarks_coherence_loss_function) * self.lambda_landmarks_coherence_loss

        total_gen_loss = gan_loss + main_loss + coherence_loss + landmarks_loss

        return total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src=src,
                          M=warp_mat,
                          dsize=(size[0], size[1]),
                          dst=None,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(img=mask,
                       points=np.int32(t2_rect_int),
                       color=(1.0, 1.0, 1.0),
                       lineType=cv2.LINE_AA,
                       shift=0)

    # Apply warpImage to small rectangular patches
    img1_rect = np.array(img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]])
    if len(img1_rect) > 0:
        rect_shape = (r2[2], r2[3])
        img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, rect_shape)
        img2_rect = img2_rect * mask
        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                    (1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def warp_all(source_image, source_triangles, target_triangles):
    source_image_warped = source_image.numpy().copy()

    for tri_source, tri_target in zip(source_triangles,
                                      target_triangles):
        warp_triangle(source_image, source_image_warped, tri_source, tri_target)

    return source_image_warped

def get_new_landmarks(source_landmarks_previous, source_landmarks_current, target_landmarks_current):
    return tf.add(tf.subtract(source_landmarks_current, source_landmarks_previous), target_landmarks_current)


def deform(target_sequence, source_sequence, landmarks_detector):
    default_triangulation = get_default_triangulation()
    target_landmarks = landmarks_detector.preprocess_and_detect_landmarks(target_sequence)
    source_landmarks = landmarks_detector.preprocess_and_detect_landmarks(source_sequence)
    warpped_sequence = [target_sequence[0]]
    for i in range(1, len(target_sequence)):
        target_image = normalized_sequence_to_images(target_sequence[i])
        target_landmarks = target_sequence[i]
        source_landmarks = source_sequence[i]
        previous_source_landmarks = source_sequence[i-1]

        target_triangles = target_landmarks[default_triangulation]
        new_landmarks = get_new_landmarks(previous_source_landmarks, source_landmarks, target_landmarks)
        new_triangles = new_landmarks[default_triangulation]

        warped_image = warp_all(target_image, new_triangles, target_triangles)
        warpped_sequence.append(warped_image)
    return tf.stack(warpped_sequence)

class DiscriminatorLoss():

    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, disc_real_output, disc_generated_output):
        real_loss = self.cross_entropy_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.cross_entropy_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss

def plot_results(sample_target, sample_input, gen_output):

  display_list = [sample_input, sample_target, gen_output]
  title = ['Input Sequence', 'Ground Truth', 'Predicted Sequence']

  for i in range(3):
    tf.print(title[i])
    display_list[i] = display_list[i]
    plot_normalized_sequence(display_list[i])

def generate_sequences(model, sample_target, sample_input):
  prediction = model(sample_input, training=True)
  plot_results(sample_input, sample_target, prediction)
