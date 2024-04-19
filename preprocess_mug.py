# %% [markdown]
# # Preprocess MUG

# %%
import tensorflow as tf
import tensorflow_io as tfio

import os
from pathlib import Path
import time
import datetime
import numpy as np
import cv2

from matplotlib import pyplot as plt
from IPython.display import display, clear_output, update_display
from tqdm.notebook import tqdm

from joblib import Memory
from joblib import Parallel, delayed

# %%
NEW_SIZE = 128

# %%
# Normalizing the images to [-1, 1]
def normalize(sequence):
  return (sequence / 127.5) - 1

# Undo normalizing the images to [0, 255]
@tf.function
def normalized_sequence_to_images(sequence):
  return tf.cast((sequence + 1) * 127.5, tf.uint8)

# %%
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

# %%
def get_save_path(video_path, save_folder, video_folder="mug"):
    save_path = str(video_path).replace(video_folder, save_folder)
    return Path(save_path).with_suffix(".npy")

def load_video(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video = []
    for i in range(n):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
    cap.release()
    return np.array(video)
    

# resize video lanczos4
def resize_video(video, size=NEW_SIZE):
    resized = []
    for frame in video:
        if frame is not None:
            resized.append(cv2.resize(frame, (size,size), interpolation=cv2.INTER_LANCZOS4))
    return resized

# BRG to RGB
def bgr_to_rgb(video):
    return np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video])

def save_video(video, video_path, save_folder):
    video_save_path = get_save_path(video_path, save_folder)
    video_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(video_save_path), video)
        
# load videos from npy
def load_videos(videos_paths, n=None):
    videos_paths = list(videos_paths)
    print(videos_paths)
    if n is None:
        n = len(videos_paths)
    else:
        videos_paths = videos_paths[:n]
    videos = []
    for path in tqdm(videos_paths, total=n):
        videos.append(np.load(path))
    return videos

def process_videos(videos_paths, save_folder):
    videos_paths = list(videos_paths)
    n = len(videos_paths)
    for video_path in tqdm(videos_paths, total=n):
        video = load_video(video_path)
        if NEW_SIZE != 448:
            video = resize_video(video)
        video = bgr_to_rgb(video)
        video = normalize(video)
        save_video(video, video_path, save_folder)


# %%
mug_path = Path('../data/mug/')
save_folder = f'mug{NEW_SIZE}'
process_videos(list(mug_path.glob('**/*.avi')), save_folder)

# %%
save_path = Path(f'../data/{save_folder}/')
videos = load_videos(save_path.glob('**/*.npy'))
len(videos)
print(videos[0].shape)

# %%
print(videos[0].shape)
plot_normalized_sequence(videos[0])


