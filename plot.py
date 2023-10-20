from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from normalize import images_from_normalized
import tensorflow as tf
import imageio
import numpy as np

def plot_sample_sequence(sample_sequence, save_path=None):
    sequence = list(sample_sequence)
    n_frames = len(sequence)
    if n_frames > 20:
        sequence = sequence[:20]
        n_frames = 20
    rows = ((n_frames-1) // 10)+1
    cols = max(n_frames % 10, 10)
    fig = plt.figure(figsize=(20, 4))
    for i, im in enumerate(sequence):
        ax = fig.add_subplot(rows,cols,i+1)
        ax.imshow(im) 
        ax.axis('off')
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_frame(frame, title=None, save_path=None):
    plt.imshow(frame)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_normalized_frame(frame, title=None):
    plot_frame(images_from_normalized(frame), title)

def sequence_generator(sample_sequence):
    for i in range(sample_sequence.shape[0]):
        yield sample_sequence[i, ...]

def plot_sequence(sample_sequence, save_path=None):
    plot_sample_sequence(sequence_generator(sample_sequence), save_path=save_path)

def inverse_sequence_generator(sample_sequence):
    for i in range(sample_sequence.shape[-1]):
        yield sample_sequence[..., i]

def plot_inverse_sequence(sample_sequence):
    plot_sample_sequence(inverse_sequence_generator(sample_sequence))

def plot_normalized_inverse_sequence(sample_sequence):
    plot_inverse_sequence(images_from_normalized(sample_sequence))

def plot_normalized_sequence(sample_sequence, save_path=None):
    plot_sequence(images_from_normalized(sample_sequence), save_path=save_path)

def plot_sequence_from_tensor(sample_sequence):
    if len(sample_sequence.shape) == 5 and sample_sequence.shape[0] == 1:
        plot_sequence(sample_sequence[0])
    else:
        raise ValueError("The tensor must be of shape (1, frames, height, width, channels)")

def plot_sequence_from_normalized_tensor(sample_sequence):
    plot_sequence_from_tensor(images_from_normalized(sample_sequence))

def plot_results_rgb(sample_target, gen_output, save_path=None):
  display_list = [sample_target, gen_output]
  title = ['Ground Truth', 'Predicted Sequence']

  plot_path = None
  for i in range(len(display_list)):
    tf.print(title[i])
    display_list[i] = display_list[i]
    if save_path is not None:
      plot_path = save_path / f"{title[i]}.pdf"
      output_video_path = save_path / f"{title[i]}.npy"
      np.save(output_video_path, display_list[i])
      save_gif_rgb(display_list[i], save_path / f"sample_{title[i]}.gif")
    plot_sequence(display_list[i], save_path=plot_path)

def plot_results(sample_target, gen_output, save_path=None):
  display_list = [sample_target, gen_output]
  title = ['Ground Truth', 'Predicted Sequence']

  for i in range(len(display_list)):
    tf.print(title[i])
    display_list[i] = display_list[i]
    plot_path = None
    if save_path is not None:
      plot_path = save_path / f"{title[i]}.pdf"
      output_video_path = save_path / f"{title[i]}.npy"
      np.save(output_video_path, display_list[i])
      save_gif(display_list[i], save_path / f"sample_{title[i]}.gif")
    plot_normalized_sequence(display_list[i], save_path=plot_path)

def generate_sequences(model, sample_target, sample_input):
  prediction = model(sample_input, training=True)
  plot_results(sample_input, sample_target, prediction)

def plot_landmarks(sample_sequence, landmarks, save_path=None):
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
        im = images_from_normalized(im)
        ax.imshow(im)
        ax.scatter(landmarks[i, :, 0], landmarks[i, :, 1], s=10, c='r') 
        ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_triangles(sample_sequence, triangles, save_path=None):
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
        im = images_from_normalized(im)
        ax.imshow(im)
        ax.axis('off')
        for triangle in triangles[i]:
            polygon = Polygon(triangle, facecolor='r', fill=False)
            ax.add_patch(polygon)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()
    
def save_gif(sample_sequence, path):
    images = []
    for i in range(sample_sequence.shape[0]):
        images.append(images_from_normalized(sample_sequence[i]))
    imageio.mimwrite(str(path), images)

def save_gif_rgb(sample_sequence, path):
    imageio.mimwrite(str(path), sample_sequence, loop=0)