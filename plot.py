from matplotlib import pyplot as plt
from normalize import images_from_normalized
import tensorflow as tf
import imageio

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
    plot_frame(images_from_normalized(frame), title)

def sequence_generator(sample_sequence):
    for i in range(sample_sequence.shape[0]):
        yield sample_sequence[i, ...]

def plot_sequence(sample_sequence):
    plot_sample_sequence(sequence_generator(sample_sequence))

def inverse_sequence_generator(sample_sequence):
    for i in range(sample_sequence.shape[-1]):
        yield sample_sequence[..., i]

def plot_inverse_sequence(sample_sequence):
    plot_sample_sequence(inverse_sequence_generator(sample_sequence))

def plot_normalized_inverse_sequence(sample_sequence):
    plot_inverse_sequence(images_from_normalized(sample_sequence))

def plot_normalized_sequence(sample_sequence):
    plot_sequence(images_from_normalized(sample_sequence))

def plot_sequence_from_tensor(sample_sequence):
    if len(sample_sequence.shape) == 5 and sample_sequence.shape[0] == 1:
        plot_sequence(sample_sequence[0])
    else:
        raise ValueError("The tensor must be of shape (1, frames, height, width, channels)")

def plot_sequence_from_normalized_tensor(sample_sequence):
    plot_sequence_from_tensor(images_from_normalized(sample_sequence))

def plot_results_rgb(sample_target, gen_output):
  display_list = [sample_target, gen_output]
  title = ['Ground Truth', 'Predicted Sequence']

  for i in range(len(display_list)):
    tf.print(title[i])
    display_list[i] = display_list[i]
    plot_sequence(display_list[i])

def plot_results(sample_target, sample_input, gen_output):
  display_list = [sample_input, sample_target, gen_output]
  title = ['Input Sequence', 'Ground Truth', 'Predicted Sequence']

  for i in range(len(display_list)):
    tf.print(title[i])
    display_list[i] = display_list[i]
    plot_normalized_sequence(display_list[i])

def generate_sequences(model, sample_target, sample_input):
  prediction = model(sample_input, training=True)
  plot_results(sample_input, sample_target, prediction)

def plot_landmarks(sample_sequence, landmarks):
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
    plt.show()
    plt.close()
    
def save_gif(sample_sequence, path, fps=10):
    images = []
    for i in range(sample_sequence.shape[0]):
        images.append(images_from_normalized(sample_sequence[i]))
    imageio.mimwrite(path, images, fps=fps)
