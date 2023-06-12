
import tensorflow as tf

# Normalizing the images to [-1, 1]
def normalize_images(sequence):
  return (sequence / 127.5) - 1

# Undo normalizing the images to [0, 255]
@tf.function
def images_from_normalized(sequence):
  return tf.cast((sequence + 1) * 127.5, tf.uint8)
