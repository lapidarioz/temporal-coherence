import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from itertools import combinations
import tensorflow_io as tfio
from normalize import images_from_normalized

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
  output = images_from_normalized(image)
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

def perceptual_loss(y_true, y_pred):
    layer_name = 'block5_conv4'
    intermediate_layer_model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)
    return keras.losses.MSE(intermediate_layer_model(y_true), intermediate_layer_model(y_pred))

class GeneratorLoss(object):

    def __init__(self,
                main_loss_function,
                lambda_main_loss=1,
                coherence_loss_function=None,
                lambda_coherence_loss=1,
                landmarks_loss_function=None,
                lambda_landmarks_loss=1,
                landmarks_coherence_loss_function=None,
                lambda_landmarks_coherence_loss=1,
                lambda_landmarks_perceptual_loss=None):
        self.main_loss_function = main_loss_function
        self.lambda_main_loss = lambda_main_loss
        self.coherence_loss_function = coherence_loss_function
        self.lambda_coherence_loss = lambda_coherence_loss
        self.landmarks_loss_function = landmarks_loss_function
        self.lambda_landmarks_loss = lambda_landmarks_loss
        self.landmarks_coherence_loss_function = landmarks_coherence_loss_function
        self.lambda_landmarks_coherence_loss = lambda_landmarks_coherence_loss
        self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lambda_landmarks_perceptual_loss = lambda_landmarks_perceptual_loss
        if self.lambda_landmarks_perceptual_loss:
            self.vgg = VGG19(weights='imagenet', include_top=False)


    def __call__(self,
      disc_generated_output,
      previous_gen,
      current_gen,
      previous_target,
      current_target,
      previous_target_landmarks,
      current_target_landmarks,
      previous_gen_landmarks,
      current_gen_landmarks,
      ):
        gan_loss = self.cross_entropy_loss(tf.ones_like(disc_generated_output), disc_generated_output)

        if self.main_loss_function:
            main_loss = mean_loss(current_target, current_gen, self.main_loss_function) * self.lambda_main_loss
        else:
            main_loss = 0

        if self.coherence_loss_function:
            coherence_loss = coherence_mean_loss(previous_target, current_target, previous_gen, current_gen, self.coherence_loss_function) * self.lambda_coherence_loss
        else:
            coherence_loss = 0
          
        if self.vgg:
            landmarks_perceptual_loss = perceptual_loss(current_target_landmarks, current_gen_landmarks) * self.lambda_landmarks_perceptual_loss
        else:
            landmarks_perceptual_loss = 0

        landmarks_loss = 0
        landmarks_coherence_loss = 0

        if self.lambda_landmarks_loss:
            landmarks_loss = mean_loss(
                current_target_landmarks,
                current_gen_landmarks,
                self.landmarks_loss_function
              ) * self.lambda_landmarks_loss
            
        if self.landmarks_coherence_loss_function:
            landmarks_coherence_loss = coherence_mean_landmarks_loss(
              previous_target_landmarks,
              current_target_landmarks,
              previous_gen_landmarks,
              current_gen_landmarks,
              self.landmarks_coherence_loss_function
            ) * self.lambda_landmarks_coherence_loss

        total_gen_loss = gan_loss + main_loss + coherence_loss + landmarks_loss

        return total_gen_loss, gan_loss, main_loss, coherence_loss, landmarks_loss, landmarks_coherence_loss



class DiscriminatorLoss():

    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, disc_real_output, disc_generated_output):
        real_loss = self.cross_entropy_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.cross_entropy_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss
