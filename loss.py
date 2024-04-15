import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow_hub as hub
from itertools import combinations
import tensorflow_io as tfio
from normalize import images_from_normalized
from tensorflow_graphics.math.interpolation.slerp import interpolate
import numpy as np


@tf.function
def euclidean_distance(a, b):
    """
    Calculates the Euclidean distance between two tensors.

    Args:
        a (tf.Tensor): The first tensor.
        b (tf.Tensor): The second tensor.

    Returns:
        tf.Tensor: The Euclidean distance between the two tensors.
    """
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(a-b, 2), axis=-1))

# This function calculates the Euclidean distance between two tensors.
# It uses the formula sqrt(sum((a - b)^2)) to compute the distance.
# The function is decorated with @tf.function to optimize its execution using TensorFlow's autograph feature.


@tf.function
def all_pairs_distances(x):
    """
    Calculates the Euclidean distance between all pairs of vectors in x.

    Args:
        x: A tensor of shape (batch_size, num_vectors, vector_dim) representing the input vectors.

    Returns:
        A tensor of shape (num_pairs,) representing the Euclidean distances between all pairs of vectors.
    """
    x = tf.reshape(x, (-1, x.shape[-1]))

    # Generate all possible pairs of indices
    indices = tf.stack(list(combinations(range(len(x)), 2)))  # TODO: fix to consider batch size

    # Split the indices into two separate tensors
    a_index = indices[:, 0]
    b_index = indices[:, 1]

    # Gather the vectors corresponding to the indices
    a = tf.gather(x, a_index)
    b = tf.gather(x, b_index)

    # Calculate the Euclidean distance between the pairs of vectors
    return euclidean_distance(a, b)


@tf.function
def l1_loss(previous, current):
    """
    Calculates the L1 loss between two tensors.

    Args:
        previous (tf.Tensor): The previous tensor.
        current (tf.Tensor): The current tensor.

    Returns:
        tf.Tensor: The L1 loss between the previous and current tensors.
    """
    return tf.math.abs(current - previous)


@tf.function
def pairwise_loss(y_true, y_pred):
    """
    Calculates the pairwise loss between the true and predicted values.

    Args:
        y_true (tf.Tensor): The true values.
        y_pred (tf.Tensor): The predicted values.

    Returns:
        tf.Tensor: The pairwise loss.

    """
    # Calculate all pairs distances for true and predicted values
    y_true = all_pairs_distances(y_true)
    y_pred = all_pairs_distances(y_pred)

    # Calculate the Euclidean distance between true and predicted values
    return tf.math.reduce_mean(euclidean_distance(y_true, y_pred))


@tf.function
def get_luminance(image):
    """
    Converts an RGB image to luminance (Y) channel using YCbCr color space.

    Args:
        image (tf.Tensor): Input RGB image tensor.

    Returns:
        tf.Tensor: Luminance (Y) channel of the input image.

    """
    # Convert image from normalized RGB to YCbCr color space
    output = images_from_normalized(image)
    output = tfio.experimental.color.rgb_to_ycbcr(output)

    # Extract the luminance (Y) channel from YCbCr image
    output = output[..., 0]

    # Cast the output to float32
    return tf.cast(output, tf.float32)


@tf.function
def apply_luminance_loss(previous, current, loss_function):
    """
    Applies the luminance loss between previous and current tensors using the given loss function.

    Args:
        previous (tf.Tensor): The previous tensor.
        current (tf.Tensor): The current tensor.
        loss_function (function): The loss function to calculate the luminance loss.

    Returns:
        tf.Tensor: The luminance loss between the previous and current tensors.

    """
    previous_luminance = get_luminance(previous)
    current_luminance = get_luminance(current)
    return loss_function(previous_luminance, current_luminance)


@tf.function
def mean_loss(y_true, y_pred, loss_function):
    """
    Calculates the mean loss between the true labels and predicted labels.

    Args:
        y_true (tf.Tensor): The true labels.
        y_pred (tf.Tensor): The predicted labels.
        loss_function (callable): The loss function to calculate the individual losses.

    Returns:
        tf.Tensor: The mean loss.

    """
    # Cast the labels to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate the individual losses and take the mean
    return tf.math.reduce_mean(loss_function(y_true, y_pred))


@tf.function
def coherence_mean_loss(previous_y_true, y_true, previous_y_pred, y_pred, loss_function):
    """
    Calculates the coherence mean loss between the true and predicted values.

    Args:
        previous_y_true (tf.Tensor): The previous true values.
        y_true (tf.Tensor): The current true values.
        previous_y_pred (tf.Tensor): The previous predicted values.
        y_pred (tf.Tensor): The current predicted values.
        loss_function (callable): The loss function to be applied.

    Returns:
        tf.Tensor: The coherence mean loss.

    """
    # Calculate the luminance loss for the true and predicted values
    loss_true = apply_luminance_loss(previous_y_true, y_true, loss_function)
    loss_pred = apply_luminance_loss(previous_y_pred, y_pred, loss_function)

    # Calculate the coherence mean loss by taking the absolute difference between the two losses
    coherence_loss = tf.math.reduce_mean(tf.abs(loss_true - loss_pred))

    return coherence_loss


@tf.function
def coherence_mean_landmarks_loss(previous_y_true_landmarks, y_true_landmarks, previous_y_pred_landmarks, y_pred_landmarks, loss_function):
    """
    Calculates the coherence mean landmarks loss between the previous and current landmarks predictions.

    Args:
        previous_y_true_landmarks (tf.Tensor): The previous true landmarks.
        y_true_landmarks (tf.Tensor): The current true landmarks.
        previous_y_pred_landmarks (tf.Tensor): The previous predicted landmarks.
        y_pred_landmarks (tf.Tensor): The current predicted landmarks.
        loss_function (callable): The loss function to calculate the loss between landmarks.

    Returns:
        tf.Tensor: The coherence mean landmarks loss.

    """
    # Casting the input tensors to float32
    previous_y_true_landmarks = tf.cast(previous_y_true_landmarks, tf.float32)
    y_true_landmarks = tf.cast(y_true_landmarks, tf.float32)
    previous_y_pred_landmarks = tf.cast(previous_y_pred_landmarks, tf.float32)
    y_pred_landmarks = tf.cast(y_pred_landmarks, tf.float32)

    # Calculating the loss between previous and current true landmarks
    loss_true = loss_function(previous_y_true_landmarks, y_true_landmarks)

    # Calculating the loss between previous and current predicted landmarks
    loss_pred = loss_function(previous_y_pred_landmarks, y_pred_landmarks)

    # Calculating the coherence mean landmarks loss
    coherence_loss = tf.math.reduce_mean(tf.abs(loss_true - loss_pred))

    return coherence_loss


def perceptual_loss(y_true, y_pred, vgg):
    """
    Calculates the perceptual loss between the true and predicted images.

    Args:
        y_true (tf.Tensor): The true image tensor.
        y_pred (tf.Tensor): The predicted image tensor.
        vgg (tf.keras.Model): The pre-trained VGG model.

    Returns:
        tf.Tensor: The mean squared error (MSE) of the intermediate layer outputs.

    Raises:
        None

    Examples:
        # Create a VGG model
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

        # Calculate the perceptual loss
        loss = perceptual_loss(y_true, y_pred, vgg)
    """
    layer_name = 'block5_conv4'
    
    # Create a new model that outputs the intermediate layer activations
    intermediate_layer_model = Model(
        inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)
    
    # Calculate the mean squared error (MSE) between the intermediate layer outputs of the true and predicted images
    return tf.reduce_mean(keras.losses.MSE(intermediate_layer_model(y_true), intermediate_layer_model(y_pred)))


def interpolation_loss(previous_gen, current_gen, previous_target, current_target, loss_function):
    """
    Calculates the interpolation loss between two generations and their corresponding targets.

    Args:
        previous_gen (tf.Tensor): The previous generation.
        current_gen (tf.Tensor): The current generation.
        previous_target (tf.Tensor): The target corresponding to the previous generation.
        current_target (tf.Tensor): The target corresponding to the current generation.
        loss_function (callable): The loss function to be used for calculating the interpolation loss.

    Returns:
        tf.Tensor: The mean interpolation loss.

    """
    # Calculate the mid-generation and mid-target
    mid_gen = (previous_gen + current_gen) / 2
    mid_target = (previous_target + current_target) / 2

    # Cast mid-generation and mid-target to float32
    mid_gen = tf.cast(mid_gen, tf.float32)
    mid_target = tf.cast(mid_target, tf.float32)

    # Calculate the interpolation loss using the provided loss function
    interpolation_loss = tf.math.reduce_mean(loss_function(mid_target, mid_gen))

    return interpolation_loss

class GeneratorLoss(object):

    class Loss:
        def __init__(self,
                     target_predicted_function,
                     lambda_target_predicted=1,
                     coherence_loss_function=None,
                     lambda_coherence_loss=1,
                     landmarks_loss_function=None,
                     lambda_landmarks_loss=1,
                     landmarks_coherence_loss_function=None,
                     lambda_landmarks_coherence_loss=1,
                     lambda_perceptual_loss=None,
                     interpolation_frames_loss_function=None,
                     lambda_interpolation_frames_loss=1,
                     interpolation_landmarks_loss_function=None,
                     lambda_interpolation_landmarks_loss=1):
            """
            Initializes the Loss object with the given parameters.

            Args:
                target_predicted_function: The function that predicts the target.
                lambda_target_predicted: The weight for the target predicted loss.
                coherence_loss_function: The function to compute the coherence loss.
                lambda_coherence_loss: The weight for the coherence loss.
                landmarks_loss_function: The function to compute the landmarks loss.
                lambda_landmarks_loss: The weight for the landmarks loss.
                landmarks_coherence_loss_function: The function to compute the landmarks coherence loss.
                lambda_landmarks_coherence_loss: The weight for the landmarks coherence loss.
                lambda_perceptual_loss: The weight for the perceptual loss.
                interpolation_frames_loss_function: The function to compute the interpolation frames loss.
                lambda_interpolation_frames_loss: The weight for the interpolation frames loss.
                interpolation_landmarks_loss_function: The function to compute the interpolation landmarks loss.
                lambda_interpolation_landmarks_loss: The weight for the interpolation landmarks loss.
            """
            self.target_predicted_function = target_predicted_function
            self.lambda_target_predicted = lambda_target_predicted
            self.coherence_loss_function = coherence_loss_function
            self.lambda_coherence_loss = lambda_coherence_loss
            self.landmarks_loss_function = landmarks_loss_function
            self.lambda_landmarks_loss = lambda_landmarks_loss
            self.landmarks_coherence_loss_function = landmarks_coherence_loss_function
            self.lambda_landmarks_coherence_loss = lambda_landmarks_coherence_loss
            self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True)
            self.lambda_perceptual_loss = lambda_perceptual_loss
            if self.lambda_perceptual_loss is not None:
                self.vgg = VGG19(weights='imagenet', include_top=False)
            else:
                self.vgg = None
            self.interpolation_frames_loss_function = interpolation_frames_loss_function
            self.lambda_interpolation_frames_loss = lambda_interpolation_frames_loss
            # if self.interpolation_frames_loss_function is not None:
            #    self.interpolation_model = hub.load("https://tfhub.dev/google/film/1")
            self.interpolation_landmarks_loss_function = interpolation_landmarks_loss_function
            self.lambda_interpolation_landmarks_loss = lambda_interpolation_landmarks_loss

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
        """
        Calculates the loss for the generator model.

        Args:
            disc_generated_output (tf.Tensor): The output of the generator for the generated images.
            previous_gen (tf.Tensor): The previous generated image.
            current_gen (tf.Tensor): The current generated image.
            previous_target (tf.Tensor): The previous target image.
            current_target (tf.Tensor): The current target image.
            previous_target_landmarks (tf.Tensor): The landmarks of the previous target image.
            current_target_landmarks (tf.Tensor): The landmarks of the current target image.
            previous_gen_landmarks (tf.Tensor): The landmarks of the previous generated image.
            current_gen_landmarks (tf.Tensor): The landmarks of the current generated image.

        Returns:
            tf.Tensor: The total loss for the generator.
            dict: A dictionary containing individual loss values.

        """
        losses = dict()

        # Calculate the GAN loss using binary cross entropy
        gan_loss = self.cross_entropy_loss(tf.ones_like(disc_generated_output), disc_generated_output)
        losses["gan_loss"] = gan_loss

        if self.target_predicted_function:
            # Calculate the loss for target prediction
            target_predicted = mean_loss(current_target, current_gen, self.target_predicted_function) * self.lambda_target_predicted
            losses["target_predicted"] = target_predicted

        if self.coherence_loss_function:
            # Calculate the coherence loss
            coherence_loss = coherence_mean_loss(previous_target, current_target, previous_gen, current_gen, self.coherence_loss_function) * self.lambda_coherence_loss
            losses["coherence_loss"] = coherence_loss

        if self.vgg is not None and self.lambda_perceptual_loss is not None:
            # Calculate the perceptual loss using VGG19 model
            perceptual_loss_value = perceptual_loss(current_target, current_gen, self.vgg) * self.lambda_perceptual_loss
            losses["perceptual_loss"] = perceptual_loss_value

        if self.interpolation_frames_loss_function:
            # Calculate the interpolation frames loss
            interpolation_frame_loss_value = interpolation_loss(previous_gen, current_gen, previous_target, current_target, self.interpolation_frames_loss_function) * self.lambda_interpolation_frames_loss
            losses["interpolation_frame_loss"] = interpolation_frame_loss_value

        if self.landmarks_loss_function is not None and self.lambda_landmarks_loss is not None:
            # Calculate the landmarks loss
            landmarks_loss = mean_loss(current_target_landmarks, current_gen_landmarks, self.landmarks_loss_function) * self.lambda_landmarks_loss
            losses["landmarks_loss"] = landmarks_loss

        if self.landmarks_coherence_loss_function is not None and self.lambda_landmarks_coherence_loss is not None:
            # Calculate the landmarks coherence loss
            landmarks_coherence_loss = coherence_mean_landmarks_loss(previous_target_landmarks, current_target_landmarks, previous_gen_landmarks, current_gen_landmarks, self.landmarks_coherence_loss_function) * self.lambda_landmarks_coherence_loss
            losses["landmarks_coherence_loss"] = landmarks_coherence_loss

        if self.interpolation_landmarks_loss_function is not None and self.lambda_interpolation_landmarks_loss is not None:
            # Calculate the interpolation landmarks loss
            interpolation_landmarks_loss_value = interpolation_loss(previous_gen_landmarks, current_gen_landmarks, previous_target_landmarks, current_target_landmarks, self.interpolation_landmarks_loss_function) * self.lambda_interpolation_landmarks_loss
            losses["interpolation_landmarks_loss"] = interpolation_landmarks_loss_value

        total_gen_loss = 0
        for loss_value in losses.values():
            total_gen_loss += loss_value

        return total_gen_loss, losses


class DiscriminatorLoss():
    """
    Calculates the loss for the discriminator model.

    Args:
        None

    Returns:
        tf.Tensor: The total loss for the discriminator.

    """

    def __init__(self):
        """
        Initializes the DiscriminatorLoss class.

        Args:
            None

        Returns:
            None

        """
        self.cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)

    def __call__(self, disc_real_output, disc_generated_output):
        """
        Calculates the loss for the discriminator model.

        Args:
            disc_real_output (tf.Tensor): The output of the discriminator for real images.
            disc_generated_output (tf.Tensor): The output of the discriminator for generated images.

        Returns:
            tf.Tensor: The total loss for the discriminator.

        """
        real_loss = self.cross_entropy_loss(
            tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.cross_entropy_loss(
            tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss
