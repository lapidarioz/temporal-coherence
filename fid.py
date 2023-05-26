# Adapted from https://github.com/tensorflow/gan/blob/3a80f96fa1c9a424d13db9b139af9677e4a8982c/tensorflow_gan/examples/cifar/util.py
# coding=utf-8
# Copyright 2022 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convenience functions for training and evaluating a TF-GAN CIFAR example."""
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan  # tf


def get_inception_scores(images, batch_size, num_inception_images):
  """Get Inception score for some images.

  Args:
    images: Image minibatch. Shape [batch size, width, height, channels]. Values
      are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Inception scores. Tensor shape is [batch size].

  Raises:
    ValueError: If `batch_size` is incompatible with the first dimension of
      `images`.
    ValueError: If `batch_size` isn't divisible by `num_inception_images`.
  """
  # Validate inputs.
  images.shape[0:1].assert_is_compatible_with([batch_size])
  if batch_size % num_inception_images != 0:
    raise ValueError(
        '`batch_size` must be divisible by `num_inception_images`.')

  # Resize images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_images = tf.image.resize(
      images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Run images through Inception.
  num_batches = batch_size // num_inception_images
  inc_score = tfgan.eval.inception_score(
      resized_images, num_batches=num_batches)

  return inc_score


def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
  """Get Frechet Inception Distance between real and generated images.

  Args:
    real_images: Real images minibatch. Shape [batch size, width, height,
      channels. Values are in [-1, 1].
    generated_images: Generated images minibatch. Shape [batch size, width,
      height, channels]. Values are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Frechet Inception distance. A floating-point scalar.

  Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
  """
  # Validate input dimensions.
  real_images.shape[0:1].assert_is_compatible_with([batch_size])
  generated_images.shape[0:1].assert_is_compatible_with([batch_size])

  # Resize input images.
  size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
  resized_real_images = tf.image.resize(
      real_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  resized_generated_images = tf.image.resize(
      generated_images, [size, size], method=tf.image.ResizeMethod.BILINEAR)

  # Compute Frechet Inception Distance.
  num_batches = batch_size // num_inception_images
  fid = tfgan.eval.frechet_inception_distance(
      resized_real_images, resized_generated_images, num_batches=num_batches)

  return fid