# Retrived from https://github.com/google-research/google-research/tree/master/frechet_video_distance
# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import six
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub


def preprocess(videos, target_resolution):
  """Runs some preprocessing on the videos for I3D model.

  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    target_resolution: (width, height): target video resolution

  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def _is_in_graph(tensor_name):
  """Checks whether a given tensor does exists in the graph."""
  try:
    tf.get_default_graph().get_tensor_by_name(tensor_name)
  except KeyError:
    return False
  return True


def create_id3_embedding(videos):
  """Embeds the given videos using the Inflated 3D Convolution network.

  Downloads the graph of the I3D from tf.hub and adds it to the graph on the
  first call.

  Args:
    videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
      Expected range is [-1, 1].

  Returns:
    embedding: <float32>[batch_size, embedding_size]. embedding_size depends
               on the model used.

  Raises:
    ValueError: when a provided embedding_layer is not supported.
  """

  batch_size = 16
  module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


  # Making sure that we import the graph separately for
  # each different input video tensor.
  module_name = "fvd_kinetics-400_id3_module_" + six.ensure_str(
      videos.name).replace(":", "_")

  assert_ops = [
      tf.Assert(
          tf.reduce_max(videos) <= 1.001,
          ["max value in frame is > 1", videos]),
      tf.Assert(
          tf.reduce_min(videos) >= -1.001,
          ["min value in frame is < -1", videos]),
      tf.assert_equal(
          tf.shape(videos)[0],
          batch_size, ["invalid frame batch size: ",
                       tf.shape(videos)],
          summarize=6),
  ]
  with tf.control_dependencies(assert_ops):
    videos = tf.identity(videos)

  module_scope = "%s_apply_default/" % module_name

  tf.print("videos shape: ", videos.shape)
  tf.print("videos name: ", videos.name)
  tf.print("module name: ", module_name)
  tf.print("module scope: ", module_scope)

  # To check whether the module has already been loaded into the graph, we look
  # for a given tensor name. If this tensor name exists, we assume the function
  # has been called before and the graph was imported. Otherwise we import it.
  # Note: in theory, the tensor could exist, but have wrong shapes.
  # This will happen if create_id3_embedding is called with a frames_placehoder
  # of wrong size/batch size, because even though that will throw a tf.Assert
  # on graph-execution time, it will insert the tensor (with wrong shape) into
  # the graph. This is why we need the following assert.
  video_batch_size = int(videos.shape[0])
  assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
  tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
  if not _is_in_graph(tensor_name):
    i3d_model = hub.Module(module_spec, name=module_name)
    i3d_model(videos)

  # gets the kinetics-i3d-400-logits layer
  tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
  tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
  return tensor


def calculate_fvd(real_activations,
                  generated_activations):
  """Returns a list of ops that compute metrics as funcs of activations.

  Args:
    real_activations: <float32>[num_samples, embedding_size]
    generated_activations: <float32>[num_samples, embedding_size]

  Returns:
    A scalar that contains the requested FVD.
  """
  return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)


# Overall Purpose

# The core function of this code is to calculate the FVD metric between real videos and videos generated by your facial expression synthesis model. Here's the gist of FVD:

#     Video-Specific Evaluation:  FVD adapts the principles of the Fréchet Inception Distance (FID), commonly used for image evaluation,  to the domain of videos.  This means it aims to assess both the quality of individual frames and the temporal coherence between them.

#     I3D Embedding: The code relies on a pre-trained Inflated 3D Convolutional Network (I3D) to extract meaningful features (embeddings)  from video frames. I3D is known to capture both appearance and motion cues.

#     Distribution Comparison:  Similar to FID, FVD ultimately compares the distributions of these feature embeddings for real and generated videos. A lower FVD indicates greater similarity between the distributions, suggesting better video generation quality.

# Functionality Breakdown

#     Preprocessing (preprocess):
#         Resizes video frames to match the expected input of the I3D model.
#         Scales the pixel values to the range of [-1, 1], likely a requirement of I3D.

#     Feature Extraction (create_id3_embedding):
#         Handles the dynamic loading of the I3D model from TensorFlow Hub.
#         Batches the video frames for efficient processing.
#         Extracts the relevant layer from the I3D network, which provides the video embeddings.

#     FVD Calculation (calculate_fvd):
#         Leverages functionality from the tensorflow_gan (TF-GAN) library to compute the FVD based on the extracted feature embeddings of real and generated videos.

# Importance for Facial Expression Synthesis

#     Beyond Single Images:  FVD, unlike image-based metrics, is sensitive to temporal smoothness and coherence, making it well-suited for evaluating your facial expression synthesis model's ability to generate realistic transitions.

#     I3D Features: The I3D model is designed to capture motion-related information, which is crucial for assessing if your generated expressions look natural and dynamic.

# Design Choices

#     External Model: Relying on a pre-trained I3D model streamlines the process and leverages a network specifically trained for video understanding.

#     TF-GAN Integration: Utilizes TF-GAN's optimized implementations for the final FVD calculation.

# Potential Limitations

#     Computational Cost: Processing videos through the I3D network and calculating FVD can be computationally demanding.

#     General Video Focus:  While I3D is powerful, it might not be perfectly  attuned to the specific nuances of facial expressions.
