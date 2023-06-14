import tensorflow as tf

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


def get_discriminator_model(img_width, img_height, n_channels):
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


def get_generator_model(img_width, img_height, n_channels):
    previous = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='previous_frame')
    warped = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='warped_frame')
    neutral = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='neutral_frame')
    distances = tf.keras.layers.Input(shape=[img_width, img_height, 1], dtype=tf.float64, name='distances')


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

    x = tf.keras.layers.concatenate([neutral, previous, warped, distances])
    # x = tf.keras.layers.concatenate([previous])

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
