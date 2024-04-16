import tensorflow as tf

def downsample(filters, size, strides=2, apply_batchnorm=True):
    """
    Creates a downsampling block for convolutional neural networks.

    This block is commonly used to reduce the spatial dimensions of an image 
    while increasing the number of feature channels.

    Args:
        filters (int): The number of convolutional filters to use.
        size (int): The kernel size of the convolution.
        strides (int): The stride of the convolution (default is 2 for downsampling by a factor of 2).
        apply_batchnorm (bool): Whether to apply BatchNormalization (default is True).

    Returns:
        tf.keras.Sequential: A TensorFlow Sequential model representing the downsampling block.
    """

    initializer = tf.random_normal_initializer(0., 0.02)  # Initialize weights

    result = tf.keras.Sequential() 
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                               kernel_initializer=initializer, use_bias=False)) # Convolutional layer

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())  # Optional Batch Normalization

    result.add(tf.keras.layers.LeakyReLU())  # LeakyReLU activation 

    return result 

def upsample(filters, size, strides=2, apply_dropout=False):
    """
    Creates an upsampling block for convolutional neural networks.

    This block performs the opposite of a downsampling block,  increasing the spatial 
    dimensions of an image while often reducing the number of feature channels.

    Args:
        filters (int): The number of convolutional filters to use.
        size (int): The kernel size of the transposed convolution.
        strides (int): The stride of the transposed convolution (default is 2 for upsampling by a factor of 2).
        apply_dropout (bool): Whether to apply Dropout regularization (default is False).

    Returns:
        tf.keras.Sequential: A TensorFlow Sequential model representing the upsampling block.
    """

    initializer = tf.random_normal_initializer(0., 0.02)  # Initialize weights

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)) # Transposed convolution 

    result.add(tf.keras.layers.BatchNormalization())  # Batch normalization

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))  # Optional Dropout

    result.add(tf.keras.layers.ReLU())  # ReLU activation

    return result 


def get_discriminator_model(img_width, img_height, n_channels):
    """
    Creates a discriminator model for use in image-based machine learning tasks (GAN).

    The discriminator's role is to distinguish between 'real' and 'fake' image samples, typically
    by learning to identify subtle differences or inconsistencies.  

    Args:
        img_width (int): Width of the input images.
        img_height (int): Height of the input images.
        n_channels (int): Number of channels (e.g., 3 for RGB images).

    Returns:
        tf.keras.Model: A TensorFlow Keras model representing the discriminator architecture.
    """

    initializer = tf.random_normal_initializer(0., 0.02)  # Initialize weights

    # Input Layers
    inp = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], name='previous_frame') 
    tar = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], name='current_frame')
    diff = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], name='difference')
    inputs = [inp, tar, diff]  # Combine image inputs 

    x = tf.keras.layers.concatenate(inputs)   

    # Downsampling Blocks (Extract Features)
    down1 = downsample(64, 4, apply_batchnorm=False)(x)
    down2 = downsample(128, 4)(down1) 
    down3 = downsample(256, 4)(down2)  

    # More Convolutional Layers 
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) 
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    # Final Convolution for Decision
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  

    return tf.keras.Model(inputs=inputs, outputs=last) 

def get_generator_model(img_width, img_height, n_channels):
    """
    Creates a generator model for use in image generation tasks, likely part of a GAN system.

    This generator appears designed to take a neutral expression, a previous frame, a warped frame, 
    and distance information to synthesize new frames (perhaps for facial animation).

    Args:
        img_width (int): Width of the input and output images.
        img_height (int): Height of the input and output images.
        n_channels (int): Number of channels in the images (e.g., 3 for RGB).

    Returns:
        tf.keras.Model: A TensorFlow Keras model representing the generator architecture.
    """

    # Input Layers
    previous = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='previous_frame')
    warped = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='warped_frame')
    neutral = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='neutral_frame')
    distances = tf.keras.layers.Input(shape=[img_width, img_height, 1], dtype=tf.float64, name='distances')
    inputs = [neutral, previous, warped, distances]

    # Encoder: Downsampling Blocks
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), 
        downsample(128, 4),
        ...  # Several more downsampling blocks
    ]

    # Decoder: Upsampling Blocks 
    up_stack = [
        upsample(512, 4, apply_dropout=True),  
        upsample(512, 4, apply_dropout=True), 
        ...  # Several more upsampling blocks
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    # Final Output Layer
    last = tf.keras.layers.Conv2DTranspose(n_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') 

    x = tf.keras.layers.concatenate(inputs)  # Combine diverse inputs 

    # Encoder (Downsampling)
    skips = []  # Store for skip connections
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder (Upsampling) with Skip Connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])  # Combine with features from encoder

    x = last(x)  # Final convolution to generate the output image 

    return tf.keras.Model(inputs=inputs, outputs=x)

def get_generator_model_no_previous(img_width, img_height, n_channels):
    """
    Creates a modified generator model for image generation tasks.  This version is likely designed 
    to synthesize facial expressions without directly depending on a previous frame.

    Args:
        img_width (int): Width of the input and output images.
        img_height (int): Height of the input and output images.
        n_channels (int): Number of channels in the images (e.g., 3 for RGB).

    Returns:
        tf.keras.Model: A TensorFlow Keras model representing the generator architecture.
    """

    # Input Layers - No Previous Frame
    warped = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='warped_frame')
    neutral = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='neutral_frame')
    distances = tf.keras.layers.Input(shape=[img_width, img_height, 1], dtype=tf.float64, name='distances')
    inputs = [neutral, warped, distances] 

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(n_channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, , frames_group_size, 64, 64, 3)

    
    x = tf.keras.layers.concatenate(inputs)
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

    return tf.keras.Model(inputs=inputs, outputs=x)


def get_generator_model_neutral(img_width, img_height, n_channels):
    """
    Creates yet another modified generator model, this time focusing on a neutral expression as the core input.

    The model likely aims to synthesize modified facial expressions given a neutral face and facial distance information.

    Args:
        img_width (int): Width of the input and output images.
        img_height (int): Height of the input and output images.
        n_channels (int): Number of channels in the images (e.g., 3 for RGB).

    Returns:
        tf.keras.Model: A TensorFlow Keras model representing the generator architecture.
    """

    # Input Layers - Focused on Neutrality
    neutral = tf.keras.layers.Input(shape=[img_width, img_height, n_channels], dtype=tf.float64, name='neutral_frame')
    distances = tf.keras.layers.Input(shape=[img_width, img_height, 1], dtype=tf.float64, name='distances')
    inputs = [neutral, distances]

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(n_channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, , frames_group_size, 64, 64, 3)

    
    x = tf.keras.layers.concatenate(inputs)
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

    return tf.keras.Model(inputs=inputs, outputs=x)
