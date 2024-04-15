import tensorflow as tf

def normalize_images(sequence):
    """
    Normalizes a sequence of images (likely an array or tensor of images) to the range [-1, 1].

    Args:
        sequence (tf.Tensor or array-like): The input image data.

    Returns:
        tf.Tensor: The normalized images within the range [-1, 1]. 
    """

    return (sequence / 127.5) - 1 

@tf.function  
def images_from_normalized(sequence):
    """
    Converts normalized images (in the range [-1, 1]) back to the standard [0, 255] range.

    Args:
        sequence (tf.Tensor or array-like): The normalized image data.

    Returns:
        tf.Tensor: The denormalized images with pixel values in [0, 255], cast as uint8 data type. 
    """

    return tf.cast((sequence + 1) * 127.5, tf.uint8)  
