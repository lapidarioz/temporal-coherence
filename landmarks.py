from settings.facial import LOW_DEFAULT_LANDMARKS, HIGH_DEFAULT_LANDMARKS
import numpy as np
import tensorflow as tf
import mediapipe as mp
import face_recognition
from scipy.interpolate import LinearNDInterpolator
from normalize import images_from_normalized


# TODO: change to worst case landmarks
# e.g. maximize the distance between landmarks for odd and even frames
def get_default_landmarks(i=None):
    """
    Returns the default landmarks based on the given index.

    Parameters:
        i (int): Index used to determine the type of default landmarks to return.
                 If i is odd, it returns high default landmarks.
                 If i is even or None, it returns low default landmarks.

    Returns:
        numpy.ndarray: Array of default landmarks.

    """
    landmarks = np.array(LOW_DEFAULT_LANDMARKS, dtype=np.float32)
    if (i is not None) and (i % 2 == 1):
        landmarks = np.array(HIGH_DEFAULT_LANDMARKS, dtype=np.float32)
    
    return landmarks

class LandmarkDetector():
    """
    Class for detecting landmarks using the MediaPipe library.
    """

    def __init__(self, num_landmarks=68):
        """
        Initializes the LandmarkDetector object.

        Args:
            num_landmarks (int): The number of landmarks to detect. Default is 68.
        """
        self.holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        self.num_landmarks = num_landmarks
    
    def preprocess_and_detect_landmarks_numpy(self, images):
        """
        Preprocesses the images and detects the landmarks using the MediaPipe library.

        Args:
            images (numpy.ndarray): The input images.

        Returns:
            numpy.ndarray: The detected landmarks.
        """
        raise NotImplementedError

    def preprocess_and_detect_landmarks(self, images):
        """
        Preprocesses the images and detects the landmarks using the MediaPipe library.

        Args:
            images (tensorflow.Tensor): The input images.

        Returns:
            tensorflow.Tensor: The detected landmarks.
        """
        batch_size = images.shape[0]
        results = tf.numpy_function(self.preprocess_and_detect_landmarks_numpy, [images], Tout=[tf.float32])
        results = tf.convert_to_tensor(results, dtype=tf.float32)
        results.set_shape((batch_size, self.num_landmarks, 2))
        return results


class MediapipeLandmarkDetector(LandmarkDetector):
    """
    Class for detecting landmarks using the MediaPipe library.
    Inherits from the LandmarkDetector base class.
    """
    def preprocess_and_detect_landmarks_numpy(self, images):
        """
        Preprocesses the images and detects the landmarks using the MediaPipe library.

        Args:
            images (numpy.ndarray): The input images.

        Returns:
            numpy.ndarray: The detected landmarks.
        """
        images = images_from_normalized(images)
        images = np.array(images, dtype=np.uint8)
        all_landmarks = []
        for i, image in enumerate(images):
            landmarks = []
            results = self.holistic_model.process(image)
            if results.face_landmarks is None:
                landmarks = get_default_landmarks(i)
            else:
                for landmark in results.face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks, dtype=np.float32)
                landmarks = landmarks[self.landmark_points_68]
            all_landmarks.append(landmarks)
        return np.array(all_landmarks, dtype=np.float32)


def dlib_landmarks_to_array(dlib_landmarks):
    """
    Converts dlib landmarks to a numpy array.

    Args:
        dlib_landmarks: The dlib landmarks.

    Returns:
        numpy.ndarray: The converted landmarks.
    """
    landmarks = []
    for current_landmarks in dlib_landmarks:
        landmarks.append([current_landmarks.x, current_landmarks.y])
    return np.array(landmarks)

class DlibLandmarksDetector(LandmarkDetector):
    """
    A class for detecting landmarks using the Dlib library.
    """

    def preprocess_and_detect_landmarks_numpy(self, images):
        """
        Preprocesses the images and detects landmarks using the Dlib library.

        Args:
            images (list): A list of images.

        Returns:
            numpy.ndarray: An array of detected landmarks.
        """
        clip_landmarks = []
        for i, frame in enumerate(images):
            frame = images_from_normalized(frame)
            frame = np.array(frame, dtype=np.uint8)
            # Only considering the first face
            faces = face_recognition.api._raw_face_landmarks(face_image=frame)
            if len(faces) > 0:
                landmarks_array = dlib_landmarks_to_array(faces[0].parts())
            else:
                height, width = frame.shape[:2]
                landmarks_array = get_default_landmarks(i)
                landmarks_array = landmarks_array * np.array([width, height])
            clip_landmarks.append(landmarks_array)
        clip_landmarks = np.array(clip_landmarks)
        return np.ndarray.astype(clip_landmarks, dtype=np.float32)


# Overall Purpose:

# This function calculates displacement maps between corresponding facial landmarks in consecutive frames of a video. These displacement maps describe how different areas of the face move, providing detailed information about the changes in expression.

# Explanation of the Code:

# Input:

# points_a: Coordinates of facial landmarks in the first frame.
# points_b: Coordinates of the corresponding landmarks in the second frame.
# image_width, image_height: Dimensions of the video frame.
# fill_value: Value for areas outside the convex hull of the landmarks (likely areas without tracked data).
# Core Calculation:

# Loop: The function iterates through each pair of corresponding landmarks.
# Displacement Calculation: It computes the L2-norm (Euclidean distance) between each landmark pair, representing the magnitude of displacement for that point.
# Interpolation: Creates LinearNDInterpolator instances to model how the displacement values change smoothly across the whole image space, not just at the landmark locations.
# Meshgrid Creation: Sets up a grid of coordinates representing all pixels in the image.
# Displacement Map: Applies the interpolators to the meshgrid to obtain a displacement map for each landmark pair.
# Averaging: Calculates the average displacement across the two interpolators (one based on starting points, one on ending points).
# Output:

# Returns a NumPy array where each element is a displacement map, providing per-pixel displacement information across the entire image.
# Importance for Facial Expression Synthesis:

# Motion Analysis: Displacement maps provide a fine-grained representation of facial motion, which is crucial for understanding how expressions change over time.
# Data for Training: These maps could be a valuable input to your model, helping it learn the patterns associated with different expressions.
# Evaluation: Displacement maps might be used in the evaluation process to measure how accurately your model replicates realistic facial motion.
# Design Choices:

# Linear Interpolation: Assumes changes in displacement are linear between landmarks. This might be suitable for subtle expressions but less accurate for large, complex movements.
# Averaging Interpolators: Improves the robustness of the displacement estimation.
# Convex Hull Handling: Includes a fill_value to handle pixels outside the area where landmarks are tracked.
# Potential Limitations:

# Landmark Accuracy: The quality of the displacement maps is heavily dependent on the accuracy of your facial landmark detection.
# Assumption of Linearity: May not fully capture complex or non-linear motion dynamics in facial expressions.

def compute_displacements_interpolation(points_a, points_b, image_width, image_height, fill_value=0):
    """
    Compute the displacements interpolation between two sets of points.

    Args:
        points_a (ndarray): Array of shape (N, 2) representing the coordinates of points in the first set.
        points_b (ndarray): Array of shape (N, 2) representing the coordinates of points in the second set.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        fill_value (float, optional): Value to fill in for points outside the convex hull. Defaults to 0.

    Returns:
        ndarray: Array of shape (M, image_height, image_width) representing the displacements map for each pair of points.
    """
    all_displacements_map = []
    for pa, pb in zip(points_a, points_b):
        # Compute the displacements values between the two sets of points
        displacements_values = np.linalg.norm(pb - pa, axis=1)

        # Create interpolators for each set of points
        interpolator_a = LinearNDInterpolator(pa, displacements_values, fill_value=fill_value)
        interpolator_b = LinearNDInterpolator(pb, displacements_values, fill_value=fill_value)

        # Create a meshgrid of coordinates for the image
        X = np.arange(0, image_height)
        Y = np.arange(0, image_width)
        X, Y = np.meshgrid(X, Y)

        # Compute the displacements map for each set of points
        displacements_map_a = interpolator_a(X, Y)
        displacements_map_b = interpolator_b(X, Y)

        # Compute the average displacements map
        displacements_map = np.mean([displacements_map_a, displacements_map_b], axis=0)

        # Append the displacements map to the list
        all_displacements_map.append(displacements_map)

    # Convert the list of displacements maps to a numpy array
    return np.array(all_displacements_map, dtype=np.float32)

'''
LinearNDInterpolator

Purpose:

Given a set of n-dimensional data points and their corresponding function values, this interpolator creates a function that estimates the value at any other n-dimensional query point within the range of the data.
Essentially, it allows you to create a smooth surface that fits your scattered data points in a high-dimensional space.
Functionality:

Data Input:

It takes two arrays as input:
points: This n-by-d array represents the data points, where n is the number of data points and d is the dimensionality of each data point (e.g., 2D for coordinates, 3D for RGB values).
values: This 1D array with length n contains the function values corresponding to each data point in points.

Internal Delaunay Triangulation:

Behind the scenes, the interpolator utilizes a technique called Delaunay triangulation. This method divides the data points into a set of n-dimensional triangles (tetrahedrons in 3D).
Each triangle is formed by connecting a set of data points that don't share a common face (imagine non-overlapping triangles forming a mesh around the data).

Linear Interpolation on Triangles:

When you provide a new query point (also n-dimensional), the interpolator first identifies the specific triangle in the triangulation mesh that encompasses the query point.
Within that triangle, it performs linear interpolation using the values of the three (or more in higher dimensions) corner points of the triangle to estimate the function value at the query point.
This essentially creates a plane within the triangle using the corner values, and the query point's estimated value is calculated based on its position relative to the corners.

Key Points:

Linear Interpolation: It's important to remember that this method uses linear interpolation. This means it assumes a straight-line relationship between data points within each triangle, which may not always be the case for complex data.
Extrapolation vs. Interpolation: The interpolator is designed for interpolation, meaning the query point should lie within the convex hull of the data points. If you use it for extrapolation (outside the data range), the results may become unreliable.

Use Cases:

LinearNDInterpolator is a valuable tool for various tasks where you need to estimate function values from scattered data in multiple dimensions. Here are some examples:
Filling in missing data points in a dataset
Creating smooth visualizations of scattered data
source: https://tc.copernicus.org/preprints/tc-2022-39/tc-2022-39.pdf
'''

@tf.function
def get_tensors_displacements(points_a, points_b, image_width, image_height, batch_size):
    """
    Compute the displacements between two sets of points and return the results as a tensor.

    Args:
        points_a (tf.Tensor): The first set of points.
        points_b (tf.Tensor): The second set of points.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        batch_size (int): The batch size.

    Returns:
        tf.Tensor: The displacements between the two sets of points as a tensor.

    """
    [results,] = tf.numpy_function(compute_displacements_interpolation, [points_a, points_b, image_height, image_width, batch_size], Tout=[tf.float32])
    results = tf.convert_to_tensor(results, dtype=tf.float32)
    results.set_shape((batch_size, image_height, image_width))
    # add batch and channel dimensions
    # results = tf.expand_dims(results, axis=0)
    # results = tf.expand_dims(results, axis=-1)
    return results

def get_tensor_display_displacements_from_images(images_a, images_b, landmark_detector, image_width, image_height, batch_size):
    """
    Calculates the tensor display displacements between two sets of images using a landmark detector.

    Args:
        images_a (list): List of input images A.
        images_b (list): List of input images B.
        landmark_detector (object): Landmark detector object used to preprocess and detect landmarks in the images.
        image_width (int): Width of the input images.
        image_height (int): Height of the input images.
        batch_size (int): Batch size used for processing the images.

    Returns:
        list: List of tensor displacements between the input images.

    """
    # Preprocess and detect landmarks in images A
    points_a = landmark_detector.preprocess_and_detect_landmarks(images_a)

    # Preprocess and detect landmarks in images B
    points_b = landmark_detector.preprocess_and_detect_landmarks(images_b)

    # Calculate tensor displacements between points A and B
    return get_tensors_displacements(points_a, points_b, image_width, image_height, batch_size)

def add_boundary_points(points, height, width):
    """
    Adds boundary points to the given set of points.

    Args:
        points (numpy.ndarray): Array of shape (N, 2) representing the coordinates of points.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        numpy.ndarray: Array of shape (M, 2) representing the coordinates of points with added boundary points.

    """
    # Define the boundary points
    bp = np.array(
        [(0, 0),
         (width / 6, 0),
         (width / 3, 0),
         (width / 2, 0),
         (2 * width / 3, 0),
         (5 * width / 6, 0),
         (width - 1, 0),
         (width - 1, height / 6),
         (width - 1, height / 3),
         (width - 1, height / 2),
         (width - 1, 2 * height / 3),
         (width - 1, 5 * height / 6),
         (width - 1, height - 1),
         (width / 6, height - 1),
         (width / 3, height - 1),
         (width / 2, height - 1),
         (2 * width / 3, height - 1),
         (5 * width / 6, height - 1),
         (0, height - 1),
         (0, height / 6),
         (0, height / 3),
         (0, height / 2),
         (0, 2 * height / 3),
         (0, 5 * height / 6),
         ]).astype(int)
    
    if len(points.shape) == 2:
        # Concatenate the points with the boundary points
        all_points = np.concatenate((points, bp), axis=0)
        return np.ndarray.astype(all_points, dtype=np.float32)
    else:
        all_points  = []
        for i in range(points.shape[0]):
            # Concatenate each set of points with the boundary points
            p = np.concatenate((points[i], bp), axis=0)
            all_points.append(p)
        return np.array(all_points, dtype=np.float32)
