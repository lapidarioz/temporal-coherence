import cv2 
import numpy as np
from settings.facial import DEFAULT_TRIANGULATION, MOUTH_OPEN_TRIANGULATION
from landmarks import add_boundary_points

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Applies an affine transformation to an image to warp it based on corresponding landmarks.

    Args:
        src (np.ndarray): The source image.
        src_tri (np.ndarray): An array of 3 points representing the source landmarks. 
                              These points likely correspond to facial features.
        dst_tri (np.ndarray): An array of 3 points representing the destination landmarks.
                              These points should align with the src_tri points in the desired warped image.
        size (tuple): The desired output image size (width, height).

    Returns:
        np.ndarray: The warped image.
    """

    # Calculate the affine transformation matrix
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri)) 

    # Apply the transformation to the image
    return cv2.warpAffine(src=src,
                          M=warp_mat,
                          dsize=(size[0], size[1]), 
                          dst=None,  
                          flags=cv2.INTER_LINEAR,  # Linear interpolation method
                          borderMode=cv2.BORDER_REFLECT_101)  # Handles pixels beyond image boundaries 


def warp_triangle(img1, img2, t1, t2):
    """
    Warps a triangular region from one image (img1) into another image (img2). 

    Args:
        img1 (np.ndarray): The source image.
        img2 (np.ndarray): The destination image.
        t1 (list or np.ndarray): A list of three (x, y) coordinates representing the source triangle.
        t2 (list or np.ndarray): A list of three (x, y) coordinates representing the destination triangle.

    Returns:
        np.ndarray: The modified destination image (img2) with the warped triangular region from img1. 
    """

    # Find bounding rectangles for triangles to isolate the relevant regions
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Calculate offsets for triangle points within their bounding rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []
    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))  
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))  

    # Create a triangular mask for blending
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.uint8)  
    cv2.fillConvexPoly(img=mask,
                       points=np.int32(t2_rect_int),  
                       color=(1.0, 1.0, 1.0),  
                       lineType=cv2.LINE_AA, 
                       shift=0) 

    # Extract the rectangular region from the source image (img1)
    img1_rect = np.array(img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]])

    # Apply affine transformation to the extracted patch
    if len(img1_rect) > 0:  # Check if the patch exists within the image bounds
        rect_shape = (r2[2], r2[3])
        img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, rect_shape)

        # Blend the transformed patch into the destination image using the mask
        img2_rect = img2_rect * mask 
        try:
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

        except ValueError:
            pass # TODO: Fix this 
    return img2 


def copy_image(image):
    """
    Creates a deep copy of an image, ensuring safe modification without altering the original.

    Args:
        image (np.ndarray or other image-like object): The image to be copied.

    Returns:
        np.ndarray: A new, independent copy of the input image.
    """

    if type(image) == np.ndarray:  # Case 1: Image is already a NumPy array
        return image.copy()  # Efficiently create a deep copy

    else:  # Case 2: Assume image-like object with a `.numpy()` method
        return image.numpy().copy()  # Convert to NumPy array and create a deep copy


def warp_all(input_image, input_triangles, new_triangles):
    """
    Applies a piecewise image warping by warping multiple triangular regions.

    Args:
        input_image (np.ndarray): The input image to be warped.
        input_triangles (list): A list of triangles (each triangle represented as a 
                                list of three (x, y) points) in the input image.
        new_triangles (list): A list of corresponding triangles in the desired warped output.

    Returns:
        np.ndarray: The warped version of the input image.
    """

    # Create a deep copy of the input image to work on
    input_image_warped = copy_image(input_image) 

    # Iterate through corresponding triangle pairs 
    for current_input_triangle, current_new_triangle in zip(input_triangles, new_triangles): 
        # Warp each triangular region using 'warp_triangle'
        warp_triangle(input_image, input_image_warped, current_input_triangle, current_new_triangle) 

    # Return the final warped image 
    return input_image_warped 


def get_new_landmarks(source_landmarks_previous, source_landmarks_current, target_landmarks_current):
    """
    Calculates predicted landmarks for manipulating facial features. 

    This function likely assumes that the change in position between landmarks in the source 
    image (from previous to current) should be mirrored in the target image based on its current landmarks.

    Args:
        source_landmarks_previous (np.ndarray): Facial landmarks for the source image in a previous frame.
        source_landmarks_current (np.ndarray): Facial landmarks for the source image in the current frame.
        target_landmarks_current (np.ndarray): Facial landmarks for the target image in the current frame.

    Returns:
        np.ndarray: Predicted new landmarks for the target image.
    """

    # Calculate the displacement vector between source landmarks in the previous and current frames
    displacement = np.subtract(source_landmarks_current, source_landmarks_previous) 

    # Apply the same displacement to the current target landmarks to estimate new positions
    return np.add(displacement, target_landmarks_current) 


def warp_mouth(target_image, target_landmarks, source_image, source_landmarks):
    """
    Warps the mouth region of a target image based on the mouth landmarks from a source image.

    Args:
        target_image (np.ndarray): The image to be modified.
        target_landmarks (np.ndarray): Facial landmarks for the target image.
        source_image (np.ndarray): The image containing the desired mouth shape.
        source_landmarks (np.ndarray): Facial landmarks for the source image.

    Returns:
        np.ndarray: The target image with the warped mouth region.
    """  

    target_image_warped = copy_image(target_image)  # Create a copy to work on

    # Isolate triangles corresponding to the mouth region
    source_triangles = source_landmarks[MOUTH_OPEN_TRIANGULATION]
    target_triangles = target_landmarks[MOUTH_OPEN_TRIANGULATION]

    # Warp each mouth triangle individually 
    for current_source_triangle, current_target_triangle in zip(source_triangles,target_triangles):
        warp_triangle(source_image, target_image_warped, current_source_triangle, current_target_triangle)

    return target_image_warped 


def deform(input_image, source_image, input_landmarks, previous_source_landmarks, current_source_landmarks):
    """
    Deforms an input image to match the facial expression of a source image.

    Args:
        input_image (np.ndarray): The image to be deformed.
        source_image (np.ndarray): The image containing the desired facial expression.
        input_landmarks (np.ndarray): Facial landmarks on the input image.
        previous_source_landmarks (np.ndarray): Facial landmarks on the source image in a previous frame.
        current_source_landmarks (np.ndarray): Facial landmarks on the source image in the current frame.

    Returns:
        tuple: 
            - np.ndarray: The deformed input image
            - np.ndarray: The updated landmarks for the deformed image
    """

    # Add boundary points to the landmarks for consistency (likely for warping edges)
    input_landmarks = add_boundary_points(input_landmarks, input_image.shape[0], input_image.shape[1]) 
    previous_source_landmarks = add_boundary_points(previous_source_landmarks, input_image.shape[0], input_image.shape[1]) 
    current_source_landmarks = add_boundary_points(current_source_landmarks, input_image.shape[0], input_image.shape[1]) 

    # Extract triangles from input landmarks based on predefined triangulation
    input_triangles = input_landmarks[DEFAULT_TRIANGULATION] 

    # Calculate new landmarks to deform the input image
    new_landmarks = get_new_landmarks(previous_source_landmarks, current_source_landmarks, input_landmarks) 

    # Extract the corresponding triangles from the new landmarks
    new_triangles = new_landmarks[DEFAULT_TRIANGULATION] 

    # Perform global warping based on facial features
    warped_image =  warp_all(input_image, input_triangles, new_triangles) 

    # Further refine the mouth region
    deformed_image, new_landmarks = warp_mouth(warped_image, new_landmarks, source_image, current_source_landmarks)

    return deformed_image, new_landmarks 

