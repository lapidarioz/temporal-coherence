import numpy as np
from pathlib import Path
from math import sqrt, atan, asin, pi, sin, cos
from settings.facial import FACIAL_FEATURES_LINES
import pandas as pd


def curve_model_variables(landmarks, im_shape):
    # this function calculates geometric properties of a curve (represented by the landmarks input).

    # Get image dimensions
    h, w, _ = im_shape  
    m = w  # Assign image width to 'm'
    n = h  # Assign image height to 'n'

    # Prepare a copy of landmarks, adjusted for coordinate system. 
    # It seems the y-axis might be inverted in the image representation
    transformed_landmarks = np.copy(landmarks) 
    transformed_landmarks[:,1] = h - transformed_landmarks[:,1] 

    # Get landmark boundary points
    x_start = transformed_landmarks[0,0]  # X-coordinate of start
    x_end = transformed_landmarks[-1,0]   # X-coordinate of end
    y_start = transformed_landmarks[0,1]  # Y-coordinate of start
    y_end = transformed_landmarks[-1,1]   # Y-coordinate of end

    # ---- Calculate Curve Length and Orientation ----

    x_difference = x_end - x_start
    y_difference = x_end - x_start 
    l = sqrt((x_difference) ** 2.0 + (y_difference) ** 2.0)  # Length 'l' of the curve
    phi = atan((y_difference) / (x_difference))            # Angle 'phi' of the curve

    # ---- Find Extrema Points Along the Curve ----

    middle_points = transformed_landmarks[1:-1]      # Exclude start/end for extrema
    middle_points_y = middle_points[:,1]             # Extract Y-coords of middle points
    argmax_y = np.argmax(middle_points_y)            # Index of highest Y-coord
    argmin_y = np.argmin(middle_points_y)            # Index of lowest Y-coord

    x_highest, y_highest = middle_points[argmax_y]   # Highest point
    x_lowest, y_lowest = middle_points[argmin_y]    # Lowest point

    # ---- Calculate Additional Geometric Properties ----

    xa, ya = x_lowest, y_lowest                
    v_height = y_highest                   # Vertical height (peak height)
    v_skew = x_highest - (l / 2)           # Vertical skew (offset of peak from midpoint)

    theta = asin(np.clip(v_skew/v_height, -1, 1))  # Angle 'theta' (likely related to curvature)

    ya_prime = ya
    xa_prime = (xa - ya * sin(theta)) / cos(theta)  # Some coordinate transformation...
    d = l / (2 * cos(theta))                         # ... (more transformation)

    # ... (more complex calculation of 'k' - this might be a curvature parameter)
    k_dividend = ((v_height ** 2.0) * ya_prime) - (v_height * (ya_prime **  2.0))
    k_divisor = (v_height * ((xa_prime - d) ** 2.0)) + ((ya_prime - v_height) * (d ** 2.0))
    k_division = max(0, k_dividend / k_divisor) if k_divisor != 0 else 0
    k = sqrt(k_division)

    # ---- Return Calculated Values ----
    return m, n, x_start, y_start, l, phi, v_height, v_skew, k 

   
def curve_model_g(landmarks, im_shape):
   # 1. Calculate geometric properties
   m, n, x_start, y_start, l, phi, v_height, v_skew, k = curve_model_variables(landmarks, im_shape)

   # 2. Normalize properties 
   return np.array([
       x_start / m,           # Normalize start point X-coordinate by image width
       y_start / n,           # Normalize start point Y-coordinate by image height
       l / m,                 # Normalize curve length by image width
       phi,                   # Angle (likely in radians) doesn't need normalization
       v_height / l,          # Normalize vertical height relative to curve length
       v_skew / l,            # Normalize vertical skew relative to curve length
       k                      # Curvature parameter (might not need normalization)
   ])


def curve_model_s(image, landmarks):
    im_shape = image.shape
    
    # Calculate normalized geometric features for each facial feature
    g_right_brow = curve_model_g(landmarks[FACIAL_FEATURES_LINES['right_brow']], im_shape)
    g_left_brow = curve_model_g(landmarks[FACIAL_FEATURES_LINES['left_brow']], im_shape)
    g_lower_nose = curve_model_g(landmarks[FACIAL_FEATURES_LINES['lower_nose']], im_shape)
    g_upper_right_eye = curve_model_g(landmarks[FACIAL_FEATURES_LINES['upper_right_eye']], im_shape)
    g_lower_right_eye = curve_model_g(landmarks[FACIAL_FEATURES_LINES['lower_right_eye']], im_shape)
    g_upper_left_eye = curve_model_g(landmarks[FACIAL_FEATURES_LINES['upper_left_eye']], im_shape)
    g_lower_left_eye = curve_model_g(landmarks[FACIAL_FEATURES_LINES['lower_left_eye']], im_shape)
    g_upper_upper_lip = curve_model_g(landmarks[FACIAL_FEATURES_LINES['upper_upper_lip']], im_shape)
    g_lower_upper_lip = curve_model_g(landmarks[FACIAL_FEATURES_LINES['lower_upper_lip']], im_shape)
    g_upper_lower_lip = curve_model_g(landmarks[FACIAL_FEATURES_LINES['upper_lower_lip']], im_shape)
    g_lower_lower_lip = curve_model_g(landmarks[FACIAL_FEATURES_LINES['lower_lower_lip']], im_shape)

    # Extract geometric properties of the facial outline 
    m, n, x_start, y_start, l, phi, v_height, v_skew, k = curve_model_variables(landmarks[FACIAL_FEATURES_LINES['facial_outline']], im_shape)
    l_squared = l ** 2.0
    facial_outline_params = np.array([
        l_squared / m,  # Scale-invariant measure related to outline size 
        phi ** 2.0,     # Likely related to overall face orientation
        (v_height ** 2.0) / (l_squared),
        (v_skew ** 2.0) / (l_squared),
        k ** 2.0
    ])
    
    # Calculate differences between feature geometries
    return (
    facial_outline_params,
    g_upper_right_eye - g_right_brow,
    g_upper_left_eye - g_left_brow,
    g_lower_right_eye - g_upper_right_eye,
    g_lower_left_eye - g_upper_left_eye,
    g_lower_nose - g_lower_right_eye,
    g_lower_nose - g_lower_left_eye,
    g_upper_upper_lip - g_lower_nose,
    g_upper_upper_lip - g_lower_upper_lip,
    g_lower_upper_lip - g_upper_lower_lip,
    g_upper_lower_lip - g_lower_lower_lip
    )
