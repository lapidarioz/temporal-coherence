import numpy as np
from pathlib import Path
from math import sqrt, atan, asin, pi, sin, cos
from settings.facial import FACIAL_FEATURES_LINES
import pandas as pd


def curve_model_variables(landmarks, im_shape):
    h, w, _ = im_shape
    m = w
    n = h
    transformed_landmarks = np.copy(landmarks)
    transformed_landmarks[:,1] = h - transformed_landmarks[:,1]
    x_start = transformed_landmarks[0,0]
    x_end = transformed_landmarks[-1,0]
    y_start = transformed_landmarks[0,1]
    y_end = transformed_landmarks[-1,1]
    x_difference = x_end - x_start
    y_difference = x_end - x_start
    l = sqrt((x_difference) ** 2.0 + (y_difference) ** 2.0)
    phi = atan((y_difference) / (x_difference))
    middle_points = transformed_landmarks[1:-1]
    middle_points_y = middle_points[:,1]
    argmax_y = np.argmax(middle_points_y)
    argmin_y = np.argmin(middle_points_y)
    x_highest, y_highest = middle_points[argmax_y]
    x_lowest, y_lowest = middle_points[argmin_y]
    xa, ya = x_lowest, y_lowest
    v_height = y_highest
    v_skew = x_highest - (l / 2)
    theta = asin(np.clip(v_skew/v_height, -1, 1))
    ya_prime = ya
    xa_prime = (xa - ya * sin(theta)) / cos(theta)
    d = l / (2 * cos(theta)) 
    k_dividend = ((v_height ** 2.0) * ya_prime) - (v_height * (ya_prime **  2.0))
    k_divisor = (v_height * ((xa_prime - d) ** 2.0)) + ((ya_prime - v_height) * (d ** 2.0))
    k_division = max(0, k_dividend / k_divisor) if k_divisor != 0 else 0
    k = sqrt(k_division)
    return m, n, x_start, y_start, l, phi, v_height, v_skew, k
   
def curve_model_g(landmarks, im_shape):
    m, n, x_start, y_start, l, phi, v_height, v_skew, k = curve_model_variables(landmarks, im_shape)
    return np.array([
    x_start / m,
    y_start / n,
    l / m,
    phi,
    v_height / l,
    v_skew / l,
    k
    ])

def curve_model_s(image, landmarks):
    im_shape = image.shape
    
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

    m, n, x_start, y_start, l, phi, v_height, v_skew, k = curve_model_variables(landmarks[FACIAL_FEATURES_LINES['facial_outline']], im_shape)
    l_squared = l ** 2.0
    facial_outline_params = np.array([
        l_squared / m,
        phi ** 2.0,
        (v_height ** 2.0) / (l_squared),
        (v_skew ** 2.0) / (l_squared),
        k ** 2.0
    ])
    
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
