import cv2
import tensorflow as tf
import pylops
import numpy as np
from normalize import normalize_images, images_from_normalized

def denoise_video(frames):
    frames = images_from_normalized(frames)
    frames = frames.numpy()
    denoised_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        denoised_frame = cv2.fastNlMeansDenoisingColored(bgr_frame, None, 10, 10, 7, 21)
        denoised_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2RGB)
        denoised_frames.append(denoised_frame)
    denoised_frames = tf.stack(denoised_frames)
    denoised_frames = tf.cast(denoised_frames, tf.float32)
    denoised_frames = normalize_images(denoised_frames)
    return denoised_frames

def deblur_image(im): # TODO: change to use cupy
    Nz, Nx, _ = im.shape

    # Blurring guassian operator
    nh = [15, 25]
    hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.03 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imdeblur = pylops.optimization.leastsquares.normal_equations_inversion(
    Cop, im.ravel(), None, maxiter=50  # solvers need 1D arrays
    )[0]
    return imdeblur.reshape(Cop.dims)


def sharpen_video(frames):
    frames = images_from_normalized(frames)
    frames = frames.numpy()
    sharpened_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        sharpened_frame = cv2.filter2D(bgr_frame, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        sharpened_frame = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)
        sharpened_frames.append(sharpened_frame)
    sharpened_frames = tf.stack(sharpened_frames)
    sharpened_frames = tf.cast(sharpened_frames, tf.float32)
    sharpened_frames = normalize_images(sharpened_frames)
    return sharpened_frames

def deblur_video(frames):
    frames = images_from_normalized(frames)
    frames = frames.numpy()
    deblurred_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(bgr_frame.shape)
        deblurred_frame = deblur_image(bgr_frame)
        deblurred_frame = cv2.cvtColor(deblurred_frame, cv2.COLOR_BGR2RGB)
        deblurred_frames.append(deblurred_frame)
    deblurred_frames = tf.stack(deblurred_frames)
    deblurred_frames = tf.cast(deblurred_frames, tf.float32)
    deblurred_frames = normalize_images(deblurred_frames)
    return deblurred_frames

def blend_frames(previous_frame, current_frame, next_frame):
    blended_frame = (0.5*previous_frame) + (0.5*current_frame)
    blended_frame = (0.5*blended_frame) + (0.5*next_frame)
    return blended_frame

def increase_frame_rate(frames):
    frames = np.insert(frames, 0, frames[0], axis=0)
    frames = np.append(frames, frames[-1:], axis=0)
    increased_frame_rate = []
    for i in range(len(frames)-2):
        previous_frame = frames[i]
        current_frame = frames[i+1]
        next_frame = frames[i+2]
        blended_frame = blend_frames(previous_frame, current_frame, next_frame)
        increased_frame_rate.append(blended_frame)
        increased_frame_rate.append(current_frame)
    increased_frame_rate = tf.stack(increased_frame_rate)
    return increased_frame_rate
