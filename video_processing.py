import cv2
import tensorflow as tf
import pylops
import numpy as np

def denoise_video(frames):
    frames = frames.numpy()
    denoised_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        denoised_frame = cv2.fastNlMeansDenoisingColored(bgr_frame, None, 10, 10, 7, 21)
        denoised_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2RGB)
        denoised_frames.append(denoised_frame)
    denoised_frames = tf.stack(denoised_frames)
    denoised_frames = tf.cast(denoised_frames, tf.uint8)
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
    frames = frames.numpy()
    sharpened_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        sharpened_frame = cv2.filter2D(bgr_frame, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        sharpened_frame = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)
        sharpened_frames.append(sharpened_frame)
    sharpened_frames = tf.stack(sharpened_frames)
    sharpened_frames = tf.cast(sharpened_frames, tf.uint8)
    return sharpened_frames

def deblur_video(frames):
    frames = frames.numpy()
    deblurred_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        deblurred_frame = deblur_image(bgr_frame)
        deblurred_frame = cv2.cvtColor(deblurred_frame, cv2.COLOR_BGR2RGB)
        deblurred_frames.append(deblurred_frame)
    deblurred_frames = tf.stack(deblurred_frames)
    deblurred_frames = tf.cast(deblurred_frames, tf.uint8)
    return deblurred_frames        
