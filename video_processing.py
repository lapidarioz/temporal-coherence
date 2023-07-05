import cv2
import tensorflow as tf

def denoise_video(frames):
    frames = frames.numpy()
    denoised_frames = []
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        denoised_frame = cv2.fastNlMeansDenoisingColored(bgr_frame, None, 10, 10, 7, 21)
        denoised_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2RGB)
        #to unit8
        denoised_frames.append(denoised_frame)
    denoised_frames = tf.stack(denoised_frames)
    denoised_frames = tf.cast(denoised_frames, tf.uint8)
    return denoised_frames
