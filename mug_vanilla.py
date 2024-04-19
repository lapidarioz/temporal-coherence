# %%
import tensorflow as tf

import os
from pathlib import Path
import datetime
import numpy as np
import random


from IPython.display import display, clear_output
from tqdm.notebook import tqdm, trange


# from fid import get_frechet_inception_distance
import pandas as pd

from normalize import images_from_normalized
from data_generator import PreprocessedDataGenerator
from plot import plot_results, save_gif
from model import get_generator_model, get_discriminator_model
from loss import l1_loss, euclidean_distance, GeneratorLoss, DiscriminatorLoss, pairwise_loss
from splits import QUALITATIVE_TRAIN_PATHS, QUALITATIVE_TEST_PATHS, LAPIS_TEST_PATHS, TRAIN_PATHS, TEST_PATHS
from settings.expresions import FACIAL_EXPRESSION_NAMES

# %%

# Vanilla Version:  The vanilla version likely focuses on the core task of facial expression synthesis – generating a single modified target frame for a given input. Its emphasis is on image quality and accuracy rather than smooth motion across a video sequence.

tf.__version__

# %%
TRAIN = True
TEST = True
TEST_VIDEO = False
PLOT = False
SAVE_GIFS = False
TEST_DEFORMATIONS = False
SAVE_GIFS_SIMILAR = False
SAVE_GIFS_LAPIS_TEST = False
QUALITATIVE = False

COHERENCE_LOSS = None
TARGET_PREDICTED_LOSS = l1_loss
LANDMARKS_LOSS = None
LANDMARKS_COHERENCE_LOSS = None
INTERPOLATION_FRAMES_LOSS = None
INTERPOLATION_LANDMARKS_LOSS = None

BLEND_DEFORMATIONS = False

# Dimensions
r = 256
IMG_WIDTH = r
IMG_HEIGHT = r
N_CHANNELS = 3

N_LANDMARKS = 68
BATCH_SIZE = 10
# STEPS = 100000
STEPS = 100 * 2
STEPS = STEPS // BATCH_SIZE
SAVING_EACH_STEPS = 50
DISPLAY_EACH_STEPS = 15
# N_TESTING_FRAMES = 42000
N_TESTING_FRAMES = 9600

LAMBDA_TARGET_PREDICTED_LOSS = 1000
LAMBDA_COHERENCE_LOSS = 0
LAMBDA_LANDMARKS_LOSS = 0
LAMBDA_LANDMARKS_COHERENCE_LOSS = 0
LAMBDA_PERCEPTUAL_LOSS = 0
LAMBDA_INTERPOLATION_FRAMES_LOSS = 0
LAMBDA_INTERPOLATION_LANDMARKS_LOSS = 0

VIDEOS_FOLDER = f'mug{r}'
BASE_PATH = Path('../')
DATA_PATH = BASE_PATH / 'data'
APP_PATH = BASE_PATH / 'app'
if QUALITATIVE:
    OUTPUT_PATH = APP_PATH / 'qualitative_output'
else:
    OUTPUT_PATH  = APP_PATH / 'quantitative_output'

default_dimension = "mug128"

# %%
id_string = f"vanilla_b{BATCH_SIZE}_r{r}_"
if TARGET_PREDICTED_LOSS:
    id_string += f"f1_{TARGET_PREDICTED_LOSS.__name__}_{LAMBDA_TARGET_PREDICTED_LOSS}"
if COHERENCE_LOSS:
    id_string += f"_f2_{COHERENCE_LOSS.__name__}_{LAMBDA_COHERENCE_LOSS}"
if LANDMARKS_LOSS:
    id_string += f"_f3_{LANDMARKS_LOSS.__name__}_{LAMBDA_LANDMARKS_COHERENCE_LOSS}"
if LANDMARKS_COHERENCE_LOSS:
    id_string += f"_f4_{LANDMARKS_COHERENCE_LOSS.__name__}_{LAMBDA_LANDMARKS_COHERENCE_LOSS}"
if LAMBDA_PERCEPTUAL_LOSS:
    id_string += f"_f5_perceptual_{LAMBDA_PERCEPTUAL_LOSS}"
if INTERPOLATION_FRAMES_LOSS:
    id_string += f"_f6_{INTERPOLATION_FRAMES_LOSS.__name__}_{LAMBDA_INTERPOLATION_FRAMES_LOSS}"
if INTERPOLATION_LANDMARKS_LOSS:
    id_string += f"_f7_{INTERPOLATION_LANDMARKS_LOSS.__name__}_{LAMBDA_INTERPOLATION_LANDMARKS_LOSS}"
id_string


# %%
OUTPUT_MEDIA_FOLDER = OUTPUT_PATH / id_string / 'media'
OUTPUT_MEDIA_FOLDER.mkdir(parents=True, exist_ok=True)

# %%
log_dir = OUTPUT_MEDIA_FOLDER / "logs/"
date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = log_dir / "fit" / date_str

# %% [markdown]
# ## Load the dataset

# %%
if QUALITATIVE:
    train_paths = QUALITATIVE_TRAIN_PATHS
    test_paths = QUALITATIVE_TEST_PATHS
else:
    train_paths = TRAIN_PATHS
    test_paths = TEST_PATHS

# %%
def replace_in_path(paths, default_dimension, new_dimension):
    return np.array([p.replace(default_dimension,new_dimension) for p in paths])


# %%
train_paths = replace_in_path(train_paths, default_dimension, f"mug{r}")
test_paths = replace_in_path(test_paths, default_dimension, f"mug{r}")

# %%
generator = get_generator_model(IMG_WIDTH, IMG_HEIGHT, N_CHANNELS)
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file=str(OUTPUT_MEDIA_FOLDER / 'generator.png'))

# %%
discriminator = get_discriminator_model(IMG_WIDTH, IMG_HEIGHT, N_CHANNELS)
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file=str(OUTPUT_MEDIA_FOLDER / 'discriminator.png'))

# %%
generator_loss = GeneratorLoss(
    # target predicted loss
    target_predicted_function=TARGET_PREDICTED_LOSS,
    lambda_target_predicted=LAMBDA_TARGET_PREDICTED_LOSS,
    # coherence loss
    coherence_loss_function=COHERENCE_LOSS,
    lambda_coherence_loss=LAMBDA_COHERENCE_LOSS,
    # landmarks loss
    landmarks_loss_function=LANDMARKS_LOSS,
    lambda_landmarks_loss=LAMBDA_LANDMARKS_LOSS,
    # landmarks coherence loss
    landmarks_coherence_loss_function=LANDMARKS_COHERENCE_LOSS,
    lambda_landmarks_coherence_loss=LAMBDA_LANDMARKS_COHERENCE_LOSS,
    # perceptual loss
    lambda_perceptual_loss=LAMBDA_PERCEPTUAL_LOSS,
    # interpolation frames loss
    interpolation_frames_loss_function=INTERPOLATION_FRAMES_LOSS,
    lambda_interpolation_frames_loss=LAMBDA_INTERPOLATION_FRAMES_LOSS,
    # interpolation landmarks loss
    interpolation_landmarks_loss_function=INTERPOLATION_LANDMARKS_LOSS,
    lambda_interpolation_landmarks_loss=LAMBDA_INTERPOLATION_LANDMARKS_LOSS
)
                    

# %%
discriminator_loss = DiscriminatorLoss()

# %%
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# %%
BASE_DIR = OUTPUT_PATH / id_string
checkpoint_dir = str(BASE_DIR / 'training_checkpoints')
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

# %% [markdown]
# ## Training
# 

# %%
summary_writer = tf.summary.create_file_writer(str(log_file))

# %%
def fit_deformed(train_dataset_generator, test_dataset_generator, steps):
  pbar = tqdm(range(steps), total=steps)
  for step in pbar:
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):

      if (step) % DISPLAY_EACH_STEPS == 0:
        clear_output(wait=True)
        display(pbar.container)
        previously_generated, generated_frames, samples_target = test_dataset_generator.generate_first_batch_deformed()
        plot_results(samples_target, generated_frames)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_total_loss, disc_loss, gen_losses = train_dataset_generator.next_loss_deformed()
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))
        with summary_writer.as_default():
          tf.summary.scalar(f'gen_total_loss', gen_total_loss, step=step)
          tf.summary.scalar(f'disc_loss', disc_loss, step=step)
          for name, value in gen_losses.items():
            tf.summary.scalar(f'gen_{name}', value, step=step)
      
      if (step + 1) % SAVING_EACH_STEPS == 0:
          checkpoint_manager.save()

# %%
def fit_previous(train_dataset_generator, test_dataset_generator, steps):
  pbar = tqdm(range(steps), total=steps)
  for step in pbar:
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):

      if (step) % DISPLAY_EACH_STEPS == 0:
        clear_output(wait=True)
        display(pbar.container)
        previously_generated, generated_frames, samples_target = test_dataset_generator.generate_first_batch()
        plot_results(samples_target, generated_frames)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_total_loss, disc_loss, gen_losses = train_dataset_generator.next_loss()
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))
        with summary_writer.as_default():
          tf.summary.scalar(f'gen_total_loss', gen_total_loss, step=step)
          tf.summary.scalar(f'disc_loss', disc_loss, step=step)
          for name, value in gen_losses.items():
            tf.summary.scalar(f'gen_{name}', value, step=step)
      
      if (step + 1) % SAVING_EACH_STEPS == 0:
          checkpoint_manager.save()

# %%
if TRAIN:
    tf.profiler.experimental.start(str(log_dir))

# %%
if TRAIN:
    train_generator = PreprocessedDataGenerator(train_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=True, blend_deformation=BLEND_DEFORMATIONS)
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    fit_deformed(train_generator, test_generator, steps=STEPS)

# %%
if TRAIN:    
    checkpoint_manager.save()

# %% [markdown]
# ## Restore the latest checkpoint and test the network

# %%
# !ls {checkpoint_dir}

# %%
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# %% [markdown]
# ## Generate some images using the test set

# %%
test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
for i in range(0,10):
  generated_frames, samples_target = test_generator.generate_next_batch()
  clear_output(wait=True)
  plot_results(samples_target, generated_frames)

# %%
if SAVE_GIFS:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, save_path=OUTPUT_MEDIA_FOLDER, blend_deformation=BLEND_DEFORMATIONS)
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                test_generator.save_next_video()
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break

# %%
if SAVE_GIFS_SIMILAR:
    OUTPUT_MEDIA_SIMILAR_FOLDER = OUTPUT_PATH / id_string / 'media_similar'
    OUTPUT_MEDIA_SIMILAR_FOLDER.mkdir(parents=True, exist_ok=True)
    for expression_name in FACIAL_EXPRESSION_NAMES:
        test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, save_path=OUTPUT_MEDIA_SIMILAR_FOLDER, blend_deformation=BLEND_DEFORMATIONS, search_similar=True, expression_name=expression_name)
        with tqdm(total=N_TESTING_FRAMES) as pbar:
            while True:
                try:
                    clear_output(wait=True)
                    test_generator.save_next_video()
                    pbar.update(BATCH_SIZE)
                except StopIteration:
                    break

# %%
# adapted from https://gist.github.com/lukoshkin/6205455d5b931dcdbd923323003058f0 
# and https://github.com/valerystrizh/mocogan_modified/blob/0860daaa6564b563b5acc1d88f19bac051c1d1c6/src/metrics.py
def acd(frames):
    N = np.multiply.reduce(frames.shape[1:-1])
    res = np.mean(
            np.linalg.norm(
              np.diff(
                np.einsum('ijkl->il', frames), 
              axis=0) / N, 
            axis=1)
          ) 

    return res

def compare_acd(target_sequence, generated_output):
    return np.abs(acd(target_sequence) - acd(generated_output))

# %%
# evaluating the results with SSIM
def evaluate_metrics(normalized_target_sequence, normalized_generated_output, batch_size):
    target_sequence = images_from_normalized(normalized_target_sequence)
    generated_output = images_from_normalized(normalized_generated_output)
    #cast to float32
    normalized_target_sequence = tf.cast(normalized_target_sequence, tf.float32)
    normalized_generated_output = tf.cast(normalized_generated_output, tf.float32)
    #compute ssim
    ssim_scores = tf.image.ssim(target_sequence, generated_output, max_val=255)
    #compute psnr
    psnr_scores = tf.image.psnr(target_sequence, generated_output, max_val=255)
    #compute mse
    mse_scores = tf.keras.losses.MSE(normalized_target_sequence, normalized_generated_output)
    #compute l1
    l1_scores = tf.keras.losses.MAE(normalized_target_sequence, normalized_generated_output)
    # compute fid
    # fid_scores = get_frechet_inception_distance(target_sequence, generated_output, batch_size=batch_size, num_inception_images=batch_size) # TODO: pre frame
    # compute fvd
    # fvd_scores = calculate_fvd(target_sequence, generated_output)
    # compute acd
    acd_scores = compare_acd(target_sequence, generated_output) # TODO: per video
    return  {
        "SSIM": tf.reduce_mean(ssim_scores).numpy(),
        "PSNR": tf.reduce_mean(psnr_scores).numpy(),
        "MSE": tf.reduce_mean(mse_scores).numpy(),
        "L1": tf.reduce_mean(l1_scores).numpy(),
        # "FVD": fvd_scores
        # "FID": fid_scores.numpy(),
        "ACD": acd_scores
    }

def compute_scores():
    scores = []
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                display(pbar.container)
                df = pd.DataFrame(scores)
                df.to_csv(f"{log_dir}/metrics.csv")
                print(df.mean())
                print(df.std())
                generated_frames, current_frames = test_generator.generate_next_batch()
                s = evaluate_metrics(current_frames, generated_frames, batch_size=BATCH_SIZE)
                scores.append(s)
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break
    return scores

# %%
if TEST:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    scores_path = f"{log_dir}/metrics.csv"
    # df = pd.read_csv(scores_path, index_col=0)  
    scores = compute_scores()
    df = pd.DataFrame(scores)
    df.to_csv(scores_path)

# %%
if TEST:
    print(df)

# %%
if TEST:
    print(df.mean())

# %%
if TEST:
    print(df.std())

# %% [markdown]
# # coherence measures

# %%
def compute_scores_diff():
    scores = []
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                display(pbar.container)
                df = pd.DataFrame(scores)
                df.to_csv(f"{log_dir}/diff_metrics.csv")
                print(df.mean())
                print(df.std())
                _, generated_frames, current_frames = test_generator.generate_diff_batch()
                s = evaluate_metrics(current_frames, generated_frames, batch_size=BATCH_SIZE)
                scores.append(s)
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break
    return scores

if TEST:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    scores_path = f"{log_dir}/diff_metrics.csv"
    # df = pd.read_csv(scores_path, index_col=0)  
    scores = compute_scores_diff()
    df = pd.DataFrame(scores)
    df.to_csv(scores_path)

# %%
if TEST:
    print(df.mean())

# %%
if TEST:
    print(df.std())

# %% [markdown]
# # Plot some images

# %%
if PLOT:
  test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
  for i in range(0,10):
    clear_output(wait=True)
    test_generator.next_plot()

# %% [markdown]
# # Save videos gifs

# %% [markdown]
# # Test per video

# %%
def compute_scores_video():
    scores = []
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                display(pbar.container)
                df = pd.DataFrame(scores)
                df.to_csv(f"{log_dir}/video_metrics.csv")
                if "path_id" in df.columns:
                    print(df.drop(columns=["path_id"], inplace=False).mean())
                    print(df.drop(columns=["path_id"], inplace=False).std())
                current_frames, generated_frames, path_id = test_generator.generate_next_video() # TODO: blended frames
                if current_frames is not None and generated_frames is not None:
                    video_size = generated_frames.shape[1]
                    s = evaluate_metrics(current_frames, generated_frames, batch_size=video_size)
                    s["path_id"] = path_id
                    scores.append(s)
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break
    return scores

# %%
if TEST_VIDEO:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    scores_path = f"{log_dir}/video_metrics.csv"
    # df = pd.read_csv(scores_path, index_col=0)  
    scores = compute_scores_video()
    df = pd.DataFrame(scores)
    df.to_csv(scores_path)

# %%
if TEST_VIDEO:
    print(df.drop(columns=["path_id"], inplace=False).mean())

# %%
if TEST_VIDEO:
    print(df.drop(columns=["path_id"], inplace=False).std())

# %%
def compute_video_scores_diff():
    scores = []
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                display(pbar.container)
                df = pd.DataFrame(scores)
                df.to_csv(f"{log_dir}/video_diff_metrics.csv")
                if "path_id" in df.columns:
                    print(df.drop(columns=["path_id"], inplace=False).mean())
                    print(df.drop(columns=["path_id"], inplace=False).std())
                current_frames, generated_frames, path_id = test_generator.generate_diff_video() # TODO: blended frames
                if current_frames is not None and generated_frames is not None:
                    video_size = generated_frames.shape[1]
                    s = evaluate_metrics(current_frames, generated_frames, batch_size=video_size)
                    s["path_id"] = path_id
                    scores.append(s)
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break
    return scores

if TEST_VIDEO:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    scores_path = f"{log_dir}/video_diff_metrics.csv"
    # df = pd.read_csv(scores_path, index_col=0)  
    scores = compute_video_scores_diff()
    df = pd.DataFrame(scores)
    df.to_csv(scores_path)

# %%
if TEST_VIDEO:
    print(df.drop(columns=["path_id"], inplace=False).mean())

# %%
if TEST_VIDEO:
    print(df.drop(columns=["path_id"], inplace=False).std())

# %% [markdown]
# # Test deformation

# %%
def compute_deformation_scores():
    scores = []
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                display(pbar.container)
                df = pd.DataFrame(scores)
                df.to_csv(f"{log_dir}/deformed_metrics.csv")
                print(df.mean())
                print(df.std())
                deformed_frames, current_frames = test_generator.deformed_next_batch()
                s = evaluate_metrics(current_frames, deformed_frames, batch_size=BATCH_SIZE)
                scores.append(s)
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break
    return scores

if TEST_DEFORMATIONS:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    scores_path = f"{log_dir}/deformed_metrics.csv"
    # df = pd.read_csv(scores_path, index_col=0)  
    scores = compute_deformation_scores()
    df = pd.DataFrame(scores)
    df.to_csv(scores_path)

# %%
if TEST_DEFORMATIONS:
    print(df.mean())

# %%
if TEST_DEFORMATIONS:
    print(df.std())

# %%
def compute_scores_deformed_diff():
    scores = []
    with tqdm(total=N_TESTING_FRAMES) as pbar:
        while True:
            try:
                clear_output(wait=True)
                display(pbar.container)
                df = pd.DataFrame(scores)
                df.to_csv(f"{log_dir}/deformed_diff_metrics.csv")
                print(df.mean())
                print(df.std())
                current_diffs, deformed_diffs = test_generator.get_deformed_diff_batch()
                s = evaluate_metrics(current_diffs, deformed_diffs, batch_size=BATCH_SIZE)
                scores.append(s)
                pbar.update(BATCH_SIZE)
            except StopIteration:
                break
    return scores

if TEST_DEFORMATIONS:
    test_generator = PreprocessedDataGenerator(test_paths, generator, discriminator, generator_loss, discriminator_loss, batch_size=BATCH_SIZE, repeat=False, blend_deformation=BLEND_DEFORMATIONS)
    scores_path = f"{log_dir}/deformed_diff_metrics.csv"
    # df = pd.read_csv(scores_path, index_col=0)  
    scores = compute_scores_deformed_diff()
    df = pd.DataFrame(scores)
    df.to_csv(scores_path)

# %%
if TEST_DEFORMATIONS:
    print(df.mean())

# %%
if TEST_DEFORMATIONS:
    print(df.std())


