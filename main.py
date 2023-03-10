from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy import load
from numpy.random import randn
from numpy.random import randint
from numpy import concatenate
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
import matplotlib.animation as animation
import skvideo.io
import numpy as np
from pathlib import Path
from numpy import load
import skvideo.io
from skimage.io import imsave
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.math import divide_no_nan, reduce_std, reduce_mean, square
from tensorflow import concat

def concat_last_sample(y, previous_y):
  last_sample = previous_y[-1]
  last_sample_reshaped = last_sample.reshape(1,last_sample.shape[0],last_sample.shape[1],1)
  return concat([last_sample_reshaped, y[0:-1]], axis=0)

class RatioLoss(object):
  def __init__(self):
    super().__init__()
    self.previous_y_pred = None
    self.previous_y_true = None

  def __call__(self, y_true, y_pred, sample_weight=None):
    if self.previous_y_pred is None:
      self.previous_y_pred = y_pred
      self.previous_y_true = y_true
    
    print('y_true')
    print(y_true.shape)
    print('y_pred')
    print(y_pred.shape)
    print('concat_last_sample')
    print(concat_last_sample(y_pred, self.previous_y_pred).shape)
    quit()
    # ratio
    ratio_pred = divide_no_nan(
        y_pred, concat_last_sample(y_pred, self.previous_y_pred)
    )
    ratio_true = divide_no_nan(
        y_true, concat_last_sample(y_true, self.previous_y_true)
    )

    #std
    std_pred = reduce_std(ratio_pred, axis=-1)
    std_true = reduce_std(ratio_true, axis=-1)

    #diff
    diff_std = std_true - std_pred

    # update previous values
    self.previous_y_pred = y_pred
    self.previous_y_true = y_true
    return diff_std

# define the standalone discriminator model
def define_discriminator(in_shape=(64,64,1)):
 model = Sequential()
 model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
 model.add(LeakyReLU(alpha=0.2))
 model.add(Dropout(0.4))
 model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
 model.add(LeakyReLU(alpha=0.2))
 model.add(Dropout(0.4))
 model.add(Flatten())
 model.add(Dense(1, activation='sigmoid'))
 # compile model
 opt = Adam(lr=0.0002, beta_1=0.5)
 model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
 return model
 
# define the standalone generator model
def define_generator(latent_dim):
 model = Sequential()
 # foundation for 16x16 image
 n_nodes = 128 * 16 * 16
 model.add(Dense(n_nodes, input_dim=latent_dim))
 model.add(LeakyReLU(alpha=0.2))
 model.add(Reshape((16, 16, 128)))
 # upsample to 32x32
 model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 model.add(LeakyReLU(alpha=0.2))
 # upsample to 64x64
 model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
 model.add(LeakyReLU(alpha=0.2))
 model.add(Conv2D(1, (16,16), activation='sigmoid', padding='same'))
 return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, loss=None):
  # make weights in the discriminator not trainable
  d_model.trainable = False
  # connect them
  model = Sequential()
  # add generator
  model.add(g_model)
  # add the discriminator
  model.add(d_model)
  # compile model
  opt = Adam(lr=0.0002, beta_1=0.5)
  if loss is None:
    total_loss='binary_crossentropy'
  else:
    total_loss = loss
  model.compile(loss=total_loss, optimizer=opt)
  return model
 
# load and prepare mnist training images
def load_real_samples():
  # load mnist dataset
  # (trainX, _), (_, _) = load_data()
  trainX = load('/home/jupyter/data/mnist_test_seq.npy')
  #Select a subset to test the algorithm
  # trainX = trainX[:,:,:,:]
  # expand to 3d, e.g. add channels dimension
  X = expand_dims(trainX, axis=-1)
  # convert from unsigned ints to floats
  X = X.astype('float32')
  # scale from [0,255] to [0,1]
  X = X / 255.0
  return X
 
# select real samples
def generate_real_samples(dataset, n_samples):
  # choose random instances
  ix = randint(0, dataset.shape[1], n_samples)
  # retrieve selected images
  X = dataset[:,ix,:,:,:]
  # generate 'real' class labels (1)
  y = ones((20, n_samples, 1))
  return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
 # generate points in the latent space
 x_input = randn(latent_dim * n_samples)
 # reshape into a batch of inputs for the network
 x_input = x_input.reshape(n_samples, latent_dim)
 return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
 # generate points in latent space
 x_input = generate_latent_points(latent_dim, n_samples*20)
 # predict outputs
 X = g_model.predict(x_input)
 # create 'fake' class labels (0)
 y = zeros((20, n_samples, 1))
 X = X.reshape(20,n_samples,X.shape[1],X.shape[2],X.shape[3])
 return X, y
 
# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=3, save_path=None, name='gan'):
  if save_path is None:
    save_path = Path.cwd()
  #each seqeunce
  for i in range(20):
    video_path = save_path / 'moving_plot_e{0:03}_v{0:02}.mp4'.format((epoch+1), i)
    writer = skvideo.io.FFmpegWriter(str(video_path))
    for j in range(n * n):
      # define subplot
      # pyplot.subplot(n, n, 1 + j)
      # turn off axis
      # pyplot.axis('off')
      # plot raw pixel data
      # im = pyplot.imshow(examples[i, j, :, :, 0], cmap='gray_r')
      # to rgb
      im = examples[i, j, :, :, 0]
      w, h = im.shape
      ret = np.empty((w, h, 3), dtype=np.uint8)
      ret[:, :, :] = im[:, :, np.newaxis]
      # save plot to frame
      writer.writeFrame(ret)
    # pyplot.close()
    writer.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=10, name='gan'):
  output_path = Path.mkdir(parents=True, exist_ok=True, path='output/{}'.format(name))
  # prepare real samples
  X_real, y_real = generate_real_samples(dataset, n_samples)
  # reshape sequences to all frames
  X_real = X_real.reshape(X_real.shape[0]*X_real.shape[1],X_real.shape[2],X_real.shape[3],X_real.shape[4])
  y_real = y_real.reshape(y_real.shape[0]*y_real.shape[1],y_real.shape[2])
  # evaluate discriminator on real examples
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
  # prepare fake examples
  X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
  # reshape sequences to all frames
  new_X_fake = X_fake.reshape(X_fake.shape[0]*X_fake.shape[1],X_fake.shape[2],X_fake.shape[3],X_fake.shape[4])
  y_fake = y_fake.reshape(y_fake.shape[0]*y_fake.shape[1],y_fake.shape[2])
  # evaluate discriminator on fake examples
  _, acc_fake = d_model.evaluate(new_X_fake, y_fake, verbose=0)
  # summarize discriminator performance
  print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
  # save plot
  save_plot(X_fake, epoch, name)
  # save the generator model tile file
  model_path = output_path / 'moving_model_{0:03}.h5'.format(epoch + 1)
  g_model.save(str(model_path))
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256, name='gan'):
 bat_per_epo = int(dataset.shape[1] / n_batch)
 half_batch = int(n_batch / 2)
 # manually enumerate epochs
 for i in range(n_epochs):
  # enumerate batches over the training set
  for j in range(bat_per_epo):
    # get randomly selected 'real' samples
    X_real, y_real = generate_real_samples(dataset, half_batch)
    # generate 'fake' examples
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    # create training set for the discriminator
    # X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
    X = concatenate((X_real, X_real), axis=1)
    y = concatenate((y_real, y_fake), axis=1)
    # sequences to all frames
    X = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4])
    y = y.reshape(y.shape[0]*y.shape[1],y.shape[2])
    # update discriminator model weights
    d_loss, _ = d_model.train_on_batch(X, y)
    # prepare points in latent space as input for the generator
    X_gan = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    y_gan = ones((n_batch, 1))
    # update the generator via the discriminator's error
    g_loss = gan_model.train_on_batch(X_gan, y_gan)
    # summarize loss on this batch
    print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
  # evaluate the model performance each epoch
  summarize_performance(i, g_model, d_model, dataset, latent_dim, name)


def vanilla_gan():
  # size of the latent space
  latent_dim = 100
  # create the discriminator
  d_model = define_discriminator()
  # create the generator
  g_model = define_generator(latent_dim)
  # create the gan
  gan_model = define_gan(g_model, d_model)
  # load image data
  dataset = load_real_samples()
  # train model
  train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100, name='vanilla_gan')

def ratio_gan():
  # size of the latent space
  latent_dim = 100
  # create the discriminator
  d_model = define_discriminator()
  # create the generator
  g_model = define_generator(latent_dim)
  # create the gan
  ratio_loss = RatioLoss()
  gan_model = define_gan(g_model, d_model, loss=ratio_loss)
  # load image data
  dataset = load_real_samples()
  # train model
  train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=100, name='vanilla_gan')

def test_loss():
  dataset = load_real_samples()
  # 20, 10000, 64, 64, 1
  first = dataset[0, 0:4, :, :, 0].reshape(4, 64, 64, 1)
  second = dataset[0, 4:8, :, :, 0].reshape(4, 64, 64, 1)
  third = dataset[0, 8:12, :, :, 0].reshape(4, 64, 64, 1)
  fourth = dataset[0, 12:16, :, :, 0].reshape(4, 64, 64, 1)
  ratio_loss = RatioLoss()
  print(first.shape)
  print(ratio_loss(first, second))
  print(ratio_loss(second, third))
  print(ratio_loss(third, fourth))
  print(ratio_loss(first, second))
  print(ratio_loss(fourth, fourth))
  print(ratio_loss(first, second))
  print(ratio_loss(fourth, fourth))
  print(ratio_loss(fourth, fourth))
  print(ratio_loss(fourth, fourth))
  print(ratio_loss(fourth, fourth).shape)


if __name__ == "__main__":
    # vanilla_gan()
    # test_loss()
    ratio_gan()
