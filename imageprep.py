import tensorflow as tf
import numpy as np
from os import listdir
from os import path as opath
from skimage import io, color
from PIL import Image as PImage

def ImportImages(path, width, height, keyA, keyB, colortype='rgb'):
    """
    Images must be in RGB space.
    ________________________________________________________________
    Function Outline:
    1. Loads list of images from the path variable (type string).
    2. Iterates through directory loads image into I/O file stream.
    3. Converts file Numpy array.
    4. Reads labels and converts to binary class.
    5. Returns Numpy array objects for images and labels.
    """

    loadedImages = []
    loadedLabels = []
    originalLabels = []
    
    imagesList = listdir(path)
  
    new_size = width, height
    
    for image in imagesList:
        if not(".DS_Store" in path +'/'+ image): #Issue in Mac File System
            (img, label) = loadImage(path, image, colortype)
            loadedImages.append(np.asarray( img, dtype="int32" ))
            originalLabels.append(label)
        
    # Convert to Binary Classification.
    for originalLabel in originalLabels:
        if keyA in originalLabel and not(keyB in originalLabel):
            loadedLabels.append([1, 0])
        else:
            loadedLabels.append([0, 1])

    return np.asarray(loadedImages), np.asarray(loadedLabels), originalLabels

def loadImage(path, image, colortype='rgb'):
    img = PImage.open(path + '/' + image)
    label = opath.splitext(image)[:]

    # Pull file name from current image- use it as a label
    img.load()

    if colortype == 'lab':
        color.rgb2lab(img)

    # Resize step- ensures that all images follow.
    #img.thumbnail(new_size, PImage.ANTIALIAS )
    #img.convert('1')
    #img.resize(new_size)
    return (img, label[0])

def shape_up3d(data, width):
    """
    Expects a NUM_IN * NUM_IN sized picture.
    Changes the shape to be (N,NUM_IN**2).
    """
    num_exs = len(data[:,0,0,0])
    new_X = np.zeros((num_exs,width**2 * 3))
    for i in range(0,num_exs):
        new_X[i,:] = data[i,:,:,:].reshape((1,width**2 * 3))
    return new_X

def shape_up2d(data, width):
    """
    Expects a NUM_IN * NUM_IN sized picture.
    Changes the shape to be (N,NUM_IN**2).
    """
    num_exs = len(data[:,0,0])
    new_X = np.zeros((num_exs,width**2))
    for i in range(0,num_exs):
        new_X[i,:] = data[i,:,:].reshape((1,width**2))
    return new_X

def shape_up_X(data, size):
    """
    shape_up_X(train_X, IMAG_X):
    Expects a NUM_IN * NUM_IN sized picture. Changes
    the shape to be (N,NUM_IN**2).
    ____________________________________
    """
    #num_exs = len(data[:,0,0])
    #for i in range(num_exs):
      #  temp.append(tf.reshape(data[i,:,:], [-1,size,size,1]))
    return np.reshape(data, [-1,size,size,1])

def out_class(Y, keyA, keyB):
    """
    Matches Class with input vector.
    """
    return np.asarray([keyB if label == 1 else keyA for label in Y])

def weight_variable(shape):
    """
    Helper function to help initialize weight variables.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Helper function to initialize bias variable.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    Helper function that returns a 2d convolutional layer
    """
    x = x.astype(np.float32, copy=False)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


