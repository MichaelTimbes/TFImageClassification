
# coding: utf-8

# In[ ]:

#Michael Timbes
#Purpose:
#Image_Classification Based on Logistic classification model. Single hidden layer, does not use convolutional layers
#does not use pooling.


# In[ ]:

import tensorflow as tf
import numpy as np
#Visualization
import matplotlib.pyplot as plt
#File import
import filecmp as fcmp
from os import listdir
from os import path as opath
from PIL import Image as PImage


# In[ ]:

#Path of test images
path = ('train_set')
#Dimension of image
IMAG_X = 10

#Trainig rate alpha
alpha = 0.5

#Number of inputs defined
NUM_IN = (IMAG_X**2)

#Number of classifications
y_out_clss = 2

#Size of batch
BatchSize = 2


# In[ ]:

#Import data
def ImportImages(path):
    """
    def ImportImages(path):
    USES VARIABLES: IMAG_X, IMAG_Y, expects these to be set already.
    If prefered, edit code to pass them as arguments.
    ________________________________________________________________
    Function Outline:
    1. Loads list of images from the path variable (type string).
    2. Iterates through directory loads image into I/O file stream.
    3. Converts file to LA grey scale and resizes.
    4. Casts the image data into Numpy array.
    5. Returns Numpy array object.
    """

    imagesList = listdir(path)
    
    loadedImages = []
    
    loadedLabels = []
  
    new_size = IMAG_X, IMAG_X
    
    for image in imagesList:
     
        img = PImage.open(path +'/'+ image)
        #Pull file name from current image- use it as a label
        loadedLabels.append(opath.splitext(image)[0])
        img.load()
        #Resize step- ensures that all images follow.
        img.thumbnail(new_size, PImage.ANTIALIAS )
        loadedImages = np.asarray( img, dtype="int32" )
    return loadedImages, loadedLabels


# In[ ]:

def shape_up_X(train_X, IMAG_X):
    """
    shape_up_X(train_X, IMAG_X):
    
    Expects a 3D numpy array train_X with 
    dimensions (width_val, height_val, image#).
    Must be square matrix with and height being
    equal.
    Returns new_X which is a reshaped train_X
    of type numpy array.
    ____________________________________
    Dimensions are (pixels, image#). The
    size of the pixels dimension is taken from
    the square of IMAG_X. The number of images are found
    by taking the length of the third column.
    """
    num_exs = len(train_X[0,0,:])
    new_X = train_X.reshape((IMAG_X**2, num_exs ))
    return new_X


# In[ ]:

def create_batch(train_X, train_Y, start, batch_size):
    """
    def create_batch(train_X, batch_size):
    __________________________________________
    Function Outline:
    1.Checks to see if batch is too large.
    2.Creates training subset of train_X and train_Y
    """
    
    if batch_size > len(train_X[0,:]):
        print("Batch size can not be greater than total training examples.")
        batch_x, batch_y = np.zeros(1,2)
    else:
        end_batch = start+batch_size
        for i in range (start,end_batch):
            batch_x = train_X[i]
            batch_y = train_Y[i]
    return batch_x, batch_y


# In[ ]:

train_X, train_Y = ImportImages(path)
print(train_X.shape)
print(train_Y)
#train_X = shape_up_X(train_X,IMAG_X)
#Num_to_Train = len(train_X)


# In[ ]:

#The x_input_layer and y_output_layer values are placeholders for the model that accept the flattened image (x) 
#and then the ouput of theclassifications (y). 
#
x_input_layer = tf.placeholder(tf.float32, shape= [None, NUM_IN])
y_output_layer = tf.placeholder(tf.float32, shape= [None, y_out_clss])


#The vectors for weights and b- the bias will be defined as variables for training later.

Weights = tf.Variable(tf.zeros([NUM_IN, y_out_clss]))
b = tf.Variable(tf.zeros([y_out_clss]))


# In[ ]:

#Outline of the model based on the probabilities calculated plus bias values.

y_model = tf.matmul(x_input_layer, Weights) + b


# In[ ]:

#This is where the train steps happens. Cross entropy is calculated by running the current model and then running 
#gradient decent. Training step stores results from the gradient descent minimizing cost function (cross_entropy).


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_output_layer, logits = y_model))

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)


# In[ ]:

#Init session and global variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# In[ ]:

#Training block
for i in range(10): #Outside train loop
    
    #p = np.random.permutation(range(Num_to_Train))
    #train_X, train_Y = train_X[p], train_Y[p]
    
    for start in range(0, Num_to_Train, BatchSize):
        end = start + BatchSize
        batch_x, batch_y = create_batch(train_X, train_Y, start, BatchSize)
        print(batch_x, batch_y)
    #train_step.run(feed_dict={x: batch_x, y_: batch_y}) 

