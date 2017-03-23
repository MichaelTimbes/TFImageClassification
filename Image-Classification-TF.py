
# coding: utf-8

# In[48]:

#Michael Timbes
#Purpose:
#Image_Classification Based on Logistic classification model. Single hidden layer, does not use convolutional layers
#does not use pooling. May not be accurate for use when non-binary classification is needed.


# In[49]:

import tensorflow as tf
import numpy as np
#Visualization
import matplotlib.pyplot as plt
#File import
from os import listdir
from os import path as opath
from PIL import Image as PImage


# In[50]:

#Path of test images
path = ('testimgs')
#Dimension of image
IMAG_X = 20

#Trainig rate alpha
alpha = 0.5

#Number of inputs defined
NUM_IN = (IMAG_X)

#Number of classifications
y_out_clss = 2


# In[51]:

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
        
        img = PImage.open(path +'/'+ image).convert('LA')
        #Pull file name from current image- use it as a label
        loadedLabels.append(opath.splitext(image)[0])
        img.load()
        #Resize step- ensures that all images follow.
        img.thumbnail(new_size, PImage.ANTIALIAS )
        loadedImages = np.asarray( img, dtype="int32" )
        
    return loadedImages, loadedLabels


# In[52]:

def create_batch(train_X, batch_size):
    """
    def create_batch(train_X, batch_size):
    __________________________________________
    Function Outline:
    1.
    """
    return 1


# In[53]:

train_X, train_Y = ImportImages(path)
print(train_Y)


# In[54]:

#The x_input_layer and y_output_layer values are placeholders for the model that accept the flattened image (x) 
#and then the ouput of theclassifications (y). 
#
x_input_layer = tf.placeholder(tf.float32, shape= [None, NUM_IN])
y_output_layer = tf.placeholder(tf.float32, shape= [None, y_out_clss])


#The vectors for weights and b- the bias will be defined as variables for training later.

Weights = tf.Variable(tf.zeros([NUM_IN, y_out_clss]))
b = tf.Variable(tf.zeros([y_out_clss]))


# In[55]:

#Outline of the model based on the probabilities calculated plus bias values.

y_model = tf.matmul(x_input_layer, Weights) + b


# In[56]:

#This is where the train steps happens. Cross entropy is calculated by running the current model and then running 
#gradient decent. Training step stores results from the gradient descent minimizing cost function (cross_entropy).


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_output_layer, logits = y_model))

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)


# In[57]:

#Init session and global variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# In[58]:

#Training block
#for i in range(10): #Outside train loop
 #   train_step.run(feed_dict={x: batch[0], y_: batch[1]}) 

