
# coding: utf-8

# In[ ]:

#Michael Timbes
#Purpose:
#Image_Classification Based on Logistic classification model. Single hidden layer, does not use convolutional layers
#does not use pooling. May not be accurate for use when non-binary classification is needed.


# In[ ]:

import tensorflow as tf
import numpy as np
import PIL
#Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#File import
from os import listdir
from PIL import Image as PImage


# In[ ]:

#Path of test images
path = ('testimgs')
#Dimension of image
IMAG_X = 10
IMAG_Y = 12

#Trainig rate alpha
alpha = 0.5

#Number of inputs defined
NUM_IN = (IMAG_X * IMAG_Y)

#Number of classifications
y_out_clss = 2


# In[ ]:

#Import data
def ImportImages(path):
    """
    def ImportImages(path):
    ________________________________________________________________
    Function Outline:
    1. Loads list of images from the path variable (type string).
    2. Iterates through directory loads image into I/O file stream.
    3. Converts file to LA grey scale.
    4. Casts the image data into Numpy array.
    5. Returns Numpy array object.
    """

    imagesList = listdir(path)
    
    loadedImages = []
    
    for image in imagesList:
        
        img = PImage.open(path +'/'+ image).convert('LA')
        img.load()
        loadedImages = np.asarray( img, dtype="int32" )
    #Add reshape code here:
    return loadedImages


# In[ ]:

train_X = ImportImages(path)


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
#for i in range(100): #Outside train loop
   # for start in range(0, len())

