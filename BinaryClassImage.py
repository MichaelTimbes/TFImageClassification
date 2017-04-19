
# coding: utf-8

# In[32]:

#Michael Timbes
#Purpose:
#Image_Classification Based on Logistic classification model. Single hidden layer, does not use convolutional layers
#does not use pooling.


# In[33]:

import tensorflow as tf
import numpy as np
#Visualization
import matplotlib.pyplot as plt
#File import
from os import listdir
from os import path as opath
from PIL import Image as PImage


# In[34]:

#Path of train images
train_path = ('train_set')
#Path of test images
test_path = ('test_set')
#Dimension of image
IMAG_X = 10

#Trainig rate alpha
alpha = 0.5

#Number of inputs defined
NUM_IN = (IMAG_X**2)

#Number of classifications
y_out_clss = 2
keyA = 'face' #True Class
keyB = 'notface' #False Class

#Size of batch
BatchSize = 2


# In[35]:

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
    3. Converts file Numpy array.
    4. Reads labels and converts to binary class.
    5. Returns Numpy array objects for images and labels.
    """

    imagesList = listdir(path)
    
    loadedImages = []
    
    loadedLabels = []
  
    new_size = IMAG_X, IMAG_X
    
    for image in imagesList:
     
        img = PImage.open(path +'/'+ image)
        # Pull file name from current image- use it as a label
        loadedLabels.append(opath.splitext(image)[0])
        img.load()
        # Resize step- ensures that all images follow.
        img.thumbnail(new_size, PImage.ANTIALIAS )
        loadedImages = np.asarray( img, dtype="int32" )
        
    # Convert to Binary Classification.
    for i in range(0,len(loadedLabels)):
        if (keyA in loadedLabels[i] and not(keyB in loadedLabels[i]) ):
            loadedLabels[i] = [1, 0]
        else:
            loadedLabels[i] = [0, 1]
    
    return loadedImages, np.asarray(loadedLabels)


# In[36]:

def shape_up_(train_X, IMAG_X):
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
    new_X = train_X.reshape((num_exs , IMAG_X**2))
    new_X.flatten('C')
    return new_X


# In[37]:

def create_batch(train_X, train_Y, start, batch_size):
    """
    def create_batch(train_X, batch_size):
    __________________________________________
    Function Outline:
    1.Creates training subset of train_X and train_Y
    """
    # Everything in the row
    batch_x = train_X[:,start:batch_size]
    batch_y = train_Y[start:batch_size]
    return batch_x, batch_y


# # Data Preparation
# * Import Images
# * Reshape Images
# * Final Array Should Be:
# $$train_{x} = \begin{pmatrix} 
# [Picture Array_{MxM}] , & [Num Examples]\\
# \end{pmatrix}$$
# $$train_{y} = \begin{pmatrix} 
# Class_{1} & Class_{2} & \cdots & Class_{n}\\
# \end{pmatrix}$$
# Where the $Class_{n}$ is either a 1 for the true class or 0 for the false class.

# In[38]:

#Training Data Preparation
train_X, train_Y = ImportImages(train_path)
m = len(train_X[0,0,:])
train_X = shape_up_(train_X,IMAG_X)
# Showing the shape of train_X
train_X[0,:]


# In[39]:

#Testing Data Preparation 
#test_x, test_y = ImportImages(test_path)
#tstm = len(test_X[0,0,:])


# # Build Logistic Model
# ## Input Layer
# Dimension for $X$ is $1xN$. For Tensorflow, x_input_layer as a placeholder must be at least a 1-D vector but can support $MxN$ so a 'None' type is used to be more dynamic. To ensure the matrix multiplication is not an issue be sure that the weight layer and X layer are $Nx1$ and $1xN$. 
# \begin{equation}
#     \begin{pmatrix}
#     x_{0} \\
#     x_{1} \\
#     \vdots \\
#     x_{n}
#     \end{pmatrix}
# \end{equation}
# ## Weights
# \begin{equation}
#     \begin{pmatrix}
#     \theta_{0} & \theta_{1} & \cdots & \theta{n} \\
#      & & \vdots & \\
#      \theta_{0} & \theta_{1} & \cdots & \theta{n} \\
#     \end{pmatrix}
# \end{equation}
# 
# ## Output Layer
# Below is for multi-class in this application where there are $\theta_{n}$ classes.
# \begin{equation}
# h_{\theta}(x) = 
# \begin{pmatrix}
#     \theta_{0} & \theta_{1} & \cdots & \theta{n} \\
#      & & \vdots & \\
#      \theta_{0} & \theta_{1} & \cdots & \theta{n} \\
#     \end{pmatrix}
#     \begin{pmatrix}
#     x_{0} \\
#     x_{1} \\
#     \vdots \\
#     x_{n}
#     \end{pmatrix} =
#      \begin{pmatrix}
#     y_{0} \\
#     y_{1} \\
#     \vdots \\
#     y_{m}
#     \end{pmatrix}
# \end{equation}

# In[40]:

#The x_input_layer and y_output_layer values are placeholders for the model that accept the flattened image (x) 
#and then the ouput of theclassifications (y). 
#
x_input_layer = tf.placeholder(tf.float32, shape=[None, NUM_IN]) 
y_output_layer = tf.placeholder(tf.float32, shape= [None, y_out_clss])


#The vectors for weights and b- the bias will be defined as variables for training later.

Weights = tf.Variable(tf.zeros([NUM_IN, y_out_clss]))
b = tf.Variable(tf.zeros([y_out_clss]))


# In[41]:

#Outline of the model based on the probabilities calculated plus bias values.

y = tf.matmul(x_input_layer, Weights) + b


# # Cross Entropy
# ## Cost Function
# \begin{equation}
# J(\theta)= -\frac{1}{m}\sum_{i=1}^{m} y_{i}log(h_{\theta}(x_{i}))+
# 									  (1-y_{i})log(1-h_{\theta}(x_{i}))
# \end{equation}
# ## Minimize Cost-Gradient Descent
# \begin{equation}
# \theta_{j} = \theta_{j}-\alpha\frac{1}{m}\sum_{i=1}^{m}\left((h_{\theta}(x_{i})-y_{i})X_{ji}\right)
# \end{equation}

# In[42]:

#This is where the train steps happens. Cross entropy is calculated by running the current model and then running 
#gradient decent. Training step stores results from the gradient descent minimizing cost function (cross_entropy).


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_output_layer, logits = y))

train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)


# In[43]:

#Init session and global variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#Design batch variables- ensure dimension
x_batch = np.ones((1, NUM_IN))
y_batch = np.ones((1, 2))


# In[44]:

#Training Block:
for i in range(400): #Outside train loop
     for j in range(0,m):
            x_batch[0,:] = train_X[j,:]
            #print("EXAMPLE",j)
            #print(x_batch)
            y_batch[0,:] = train_Y[j]
            train_step.run(feed_dict={x_input_layer: x_batch, y_output_layer: y_batch})
print("DONE.")


# In[45]:

#Test Block:

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_output_layer,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for j in range(0,m):
    x_batch[0,:] = train_X[j,:]
    y_batch[0,:] = train_Y[j]
    print(accuracy.eval(feed_dict={x_input_layer: x_batch, y_output_layer: y_batch}))

