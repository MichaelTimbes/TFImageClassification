#!/usr/bin/env python3

"""
Sorts images from a source path into training and test folders.

Original written by Michael Timbes.
Contributions by James Church.

To see all of the options for this script, use this line:

python3 form_pics.py -h

Last modified on June 15, 2017.
"""

import random
import argparse
from os import listdir
from os import path as opath
from PIL import Image as PImage
from PIL import ImageOps

def prepareImage(fullImagePath, width, height, colormode):
    """
    Prepares an image. The image is opened, modified in memory, and returned.
    Precondition: width and height are positive
    Postcondition: none.
    Return: the modified image

    Parameters:
    fullImagePath: the full path of the image
    width: desired width of the image
    height: desired height of the image
    colormode: color mode (according to PIL documentation)
    """

    assert width > 0
    assert height > 0

    size = (width, height)
    modifiedImage = PImage.open(fullImagePath)

    modifiedImage = modifiedImage.convert(colormode)
    modifiedImage = ImageOps.fit(modifiedImage, size, PImage.ANTIALIAS,0,(0.5,0.5))
    return modifiedImage

def randomizeImages(sourcePath):
    """
    Reads in all files on a path, takes the ones ending with jpg, bmp, or png,
    then adds all of the non-hidden files to a list and randomizes the list.

    Precondition: none
    Postcondition: none
    Return: the randomized list of images

    Parameters:
    sourcePath: the path containing original images
    """
    images = []
    for image in listdir(sourcePath):
        if not image.startswith(".") and (image.endswith("jpg") or image.endswith("bmp") or image.endswith("png")):
            images.append(image) 

    random.shuffle(images)
    return images

def divideImages(images, probability):
    """
    Takes a list of images and divides them into two groups: training and test.
    The number put into training is the first int(probability * len(images)) images.
    The rest go into the test set.

    Precondition: probability > 0 and probability < 1
    Postcondition: none
    Returns: two lists (trainingImages, testImages)

    Parameters:
    images: a list of image filenames
    probability: The percentage of desired training images
    """

    assert probability > 0 and probability < 1

    trainingImages = []
    testImages = []
    trainingImageCount = int(probability * len(images))
    i = 0
    for image in images:
        if i <= trainingImageCount:
            trainingImages.append(image)
        else:
            testImages.append(image)
        i += 1
    return (trainingImages, testImages)

def prepareAndStoreImages(images, sourcePath, savePath, width, height, colormode):
    """
    Prepares and stores image into a folder.

    Precondition: width and height are positive
    Postcondition: the images are modified and saved in the savePath
    Returns nothing.

    Parameters:
    images: a list of image filenames
    sourcePath: the source path of the original images
    savePath: the desired location of the modified images
    width: desired width of the modified images
    height: desired height of the modified images
    colormode: color mode of the modified images (according to PIL)
    """

    assert width > 0
    assert height > 0

    for image in images:
        modifiedImage = prepareImage(sourcePath + '/' + image, width, height, colormode)
        modifiedFileName = opath.splitext(image)[0] + ".bmp"
        modifiedImage.save(savePath + "/" + modifiedFileName)
        print(image, "added to", savePath)

def sortImages(sourcePath, trainingPath, testPath, width, height, probability, colormode):
    """
    Sorts images from a source path into test and training paths.
    The number of files put into training path is determined by the probability.

    Precondition: width and height are positive, probability must be between 0 and 1.
    Postcondition: the images are modified and saved in the test and training paths
    Returns nothing.

    Parameters:
    sourcePath: the source path of the original images
    trainingPath: the desired location of the modified training images
    testPath: the desired location of the modified testing images
    width: desired width of the modified images
    height: desired height of the modified images
    probability: The percentage of desired training images
    colormode: color mode of the modified images (according to PIL)
    """

    assert width > 0
    assert height > 0
    assert probability > 0 and probability < 1

    images = randomizeImages(sourcePath)
    (trainingImages, testImages) = divideImages(images, probability)
    prepareAndStoreImages(trainingImages, sourcePath, trainingPath, width, height, colormode)
    prepareAndStoreImages(testImages, sourcePath, testPath, width, height, colormode)

if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Prepares images for testing and training data set.")

    p.add_argument("-s", "--source", metavar="images", dest="source", default="images",
                   help="Sets the source directory of images.")
    p.add_argument("-tr", "--training", metavar="train_set", dest="training", default="train_set",
                   help="Sets the training directory of images.")
    p.add_argument("-te", "--testing", metavar="test_set", dest="testing", default="test_set",
                   help="Sets the training directory of images.")
    p.add_argument("-he", "--height", metavar="100", dest="height", default="100",
                   help="Sets the height of each image.")
    p.add_argument("-wi", "--width", metavar="100", dest="width", default="100",
                   help="Sets the width of each image.")
    p.add_argument("-p", "--trainprob", metavar="0.75", dest="probability", default="0.75",
                   help="Probability that an image will be put into the training set.")
    p.add_argument("-m", "--mode", metavar="L", dest="colormode", default="L",
                   help="Sets the color mode of each image. 1=BW, L=Gray, RGB=Color, RGBA=Color with transparency")
    options = p.parse_args()

    sortImages(options.source,
               options.training,
               options.testing,
               int(options.width),
               int(options.height),
               float(options.probability),
               options.colormode)
