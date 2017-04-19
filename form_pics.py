#Script to automatically format pictures to gray scale and to .bmp format
from os import listdir
from os import path as opath
from PIL import Image as PImage
#Set Path to Directory
path = 'test_set'

dir_images = listdir(path)

for image in dir_images:
    #OPEN FILE
    temp_file = PImage.open(path + '/' + image)
    #GRAB FILE NAME
    f_name = opath.splitext(image)[0]
    #CONVERT TO GRAY SCALE
    temp_file = temp_file.convert('1')
    f_name = f_name +".bmp"
    temp_file.save(f_name)
