#Script to automatically format pictures to gray scale and to .bmp format
from os import listdir
from os import path as opath
from PIL import Image as PImage
from PIL import ImageOps
#Set Path to Directory
path = 'images'
new_size = 100,100
dir_images = listdir(path)

for image in dir_images:
    if not(".DS_Store" in path +'/'+ image): #Issue in Mac File System
    #OPEN FILE
        temp_file = PImage.open(path + '/' + image)
    #GRAB FILE NAME
        f_name = opath.splitext(image)[0]
    #CONVERT TO GRAY SCALE
        temp_file = temp_file.convert('1')
        #temp_file.resize(new_size)
        temp_file = ImageOps.fit(temp_file,(new_size),PImage.ANTIALIAS,0,(0.5,0.5))
        f_name = f_name +".bmp"
        temp_file.save(f_name)
