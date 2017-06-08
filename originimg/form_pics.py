#Script to automatically format pictures to gray scale and to .bmp format
import os
from os import listdir
from os import path as opath
from PIL import Image as PImage
from PIL import ImageOps

#Set Path to Directory Source
path = "faces"

#Set Path to Directory Output
outpath = "Face/"

#Set Resize Params
new_size = 100,100

#Define directory
dir_images = listdir(path)

#Set Classification Label
class_name = "face"

#Create Index Variable
img_num = 0

#Create Output Directory 
if not opath.exists(outpath):
    os.mkdir(outpath)

for image in dir_images:
    if not(".DS_Store" in path +'/'+ image): #Issue in Mac File System
    #OPEN FILE
        temp_file = PImage.open(path + '/' + image)
    #GRAB FILE NAME
        #f_name = opath.splitext(image)[0]
        f_name = class_name + "_" + str(img_num)
    #CONVERT TO GRAY SCALE
       # temp_file = temp_file.convert('L')
    #RESIZE
        temp_file = ImageOps.fit(temp_file,(new_size),PImage.ANTIALIAS,0,(0.5,0.5))
    #RENAME and SAVE
        f_name = f_name + ".jpeg"
        temp_file.save(outpath + f_name,"JPEG")
    #INC IMG INDX
        img_num = img_num + 1
