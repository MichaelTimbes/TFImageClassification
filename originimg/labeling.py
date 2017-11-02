from os import listdir
from os import path as opath
from PIL import Image as PImage
from PIL import ImageOps


sourcePath = "NotFace"
savepath = "temps"
images = []
nlabel = "notface_"

# Pull Images From Path
for image in listdir(sourcePath):
	images.append(image)
# Rename Images
i = 0 # Index Reference
for image in images:
	modifiedImage = PImage.open(sourcePath + '/' + image)
	modifiedFileName = nlabel + str(i) + ".jpeg"
	i+=1
	#print(modifiedFileName)
	modifiedImage.save(sourcePath + "/" + modifiedFileName)
