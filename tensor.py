import os
import tensorflow as tf
import skimage
from skimage import data

def get_that_data(dir):
	dirs = [thing for thing in os.listdir(dir) if os.path.isdir(os.path.join(dir, thing))]
	labels = []
	images = []
	for thing in dirs:
		l_dir = os.path.join(dir, thing) #This will give the bath of the directory / the thing
		
		#Gets the file names in the current directory of all of the files in that directory as long as it is an image
		file_names = [os.path.join(l_dir, file) 
						for file in os.listdir(l_dir) if file.endswith(".ppm")]
						
		#Add each loaded image for each file (image) that we just found into images
		for file in file_names: 
			images.append(skimage.data.imread(file))
			labels.append(int(thing))
			
	return images, labels

PATH = os.getcwd()
train_PATH = os.path.join(PATH, "images\Training")
test_PATH = os.path.join(PATH, "images\Testing")

images, labels = get_that_data(train_PATH)
