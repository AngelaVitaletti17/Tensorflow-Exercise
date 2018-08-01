import os
import tensorflow as tf
import skimage
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import transform

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

'''
print(np.array(images).ndim)
print(np.array(images).size)
print(images[0]) #Note: This prints out a single image as a set of multidimensional arrays? This is cool!

print(np.array(labels).ndim)
print(np.array(labels).size)
print (len(set(np.array(labels)))) #So this actually prints out correctly an identification of 62 signs specified in the tutorial... which makes sense since there are 62 directories
'''

'''
#Now let's see some of that data
ran_idxs = [102, 234, 1234, 908]
for i in range(len(ran_idxs)): #This actually shows the images and adjust some plot settings (format)
	plt.subplot(2, 2, i + 1) #This adjusts the format, columns/rows of output
	plt.axis('off') #This shows the axis' for each image
	plt.imshow(images[ran_idxs[i]]) #Shows the image
	plt.subplots_adjust(wspace=0.5) #Adjusts the whitespace
	print("Shape: {0}, Min Signs: {1}, Max Signs {2}".format(images[ran_idxs[i]].shape,
															 images[ran_idxs[i]].min(),
															 images[ran_idxs[i]].max()))
	
plt.show() #Make the screen come up to show all images
'''

'''
#Further, let's see the signs by labels
cate = set(labels)
plt.figure(figsize=(15,15))
i = 0 #I'm a C#, C++ programmer, it doesn't feel right starting a counter from 1
for label in cate:
	im = images[labels.index(label)]
	plt.subplot(8,8,i+1)
	plt.axis('off')
	plt.title("Label {0} ({1})".format(label, labels.count(label)))
	i += 1;
	plt.imshow(im)
plt.show()

#So all of ^^ shows the classification of each sign by it's unique type! This keeps getting better.
'''

#Now let's do some work with tensor flow!

