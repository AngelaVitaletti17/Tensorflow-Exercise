import os
import tensorflow as tf
import skimage
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import transform
import random

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

#We're going to resize these images first
n_images = [transform.resize(image, (28, 28)) for image in images]
#Now convert them to gray
n_images = np.array(n_images)
n_images = rgb2gray(n_images)

#Let's set up some placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28]) #unflattened tensor
y = tf.placeholder(dtype = tf.int32, shape = [None]) #Assuming this will be used for each classification, of which a sign can only have one

f_imgs = tf.contrib.layers.flatten(x) #This flattens actual data (non-placeholder)

logits = tf.contrib.layers.fully_connected(f_imgs, 62, tf.nn.relu) #This will connect the images, assumably, into a layer based on features.

f_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)) #From my understanding, a loss function pretty much tells you if your algorithm is working properly, in terms of accurate predictions/assumptions
#This specific function from tensorflow says that it computers the error when classifying things, according to the documentation, assuming there can only be one classification

train_op = tf.train.AdamOptimizer(learning_rate=0.002).minimize(f_loss) #The Adam Optimizer implements an algorithm that (again from my understanding) is used to keep track of learning rates for multiple network layers
#This line of code should allow for an initialization before training of a model

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #This should compute our accuracy as a floating point value

print("FLAT: ", f_imgs)
print("LOGITS: ", logits) #Logits are apparently predicitions that the model creates
print("LOSS: ", f_loss)
print("PREDICTION: ", correct_pred)

tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(201): #Using 201 to be able to get to 200
	print('EPOCH', i)
	_, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: n_images, y: labels}) #going to run the session, let's train the model
	if i % 10 == 0: #stop every 10 and print the loss
		print("Loss: ", f_loss)
	print('DONE')
	
#We're going to pick some images
sample = random.sample(range(len(n_images)), 40)
sample_imgs = [n_images[i] for i in sample]
sample_lbls = [labels[i] for i in sample]

pred = sess.run([correct_pred], feed_dict={x: sample_imgs})[0] #Run on the sample images we've just picked
print(sample_lbls)
print(pred)

#Actually show the stuff
fig = plt.figure(figsize=(10,10))
for i in range (len(sample_imgs)):
	truth = sample_lbls[i]
	predi = pred[i]
	plt.subplot(5,8,i+1)
	plt.axis('off')
	color='green' if truth == predi else 'red'
	plt.text(40, 10, "Truth:		{0}\nPrediction: {1}".format(truth, predi), fontsize=12, color=color)
	plt.imshow(sample_imgs[i], cmap="gray")
plt.show()

#Let's compare to test
test_images, test_labels = get_that_data(test_PATH)
n_test_images = [transform.resize(image, (28, 28)) for image in test_images]
n_test_images = rgb2gray(np.array(n_test_images))

pred = sess.run([correct_pred], feed_dict={x:n_test_images})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, pred)])

accuracy = match_count / len(test_labels)

print("Test Results: {:.3f}".format(accuracy)) #print accuracy to 3 decimal points

sess.close()
