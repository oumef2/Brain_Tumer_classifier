#import os
import PIL
#import zipfile
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py
import cv2
#from google.colab.patches import cv2_imshow
#%matplotlib inline



img = None

labels = []
borders = []
images = []

#Save images of brain tumor, masks and store labels and borders in their respective lists iteratively.
filename = None
for filename in range(1, 3065):
	with h5py.File('dataset/{}.mat'.format(filename), 'r') as f:
		img = f['cjdata']['image']
		img = np.array(img, dtype=np.float32)

		label = f['cjdata']['label'][0][0]
		border = f['cjdata']['tumorBorder'][0]
		border = np.array(border, dtype=np.float64)

		images.append(img)
		labels.append(int(label))
		borders.append(border)
		plt.axis('off')
		plt.imsave("new_dataset/bt_images/{}.jpg".format(filename), img, cmap='gray')
	  
	  
print("{} files successfully saved".format(filename))

#Convert the Python lists to a Numpy arrays
label_names = np.array(labels, dtype=np.int64)

#Check if the array has the right shape & length.
label_names.shape

#Store the labels and tumor border (coordinates) as a pickle file, which can be loaded whenever we want to use it.
pickle_out = open("new_dataset/labels.pickle","wb")                    
pickle.dump(label_names, pickle_out)
pickle_out.close() 


pickle_out = open("new_dataset/borders.pickle","wb")                    
pickle.dump(borders, pickle_out)
pickle_out.close() 



#Create an empty list named 'training_data' in which we'll store our images and their respective labels as arrays
training_data = []
img = None
label = None
i = None

#Read the images from bt_images folder from Google Drive and convert it to RGB images and store it along with their respective labels in the training_data list.
for i in range(1, 3065):
  img = cv2.imread("new_dataset/bt_images/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
  img = cv2.resize(img, (512, 512))
  label = labels[i-1]
  training_data.append([img, label])

print("shape: {} label: {} | {} samples successfully preprocessed".format(img.shape, label, i))

pickle_out = open("new_dataset/training_data.pickle","wb")                    
pickle.dump(training_data, pickle_out)
pickle_out.close()
print("done")