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

def get_pixel(img, center, x, y):
	new_value = 0
	try:
		if img[x][y] >= center:
			new_value = 1
	except:
		pass
	return new_value
def lbp_calculated_pixel(img, x, y):
	'''
	 64 | 128 |   1
	----------------
	 32 |   0 |   2
	----------------
	 16 |   8 |   4    
	'''    
	center = img[x][y]
	val_ar = []
	val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
	val_ar.append(get_pixel(img, center, x, y+1))       # right
	val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
	val_ar.append(get_pixel(img, center, x+1, y))       # bottom
	val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
	val_ar.append(get_pixel(img, center, x, y-1))       # left
	val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
	val_ar.append(get_pixel(img, center, x-1, y))       # top
	
	power_val = [1, 2, 4, 8, 16, 32, 64, 128]
	val = 0
	for i in range(len(val_ar)):
		val += val_ar[i] * power_val[i]
	return val    


def LBP(img):
	print(img.shape)
	height, width, channel = img.shape
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	img_lbp = np.zeros((height, width,3), np.uint8)
	for i in range(0, height):
		for j in range(0, width):
			img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
	return img_lbp
img = None

labels = []
images = []
lbp_data = []

#Save images of brain tumor, masks and store labels and borders in their respective lists iteratively.
filename = None
for filename in range(1, 3065):
	with h5py.File('dataset/{}.mat'.format(filename), 'r') as f:
		
		img = f['cjdata']['image']
		img = np.array(img, dtype=np.float32)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		img = cv2.resize(img, (512, 512))

		lbp_img = LBP(img)
		
		label = f['cjdata']['label'][0][0]
		
		images.append(lbp_img)
		labels.append(int(label))
		print(lbp_img.shape)
		"""
		plt.axis('off')
		plt.imsave("C:/Users/HP/downloads/pytry/new_dataset/lbp_images/{}.jpg".format(filename), lbp_img, cmap='gray')
	"""

#Convert the Python lists to a Numpy arrays
labels = np.array(labels, dtype=np.int64)
print(labels[0]," to ",labels[3063])
"""
for i in range(1,3065):
	img = cv2.imread('C:/Users/HP/desktop/pytry/new_dataset/lbp_images/{}.jpg'.format(i))
	images.append(img)
"""
	  
print("images sorted...",len(images))

plt.subplot(121),plt.imshow(images[2]),plt.title('Original')
plt.show()
for i in range(0, 3064):
	label = labels[i]
	img = images[i]
	lbp_data.append([img, label])	

print("data matrix ready...",len(lbp_data))

images = None
labels = None

pickle_out = open("new_dataset/lbp_data.pickle","wb")  
print("lbp_pickle opened")                  
pickle.dump(lbp_data, pickle_out)
pickle_out.close()
print("done")