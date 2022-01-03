import PIL
#import zipfile
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py
import cv2

def gabor(img):
	# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
	# ksize - size of gabor filter (n, n)
	# sigma - standard deviation of the gaussian function
	# theta - orientation of the normal to the parallel stripes
	# lambda - wavelength of the sunusoidal factor
	# gamma - spatial aspect ratio
	# psi - phase offset
	# ktype - type and range of values that each pixel in the gabor kernel can hold
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ksize =( 5,5)
	sigma = 1
	theta = 6*np.pi/4
	lamda = np.pi/4

	g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
	#g_kernel /= 1.5*g_kernel.sum()
	#img = cv2.imread("C:/Users/HP/desktop/pytry/new_dataset/bt_images/5.jpg")
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

	return filtered_img

labels = []
images = []
gabor_data = []

#Save images of brain tumor, masks and store labels and borders in their respective lists iteratively.
filename = None
for filename in range(1, 3065):
	with h5py.File('dataset/{}.mat'.format(filename), 'r') as f:
		label = f['cjdata']['label'][0][0]
		labels.append(int(label))
		img = f['cjdata']['image']
		img = np.array(img, dtype=np.float32)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		img = cv2.resize(img, (512, 512))
		gabor_img = gabor(img)
		images.append(gabor_img)
		print(gabor_img.shape)
		"""
		plt.axis('off')
		plt.imsave("new_dataset/gabor_images/{}.jpg".format(filename), gabor_img, cmap='gray')
		"""

#Convert the Python lists to a Numpy arrays
labels = np.array(labels, dtype=np.int64)
print("labels sorted...",len(labels))
print(labels[0]," to ",labels[3063])
"""
for filename in range(1,3065):
	#img = cv2.imread("new_dataset/bt_images/{}.jpg".format(filename))

	img = cv2.resize(img, (512, 512))
	gabor_img = gabor(img)
	images.append(gabor_img)
	print(gabor_img.shape)
"""

	  
print("images sorted...",len(images))
#plot random image to test 
plt.subplot(121),plt.imshow(images[654]),plt.title('Original')
plt.show()

for i in range(0, 3064):
	label = labels[i]
	img = images[i]
	gabor_data.append([img, label])	

print("data matrix ready...",len(gabor_data))

images = None
labels = None

pickle_out = open("new_dataset/gabor_data.pickle","wb")  
print("gabor_pickle opened")                  
pickle.dump(gabor_data, pickle_out)
pickle_out.close()
print("done")