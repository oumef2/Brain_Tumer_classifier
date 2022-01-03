import PIL
#import zipfile
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py
import cv2

labels = []
images = []
blur_data = []

#Save images of brain tumor, masks and store labels and borders in their respective lists iteratively.
filename = None
for filename in range(1, 3065):
	with h5py.File('dataset/{}.mat'.format(filename), 'r') as f:
		img = f['cjdata']['image']
		img = np.array(img, dtype=np.float32)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		img = cv2.resize(img, (512, 512))
		blur_img = cv2.bilateralFilter(img,9,75,75)
		#blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

		label = f['cjdata']['label'][0][0]
		
		images.append(blur_img)
		labels.append(int(label))
		print(blur_img.shape)
		"""
		#save images 
		plt.axis('off')
		plt.imsave("C:/Users/HP/desktop/pytry/new_dataset/blur_images/{}.jpg".format(filename), blur_img, cmap='gray')
		"""

#Convert the Python lists to a Numpy arrays
labels = np.array(labels, dtype=np.int64)
print("labels sorted...",len(labels))
print(labels[0]," to ",labels[3063])
	  
print("images sorted...",len(images))

plt.subplot(121),plt.imshow(images[2]),plt.title('Original')
plt.show()

for i in range(0, 3064):
	label = labels[i]
	img = images[i]
	blur_data.append([img, label])	

print("data matrix ready...",len(blur_data))

images = None
labels = None

pickle_out = open("new_dataset/blur_data.pickle","wb")  
print("blur_pickle opened")                  
pickle.dump(blur_data, pickle_out)
pickle_out.close()
print("done")