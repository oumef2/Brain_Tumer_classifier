#import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.preprocessing import image

training_data = pickle.load(open('C:/users/hp/desktop/pytry/new_dataset/blur_data.pickle', 'rb'))

X = []
y = []
image = None
label = None
for image,label in training_data:
	#image = np.array(image, dtype=np.float32)
	if (image.shape != (512, 512)):
		continue
	X.append(image)
	y.append(label)
#y= np.array(y, dtype=np.int64)
print("xshape",image.shape," y shape",label.shape)

training_data = None
image = None
label = None
X = np.array(X)
y= np.array(y)
img_width, img_height = 512, 512
X_train, X_vald, y_train, y_vald = train_test_split(X, y, test_size=0.3, shuffle=True)  # 70% training, 30% testing

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
nb_epoch = 10
nb_train_samples = len(X_train)
nb_validation_samples = len(X_vald)
"""
dataAugmentaion = ImageDataGenerator(rotation_range = 90, horizontal_flip = True)
training_data = dataAugmentaion.flow(X_train, y_train)
"""
validation_data = (X_vald, y_vald)

# training the model

model.fit_generator(
		(X_train,y_train),
		samples_per_epoch=nb_train_samples,
		nb_epoch=nb_epoch,
		validation_data=validation_data,
		nb_val_samples=nb_validation_samples)

model.evaluate_generator(validation_generator, nb_validation_samples)