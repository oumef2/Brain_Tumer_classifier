import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import make_grid
import os
#import random
import numpy as np
#import pandas as pd
import pickle
#import time
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score
#from google.colab import drive 
import cv2


torch.cuda.empty_cache()
training_data = pickle.load(open("C:/Users/HP/Desktop/pytry/new_dataset/gabor_data.pickle", 'rb'))

Xt = []
yt = []
features = None
labels = None

for features,labels in training_data:
  Xt.append(features)
  yt.append(labels)

# 70 % training, 15% validating, 15% testing
X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.3, shuffle=True)  # 70% training, 30% testing
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)  # split testing set into 50% validation , 50% testing 

print("nb of training data: ",len(X_train))
print("nb of testing data :",len(X_test),"\nnb of validation data :",len(X_valid))

#free memory 
Xt = None
yt = None
features = None
labels = None
training_data = None 


t1 = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

train_set = []
test_set = []
valid_set =[]


for i in range (0,len(X_train)):
	img1 = t1(X_train[i])
	train_set.append((img1, y_train[i]))

for i in range (0,len(X_test)):
	img1 = t1(X_test[i])
	test_set.append((img1, y_test[i]))
for i in range (0,len(X_valid)):
	img1 = t1(X_valid[i])
	valid_set.append((img1, y_valid[i]))

print(len(train_set)," ",len(test_set)," ",len(valid_set))

X_train = None
y_train = None
X_valid = None
y_train = None
X_valid = None
y_valid = None

train_gen = DataLoader(train_set, batch_size=4, shuffle=True, pin_memory=True, num_workers=8)
valid_gen = DataLoader(valid_set, batch_size=4, shuffle=True, pin_memory=True, num_workers=8)
test_gen = DataLoader(test_set, batch_size=10, shuffle=True, pin_memory=True, num_workers=8)

device_name = "cuda:0:" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

print("---building model---")
# instantiate transfer learning model
resnet_model = models.resnet50(pretrained=True)

# set all paramters as trainable
for param in resnet_model.parameters():
	param.requires_grad = True

# get input of fc layer
n_inputs = resnet_model.fc.in_features

# redefine fc layer / top layer/ head for our classification problem
resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
								nn.SELU(),
								nn.Dropout(p=0.4),
								nn.Linear(2048, 2048),
								nn.SELU(),
								nn.Dropout(p=0.4),
								nn.Linear(2048, 4),
								nn.LogSigmoid())
# set all paramters of the model as trainable
for name, child in resnet_model.named_children():
  for name2, params in child.named_parameters():
  	params.requires_grad = True
# set model to run on GPU or CPU absed on availibility
resnet_model.to(device)

# print the trasnfer learning NN model's architecture
resnet_model

print("---training configuration---")
# if GPU is available set loss function to use GPU
criterion = nn.CrossEntropyLoss().to(device)

# optimizer
optimizer = torch.optim.SGD(resnet_model.parameters(), momentum=0.9, lr=3e-4)

# number of training iterations
epochs = 30
# set best_prec loss value as 2 for checkpoint threshold
best_prec1 = 2

# empty lists to store losses and accuracies
train_losses = []
test_losses = []
train_correct = []
test_correct = []

b = None
train_b = None
test_b = None
print("traning start...")

for i in range(epochs):
	# empty training correct and test correct counter as 0 during every iteration
	trn_corr = 0
	tst_corr = 0
	
	# train in batches
	for b, (y, X) in enumerate(train_gen):
		print(X,y)
		# set label as cuda if device is cuda
		X, y = X.to(device), y.to(device)

		# forward pass image sample
		y_pred = resnet_model(X.view(-1, 3, 512, 512))
		# calculate loss
		loss = criterion(y_pred.float(), torch.argmax(y.view(32, 4), dim=1).long())
		# get argmax of predicted tensor, which is our label
		predicted = torch.argmax(y_pred, dim=1).data
		# if predicted label is correct as true label, calculate the sum for samples
		batch_corr = (predicted == torch.argmax(y.view(32, 4), dim=1)).sum()
		# increment train correct with correcly predicted labels per batch
		trn_corr += batch_corr
		# set optimizer gradients to zero
		optimizer.zero_grad()
		# back propagate with loss
		loss.backward()
		# perform optimizer step
		optimizer.step()
	# print training metrics
	print(f'Epoch {(i+1)} Batch {(b+1)*4}\nAccuracy: {trn_corr.item()*100/(4*8*b):2.2f} %  Loss: {loss.item():2.4f}  Duration: {((e_end-e_start)/60):.2f} minutes') # 4 images per batch * 8 augmentations per image * batch length

	# some metrics storage for visualization
	train_b = b
	train_losses.append(loss)
	train_correct.append(trn_corr)

	X, y = None, None
	# validate using validation generator
	# do not perform any gradient updates while validation
	with torch.no_grad():
		for b, (y, X) in enumerate(valid_gen):
			# set label as cuda if device is cuda
			X, y = X.to(device), y.to(device)
			# forward pass image
			y_val = resnet_model(X.view(-1, 3, 512, 512))
			# get argmax of predicted tensor, which is our label
			predicted = torch.argmax(y_val, dim=1).data
			# increment test correct with correcly predicted labels per batch
			tst_corr += (predicted == torch.argmax(y.view(32, 4), dim=1)).sum()

	# get loss of validation set
	loss = criterion(y_val.float(), torch.argmax(y.view(32, 4), dim=1).long())
	# print validation metrics
	print(f'Validation Accuracy {tst_corr.item()*100/(4*8*b):2.2f} Validation Loss: {loss.item():2.4f}\n')
	# if current validation loss is less than previous iteration's validatin loss create and save a checkpoint
	is_best = loss < best_prec1
	best_prec1 = min(loss, best_prec1)
	# some metrics storage for visualization
	test_b  = b
	test_losses.append(loss)
	test_correct.append(tst_corr)

print("training is over")