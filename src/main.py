'''
This is the main file of my project, you can reproduce all outputs by running this file.
'''
# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from lib import input_data
from lib import graph
from lib import model
from lib import corrupt
from keras import backend as K

### Keras Backend
#print(K.backend())
#print(K.image_dim_ordering(), ": ", K.image_data_format())

### Loading data
# Training Set: 60,000 datapoints - Test Set: 10,000 datapoints
print("[INFO] Downloading/fetching MNIST dataset")
dataset = input_data.read_data_sets("../data", False)
# Images
# Each image is a vector of dimension 784 (1*28*28 - one channel, grayscale images)
# Values are between 0 and 1, they are already normalized
# Note: I won't use the valisation dataset to tune hyperparameters
x_train = np.concatenate((dataset.train.images, dataset.validation.images), axis=0)
x_test = dataset.test.images
# Labels
y_train = np.concatenate((dataset.train.labels, dataset.validation.labels), axis=0)
y_test = dataset.test.labels

### Plot Sample images (4 - grayscale)
graph.plot_sample(x_train)

### Image Definition
# number of classes
num_classes = 10
# number of pixels per image
num_pixels = 784
# width, height, depth
width, height, depth = 28, 28, 1

### Formating Data
y_train_cat = np_utils.to_categorical(y_train)
y_test_cat = np_utils.to_categorical(y_test)
x_train = x_train.reshape((x_train.shape[0], depth, width, height)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], depth, width, height)).astype('float32')

################################ PART 1 ################################
print("\n[PART 1]\n")

### Parameters
NUM_EPOCHS = 1
SIZE_BATCHS = 128
NUM_BATCHS = 60000 // SIZE_BATCHS

### Training Model
(trained_model, history) = model.train(x_train, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
                         width, height, depth, num_classes)

### Testing Model
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_0")

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_0")


################################ PART 2 ################################
print("\n[PART 2]\n")

### Store results
error_p2 = list()
std_list = list()

### Add gaussian noise to the images
print("[INFO] Adding Additional Gaussian Noise to the Images (std=8)")
std = 8; std_list.append(std)
x_train_corrupted = corrupt.corrupt_images(x_train, std)

### Train and fit model using the same Parameters
(trained_model, history) = model.train(x_train_corrupted, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
                                       width, height, depth, num_classes)
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_1")
error_p2.append((1-accuracy)*100)

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_1")


### Add gaussian noise to the images
print("[INFO] Adding Additional Gaussian Noise to the Images (std=32)")
std = 32; std_list.append(std)
x_train_corrupted = corrupt.corrupt_images(x_train, std)

### Train and fit model using the same Parameters
(trained_model, history) = model.train(x_train_corrupted, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
                                       width, height, depth, num_classes)
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_2")
error_p2.append((1-accuracy)*100)

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_2")


### Add gaussian noise to the images
print("[INFO] Adding Additional Gaussian Noise to the Images (std=128)")
std = 128; std_list.append(std)
x_train_corrupted = corrupt.corrupt_images(x_train, std)

### Train and fit model using the same Parameters
(trained_model, history) = model.train(x_train_corrupted, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
                                       width, height, depth, num_classes)
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_3")
error_p2.append((1-accuracy)*100)

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_3")

### Plot Summary Graph for Part 2
print("[INFO] Saving Summary Graph of part 2")
graph.std_error(std_list, error_p2)

################################ PART 3 ################################
print("\n[PART 3]\n")

### Store results
error_p3 = list()
per_list = list()

### Add noise to the labels
print("[INFO] Training with 5'%' of noisy labels")
per = 0.05; per_list.append(per*100)
y_train_cat_corrupted = corrupt.corrupt_labels(y_train, per)

### Train and fit model using the same Parameters
(trained_model, history) = model.train(x_train, y_train_cat_corrupted, SIZE_BATCHS, NUM_EPOCHS,
                                       width, height, depth, num_classes)
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_4")
error_p3.append((1-accuracy)*100)

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_4")


### Add noise to the labels
print("[INFO] Training with 15'%' of noisy labels")
per = 0.15; per_list.append(per*100)
y_train_cat_corrupted = corrupt.corrupt_labels(y_train, per)

### Train and fit model using the same Parameters
(trained_model, history) = model.train(x_train, y_train_cat_corrupted, SIZE_BATCHS, NUM_EPOCHS,
                                       width, height, depth, num_classes)
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_5")
error_p3.append((1-accuracy)*100)

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_5")


### Add noise to the labels
print("[INFO] Training with 50'%' of noisy labels")
per = 0.50; per_list.append(per*100)
y_train_cat_corrupted = corrupt.corrupt_labels(y_train, per)

### Train and fit model using the same Parameters
(trained_model, history) = model.train(x_train, y_train_cat_corrupted, SIZE_BATCHS, NUM_EPOCHS,
                                       width, height, depth, num_classes)
(y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, "output_6")
error_p3.append((1-accuracy)*100)

### Save and Plot results
graph.output_graphs(y_test, y_pred, history, NUM_BATCHS, "output_6")

### Plot Summary Graph for Part 3
print("[INFO] Saving Summary Graph of part 3")
graph.per_error(per_list, error_p3)
