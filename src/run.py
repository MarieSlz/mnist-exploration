'''
CNN implementation for the MNIST dataset

Marie Schiltz
'''

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
import argparse

from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from lib import input_data
from lib import graph
from lib import model
from lib import corrupt
from keras import backend as K

# Parameters
NUM_EPOCHS = 1
SIZE_BATCHS = 128
NUM_BATCHS = 60000 // SIZE_BATCHS


def load(depth, width, height):
    '''
    Loading datasets
    '''
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

    x_train = x_train.reshape((x_train.shape[0], depth, width, height)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], depth, width, height)).astype('float32')

    return (x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    # Model Parameters
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-f', '--folder', type=str, default="default")
    parser.add_argument('-i', '--noiseImages', type=int, default=0)
    parser.add_argument('-l', '--noiseLabels', type=int, default=0)
    args = parser.parse_args()

    # Image Definition
    num_classes = 10
    num_pixels = 784
    width, height, depth = 28, 28, 1

    # Loading data
    (x_train, x_test, y_train, y_test) = load(depth, width, height)
    y_train_cat = np_utils.to_categorical(y_train)
    y_test_cat = np_utils.to_categorical(y_test)

    # Training Model
    if (args.noiseImages !=0 and args.noiseLabels == 0):
        print("[INFO] Adding Additional Gaussian Noise to the Images (std=", args.noiseImages,")", sep="")
        x_train_corrupted = corrupt.corrupt_images(x_train, args.noiseImages)
        (trained_model, history) = model.train(x_train_corrupted, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
                                               width, height, depth, num_classes)
    elif (args.noiseImages ==0 and args.noiseLabels != 0):
        print("[INFO] Replacing randomly ", args.noiseImages, "% of the labels", sep="")
        y_train_cat_corrupted = corrupt.corrupt_labels(y_train, args.noiseLabels/100)
        (trained_model, history) = model.train(x_train, y_train_cat_corrupted, SIZE_BATCHS, NUM_EPOCHS,
                                               width, height, depth, num_classes)
    elif (args.noiseImages !=0 and args.noiseLabels != 0):
        print("[INFO] Adding Additional Gaussian Noise to the Images (std=", args.noiseImages, ")", sep="")
        print("[INFO] Replacing randomly ", args.noiseImages, "% of the labels", sep="")
        x_train_corrupted = corrupt.corrupt_images(x_train, args.noiseImages)
        y_train_cat_corrupted = corrupt.corrupt_labels(y_train, args.noiseLabels/100)
        (trained_model, history) = model.train(x_train_corrupted, y_train_cat_corrupted, SIZE_BATCHS, NUM_EPOCHS,
                                               width, height, depth, num_classes)
    else:
        (trained_model, history) = model.train(x_train, y_train_cat, SIZE_BATCHS, NUM_EPOCHS,
                                               width, height, depth, num_classes)

    # Testing Model
    (y_pred, accuracy) = model.fit(x_test, y_test_cat, trained_model, args.folder)
    path = os.path.join("../output/raw", args.folder, "y_pred.npy")
    np.save(path, y_pred)
    # Save and Plot results
    print("\n[INFO] Saving and Ploting results")
    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    graph.save_cnf_matrix(cnf_matrix, args.folder)
    # Prediction Per Class - Multiclass classification
    # Plot and Save, Test Error, Precision and Recall
    # Precision: fraction of retrieved instances that are relevant tp/(tp+fp)
    # Recall: fraction of relevant instances that are retrieved tp/(tp+fn)
    score = precision_recall_fscore_support(y_test, y_pred,
                                            average=None,
                                            labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    graph.error(score, args.folder)
    graph.precision(score, args.folder)
    graph.recall(score, args.folder)
    # Loss per batch
    graph.loss(history.losses, NUM_BATCHS, args.folder)
