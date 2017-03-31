import numpy as np
from keras.utils import np_utils

def corrupt_images(x_train, std=0):
    '''
    Corrupting the images
    '''
    x_train_corrupted = np.random.normal(x_train, scale=std/255)
    # Make sure that the values are still between 0 and 1
    x_train_corrupted[x_train_corrupted > 1] = 1
    x_train_corrupted[x_train_corrupted < 0] = 0

    return x_train_corrupted

def corrupt_labels(y_train, percentage):
    '''
    Corrupting the labels
    '''
    # random boolean mask for which values will be changed
    mask = np.random.choice(a=[0,1],size=y_train.shape,p=[1-percentage, percentage]).astype(np.bool)
    # vector with random labels
    r = np.random.randint(0,10,size=y_train.shape)
    # replace
    y_train_corrupted = np.empty([60000,])
    y_train_corrupted[mask] = r[mask]
    y_train_corrupted[np.logical_not(mask)] = y_train[np.logical_not(mask)]
    # reshape
    y_train_cat_corrupted = np_utils.to_categorical(y_train_corrupted)

    return y_train_cat_corrupted
