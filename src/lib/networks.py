'''
Those networks have been inspired from:
http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
'''
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout

# Implementation of the LeNet network
class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		'''
		Building the LeNet network
		'''
		# initialize the model
		model = Sequential()

		# first set of CONV => RELU => POOL
		model.add(Conv2D(20, (5, 5), padding="same",
			      input_shape=(depth, height, width)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV => RELU => POOL
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

class ConvNet:
	def build(width, height, depth, classes):
		'''
		Building a simple cnn
		'''
		# initialize the model
		model = Sequential()
		model.add(Convolution2D(30, 5, 5, border_mode='valid',
								input_shape=(1, 28, 28), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(15, 3, 3, activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(classes, activation='softmax'))

		return model


# For testing - define very simple & fast to run model
class NetworkTest:
	def build(num_pixels, num_classes):
		'''
		Building test network - sequential network, fast to train
		'''
		# initialize the model
		model = Sequential()

		# set of two Dense layers with activation relu and softmax
		model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
		model.add(Dense(num_classes, init='normal', activation='softmax'))

		return model
