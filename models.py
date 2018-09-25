import tensorflow as tf
import numpy as np

# Baseline Model with 3 layer
class BaselineModel3Layer:
	def __init__(self,X,weights,biases):
		self.X = X
		self.weights = weights
		self.biases = biases

	# Define Conv layer
	def conv2d(self,x,W,b,stride):
		x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
		x = tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	def conv2d_transpose(self,x,W,b,stride):
		new_shape = x.get_shape().as_list()
		new_shape[1] = new_shape[1]*2
		new_shape[2] = new_shape[2]*2
		new_shape[3] = b.shape[0].value
		x = tf.nn.conv2d_transpose(x,W,output_shape=new_shape,strides=[1,stride,stride,1],padding="SAME")
		x = tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	def feed(self):
		self.X = self.conv2d(self.X,self.weights['wc1'],self.biases['bc1'],2)
		self.X = self.conv2d(self.X,self.weights['wc2'],self.biases['bc2'],2)
		self.X = self.conv2d(self.X,self.weights['wc3'],self.biases['bc3'],2)
		self.X = self.conv2d_transpose(self.X,self.weights['wc4'],self.biases['bc4'],2)
		self.X = self.conv2d_transpose(self.X,self.weights['wc5'],self.biases['bc5'],2)
		self.X = self.conv2d_transpose(self.X,self.weights['wc6'],self.biases['bc6'],2)
		return self.X