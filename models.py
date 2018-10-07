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
		return tf.clip_by_value(self.X,-1,1) # Limit X range to generate valid image

class LatentResidualModel3Layer:
	def __init__(self,X,Y,g_weights,f_weights,g_biases,f_biases,phi_wc,phi_bc):
		self.X = X
		self.Y = Y
		self.g_weights = g_weights
		self.f_weights = f_weights
		self.g_biases = g_biases
		self.f_biases = f_biases
		self.phi_wc = phi_wc
		self.phi_bc = phi_bc

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

	def g_network(self):
		result = self.conv2d(self.X,self.g_weights['wc1'],self.g_biases['bc1'],2)
		result = self.conv2d(result,self.g_weights['wc2'],self.g_biases['bc2'],2)
		result = self.conv2d(result,self.g_weights['wc3'],self.g_biases['bc3'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc4'],self.g_biases['bc4'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc5'],self.g_biases['bc5'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc6'],self.g_biases['bc6'],2)
		return tf.clip_by_value(result,-1,1) # Limit X range to generate valid image

	def f_encoder(self):
		result = 0
		return result

	def phi_network(self):
		result = 0
		return result

	def f_decoder(self):
		result = 0
		return result
















