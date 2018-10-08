import tensorflow as tf
import numpy as np

# Baseline Model(Deterministic) with 3 layer
class BaselineModel3Layer:
	def __init__(self,X,weights,biases):
		# X : input X data (shape : [batch_size,height,width,num_channel] type: tensor)
		# weights : Weights of deterministic model (type: dictionary)
		# biases : Biases of deterministic model (type: dictionary)
		self.X = X
		self.weights = weights
		self.biases = biases

	# Define Conv layer (for encoder)
	# Activation : ReLU
	def conv2d(self,x,W,b,stride):
		x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
		x = tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	# Define Transpose Convolutional layer (for decoder)
	# Activation : ReLU
	def conv2d_transpose(self,x,W,b,stride):
		# Calculate new Output shape for Transpose Convolutional Operation
		new_shape = x.get_shape().as_list()
		new_shape[1] = new_shape[1]*2
		new_shape[2] = new_shape[2]*2
		new_shape[3] = b.shape[0].value
		# Compute Transpose Convolution
		x = tf.nn.conv2d_transpose(x,W,output_shape=new_shape,strides=[1,stride,stride,1],padding="SAME")
		x = tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	# Feeding Operation of Deterministic Model
	def feed(self):
		# 1. Encode with 3 Convolutional layer
		self.X = self.conv2d(self.X,self.weights['wc1'],self.biases['bc1'],2)
		self.X = self.conv2d(self.X,self.weights['wc2'],self.biases['bc2'],2)
		self.X = self.conv2d(self.X,self.weights['wc3'],self.biases['bc3'],2)
		# 2. Decode with 3 Transpose Convolutional layer
		self.X = self.conv2d_transpose(self.X,self.weights['wc4'],self.biases['bc4'],2)
		self.X = self.conv2d_transpose(self.X,self.weights['wc5'],self.biases['bc5'],2)
		self.X = self.conv2d_transpose(self.X,self.weights['wc6'],self.biases['bc6'],2)
		# 3. Return clipped output value for it to range between -1 and 1
		# --> for generating valid image
		return tf.clip_by_value(self.X,-1,1)

# Latent Residual Model (Phi network Implemented) with 3 layer
class LatentResidualModel3Layer:
	def __init__(self,X,Y,g_weights,f_weights,g_biases,f_biases,phi_weights,phi_biases):
		# X, Y : input X and Y(target) data
		# g_ : stands for G Network (pre-trained deterministic network)
		# f_ : stands for F Network (latent variable implemented Latent Residual Network)
		# phi_ : stand for Phi Network (Error Encoding Phi Network)

		self.X = X
		self.Y = Y
		self.g_weights = g_weights
		self.f_weights = f_weights
		self.g_biases = g_biases
		self.f_biases = f_biases
		self.phi_weights = phi_weights
		self.phi_biases = phi_biases

	# Define Conv layer (for encoder)
	# Activation : ReLU
	def conv2d(self,x,W,b,stride):
		x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
		x = tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	# Define Transpose Convolutional layer (for decoder)
	# Activation : ReLU
	def conv2d_transpose(self,x,W,b,stride):
		# Calculate new Output shape for Transpose Convolutional Operation
		new_shape = x.get_shape().as_list()
		new_shape[1] = new_shape[1]*2
		new_shape[2] = new_shape[2]*2
		new_shape[3] = b.shape[0].value
		# Compute Transpose Convolution
		x = tf.nn.conv2d_transpose(x,W,output_shape=new_shape,strides=[1,stride,stride,1],padding="SAME")
		x = tf.nn.bias_add(x,b)
		return tf.nn.relu(x)

	# Feeding Operation of Deterministic Model
	def feed(self):
		# 1. Get g_network result
		g_result = self.g_network()

		# 2. Compute Residual Error
		residual_error = self.Y-g_result
		
		# 3. Get z latent variable from phi network
		z = self.phi_network(residual_error)
		
		# 4. Encode z to embedable form
		z_emb = self.latent_encoder(z)
		
		# 5. Get f_network result
		f_result = self.f_network(z_emb)

		# Return G network result, F network result, and Latent Variable
		return g_result, f_result, z

	# Define G Network (Same with Deterministic Model)
	def g_network(self):
		# 1. Encode with 3 Convolutional layer
		result = self.conv2d(self.X,self.g_weights['wc1'],self.g_biases['bc1'],2)
		result = self.conv2d(result,self.g_weights['wc2'],self.g_biases['bc2'],2)
		result = self.conv2d(result,self.g_weights['wc3'],self.g_biases['bc3'],2)
		# 2. Decode with 3 Transpose Convolutional layer
		result = self.conv2d_transpose(result,self.g_weights['wc4'],self.g_biases['bc4'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc5'],self.g_biases['bc5'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc6'],self.g_biases['bc6'],2)
		# 3. Return clipped output value for it to range between -1 and 1
		# --> for generating valid image
		return tf.clip_by_value(result,-1,1)

	# Define F Network (Encoded Latent Variable Implemented)
	def f_network(self,z_emb):
		# 1. Encode with 3 Convolutional layer
		result = self.conv2d(self.X,self.f_weights['wc1'],self.f_biases['bc1'],2)
		result = self.conv2d(result,self.f_weights['wc2'],self.f_biases['bc2'],2)
		result = self.conv2d(result,self.f_weights['wc3'],self.f_biases['bc3'],2)

		# 2. Add Latent Variable (z_emb --> encoded latent variable)
		result = result + z_emb

		# 3. Decode with 3 Transpose Convolutional layer
		result = self.conv2d_transpose(result,self.g_weights['wc4'],self.g_biases['bc4'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc5'],self.g_biases['bc5'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc6'],self.g_biases['bc6'],2)
		# 4. Return clipped output value for it to range between -1 and 1
		# --> for generating valid image
		return tf.clip_by_value(result,-1,1)

	# Define Phi Network (Error Encoding Network with 4 Conv layers)
	def phi_network(self,residual_error):
		# Get residual_error as input
		# 1. Encode residual error with 4 conv layers
		conv_result = self.conv2d(residual_error,self.phi_weights['wc1'],self.phi_biases['bc1'],2)
		conv_result = self.conv2d(conv_result,self.phi_weights['wc2'],self.phi_biases['bc2'],2)
		conv_result = self.conv2d(conv_result,self.phi_weights['wc3'],self.phi_biases['bc3'],2)
		conv_result = self.conv2d(conv_result,self.phi_weights['wc4'],self.phi_biases['bc4'],2)
		# 2. Reshape Conv Result to [n_batch, -1]
		conv_result = tf.reshape(conv_result,[conv_result.shape[0].value,-1])
		# 3. Compute Fully Connected Process of Phi Network
		fc_result = tf.nn.relu(tf.matmul(conv_result,self.phi_weights['wf1'])+self.phi_biases['bf1'])
		fc_result = tf.nn.relu(tf.matmul(fc_result,self.phi_weights['wf2'])+self.phi_biases['bf2'])
		fc_result = tf.nn.relu(tf.matmul(fc_result,self.phi_weights['wf3'])+self.phi_biases['bf3'])
		# 4. Return latent variable
		return fc_result

	# Encodes latent variable using fully connected network
	# Converts latet variable to the size of '-nfeature'
	# --> For latent variable to be addable with encoded F network layer
	def latent_encoder(self,latent):
		# Matrix Multiplication
		result = tf.matmul(latent,self.phi_weights['wf4'])
		# Reshape to right size
		result = tf.reshape(result,[result.shape[0].value,1,1,result.shape[1].value])
		# Return Result
		return result











