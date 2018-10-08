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
	def __init__(self,X,Y,g_weights,f_weights,g_biases,f_biases,phi_weights,phi_biases):
		self.X = X
		self.Y = Y
		self.g_weights = g_weights
		self.f_weights = f_weights
		self.g_biases = g_biases
		self.f_biases = f_biases
		self.phi_weights = phi_weights
		self.phi_biases = phi_biases

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

	def train(self):
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

		return g_result, f_result, z

	def g_network(self):
		result = self.conv2d(self.X,self.g_weights['wc1'],self.g_biases['bc1'],2)
		result = self.conv2d(result,self.g_weights['wc2'],self.g_biases['bc2'],2)
		result = self.conv2d(result,self.g_weights['wc3'],self.g_biases['bc3'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc4'],self.g_biases['bc4'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc5'],self.g_biases['bc5'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc6'],self.g_biases['bc6'],2)
		return tf.clip_by_value(result,-1,1) # Limit X range to generate valid imags

	def f_network(self,z_emb):
		# Encode
		result = self.conv2d(self.X,self.f_weights['wc1'],self.f_biases['bc1'],2)
		result = self.conv2d(result,self.f_weights['wc2'],self.f_biases['bc2'],2)
		result = self.conv2d(result,self.f_weights['wc3'],self.f_biases['bc3'],2)

		# Add Latent Variable
		result = result + z_emb

		# Decode
		result = self.conv2d_transpose(result,self.g_weights['wc4'],self.g_biases['bc4'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc5'],self.g_biases['bc5'],2)
		result = self.conv2d_transpose(result,self.g_weights['wc6'],self.g_biases['bc6'],2)
		return tf.clip_by_value(result,-1,1)

	def phi_network(self,residual_error):
		# Compute Convolutional Process of Phi Network
		conv_result = self.conv2d(residual_error,self.phi_weights['wc1'],self.phi_biases['bc1'],2)
		conv_result = self.conv2d(conv_result,self.phi_weights['wc2'],self.phi_biases['bc2'],2)
		conv_result = self.conv2d(conv_result,self.phi_weights['wc3'],self.phi_biases['bc3'],2)
		conv_result = self.conv2d(conv_result,self.phi_weights['wc4'],self.phi_biases['bc4'],2)
		# Reshape Conv Result to [n_batch, -1]
		conv_result = tf.reshape(conv_result,[conv_result.shape[0].value,-1])
		# Compute Fully Connected Process of Phi Network
		fc_result = tf.nn.relu(tf.matmul(conv_result,self.phi_weights['wf1'])+self.phi_biases['bf1'])
		fc_result = tf.nn.relu(tf.matmul(fc_result,self.phi_weights['wf2'])+self.phi_biases['bf2'])
		fc_result = tf.nn.relu(tf.matmul(fc_result,self.phi_weights['wf3'])+self.phi_biases['bf3'])
		return fc_result

	def latent_encoder(self,latent):
		result = tf.matmul(latent,self.phi_weights['wf4'])
		result = tf.reshape(result,[result.shape[0].value,1,1,result.shape[1].value])
		return result











