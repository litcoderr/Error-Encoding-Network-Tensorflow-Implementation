'''
Version 1.0 train_een_latent

[main functionality]
-used when training een latent model

Developed By James Youngchae Chee @Litcoderr
You are welcome to contribute
'''

import tensorflow as tf
import numpy as np
import argparse
import os
import skimage.io as io
from matplotlib import pyplot as plt

import dataloader
import models

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-width', type=int, default=480, help='video width')
parser.add_argument('-height', type=str, default=480, help='video height')
parser.add_argument('-pred_frame', type=int, default=5, help='number of frames to learn and predict')
parser.add_argument('-time_interval', type=int, default=2, help='time interval between frames in milliseconds')
parser.add_argument('-frame_interval', type=int, default=150, help='frame interval when generating datasets')
parser.add_argument('-batch_size', type=int, default=5, help='batch size')
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps in convnet')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch', type=int, default=500, help='number of epochs')
parser.add_argument('-videopath', type=str, default='./data/flower.mp4', help='video folder')
parser.add_argument('-tfrecordspath', type=str, default='./data/dataset.tfrecords', help='tfrecords file path')
parser.add_argument('-model_path', type=str, default='./model/deterministic/deterministic_model-10.meta', help='deterministic model path')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
arg = parser.parse_args()

### Setup Training Environment ###
# Initialize dataloader
dataloader = dataloader.dataloader(arg)
videoInfo = dataloader.getVideoInfo()
print('original width: {0[0]} original height: {0[1]} number_of_Frame: {0[2]} FPS: {0[3]}'.format(videoInfo))
# if tfrecords doesn't exist make one
if not(os.path.isfile(arg.tfrecordspath)):
	dataloader.gen_tfrecords()
else:
	print('dataloader: {} exists'.format(arg.tfrecordspath))

# Make tfrecord filename queue
file_name_queue = tf.train.string_input_producer([arg.tfrecordspath])

# Decode tfrecord file to usable numpy array
x_train , y_train = dataloader.decode(file_name_queue)


X = tf.placeholder(tf.float32, shape=(arg.batch_size,arg.height,arg.width,dataloader.channel))
Y = tf.placeholder(tf.float32, shape=(arg.batch_size,arg.height,arg.width,dataloader.channel))

# Empty dictionary to store weights
g_weights = {}
g_biases = {}
f_weights = {}
f_biases = {}
# Phi Network parameters
## W stands for Weight ;; B stands for Bias ;; P stands for Phi
phi_wc={
	'wc1' : tf.get_variable("WP1", shape=[7,7,dataloader.channel,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc2' : tf.get_variable("WP2", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc3' : tf.get_variable("WP3", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc4' : tf.get_variable("WP4", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer())
}
phi_bc={
	'bc1' : tf.get_variable("BP1", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc2' : tf.get_variable("BP2", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc3' : tf.get_variable("BP3", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc4' : tf.get_variable("BP4", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer())
}

# Operations
# Initialization
init_global_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()


with tf.Session() as sess:
	sess.run(init_global_op)
	sess.run(init_local_op)

	saver = tf.train.import_meta_graph(arg.model_path)
	saver.restore(sess,tf.train.latest_checkpoint('./model/deterministic/'))
	graph = tf.get_default_graph()

	# Load pre-trained deterministic model's weights and biases
	for i in range(1,7):
		g_weights['wc{}'.format(i)] = graph.get_tensor_by_name('W{}:0'.format(i))
		f_weights['wc{}'.format(i)] = graph.get_tensor_by_name('W{}:0'.format(i))
		g_biases['bc{}'.format(i)] = graph.get_tensor_by_name('B{}:0'.format(i))
		f_biases['bc{}'.format(i)] = graph.get_tensor_by_name('B{}:0'.format(i))

	## Train
	# Define model
	model = models.LatentResidualModel3Layer(X,Y,g_weights,f_weights,g_biases,f_biases,phi_wc,phi_bc)

	# Start Coordinator to feed in data from batch shuffler
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	for epochs in range(arg.epoch):
		print('epochs : {}'.format(epochs))
		# feed-dict in data to X : placeholder and Y : placeholder
		x_data = sess.run(x_train)
		y_data = sess.run(y_train)
		g_result = sess.run(model.g_network(),feed_dict={X:x_data,Y:y_data})
		print(g_result.shape)
	# stop coordinator and join threads
	coord.request_stop()
	coord.join(threads)

