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
parser.add_argument('-data_interval', type=int, default=150, help='number of frame interval between start of each dataset')
parser.add_argument('-batch_size', type=int, default=5, help='batch size')
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps in convnet')
parser.add_argument('-nlatent', type=int, default=4, help='Number of Latent Variables')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch', type=int, default=500, help='number of epochs')
parser.add_argument('-videopath', type=str, default='./data/flower.mp4', help='video folder')
parser.add_argument('-tfrecordspath', type=str, default='./data/dataset.tfrecords', help='tfrecords file path')
parser.add_argument('-tensorboard_path', type=str, default='./results/tensorboard/test1/latent', help='where tensorboard data is stored')
parser.add_argument('-deterministic_model_path', type=str, default='./model/deterministic/deterministic_model-10.meta', help='deterministic model path')
parser.add_argument('-model_save_path', type=str, default='./model/latent/latent_model', help='latent model path')
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

## --------------- Variables ---------------- ##

# Empty dictionary to store pre-trained weights
g_weights = {}
g_biases = {}
f_weights = {}
f_biases = {}
# List to store trainable weights
trainable = []

# Phi Network parameters
# --> W stands for Weight ;; B stands for Bias ;; P stands for Phi
fc_size = 30*30*64
phi_weights={
	'wc1' : tf.get_variable("WCP1", shape=[7,7,dataloader.channel,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc2' : tf.get_variable("WCP2", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc3' : tf.get_variable("WCP3", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc4' : tf.get_variable("WCP4", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wf1' : tf.get_variable("WFP1", shape=[fc_size,100],initializer=tf.contrib.layers.xavier_initializer()),
	'wf2' : tf.get_variable("WFP2", shape=[100,100],initializer=tf.contrib.layers.xavier_initializer()),
	'wf3' : tf.get_variable("WFP3", shape=[100,arg.nlatent],initializer=tf.contrib.layers.xavier_initializer()),
	'wf4' : tf.get_variable("WFP4", shape=[arg.nlatent,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer())
}
phi_biases={
	'bc1' : tf.get_variable("BCP1", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc2' : tf.get_variable("BCP2", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc3' : tf.get_variable("BCP3", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc4' : tf.get_variable("BCP4", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bf1' : tf.get_variable("BFP1", shape=[100],initializer=tf.contrib.layers.xavier_initializer()),
	'bf2' : tf.get_variable("BFP2", shape=[100],initializer=tf.contrib.layers.xavier_initializer()),
	'bf3' : tf.get_variable("BFP3", shape=[arg.nlatent],initializer=tf.contrib.layers.xavier_initializer())
}

# Session to Retrieve Pre-trained Variables
with tf.Session() as sess:
	# Initialize variables
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Create saver --> import meta graph (pre-trained graph)
	saver = tf.train.import_meta_graph(arg.deterministic_model_path)
	# restore graph
	saver.restore(sess,tf.train.latest_checkpoint('./model/deterministic/'))
	graph = tf.get_default_graph()

	# Load pre-trained weights and biases as numpy array
	for i in range(1,7):
		g_weights['wc{}'.format(i)] = sess.run(graph.get_tensor_by_name('W{}:0'.format(i)))
		f_weights['wc{}'.format(i)] = sess.run(graph.get_tensor_by_name('W{}:0'.format(i)))
		g_biases['bc{}'.format(i)] = sess.run(graph.get_tensor_by_name('B{}:0'.format(i)))
		f_biases['bc{}'.format(i)] = sess.run(graph.get_tensor_by_name('B{}:0'.format(i)))
# Convert numpy array to trainable tf.Variable
for i in range(1,7):
	g_weights['wc{}'.format(i)] = tf.Variable(g_weights['wc{}'.format(i)],name='WG{}'.format(i))
	f_weights['wc{}'.format(i)] = tf.Variable(f_weights['wc{}'.format(i)],name='WF{}'.format(i))
	g_biases['bc{}'.format(i)] = tf.Variable(g_biases['bc{}'.format(i)],name='BG{}'.format(i))
	f_biases['bc{}'.format(i)] = tf.Variable(f_biases['bc{}'.format(i)],name='BF{}'.format(i))

# List of weights to train
trainable = trainable + list(f_weights.values())
trainable = trainable + list(f_biases.values())
trainable = trainable + list(phi_weights.values())
trainable = trainable + list(phi_biases.values())

# Storing f and phi weights and biases as tf.summary
for i in range(1,7):
	tf.summary.histogram('f_wc{}'.format(i),f_weights['wc{}'.format(i)])
	tf.summary.histogram('f_bc{}'.format(i),f_biases['bc{}'.format(i)])
for i in range(1,5):
	tf.summary.histogram('phi_wc{}'.format(i),phi_weights['wc{}'.format(i)])
	tf.summary.histogram('phi_bc{}'.format(i),phi_biases['bc{}'.format(i)])
for i in range(1,4):
	tf.summary.histogram('phi_wf{}'.format(i),phi_weights['wf{}'.format(i)])
	tf.summary.histogram('phi_bf{}'.format(i),phi_biases['bf{}'.format(i)])
tf.summary.histogram('phi_wf4',phi_weights['wf4'])

## --------------- Operations ---------------- ##

# Make tfrecord filename queue
file_name_queue = tf.train.string_input_producer([arg.tfrecordspath])
# Decode tfrecord file to usable numpy array
x_train , y_train = dataloader.decode(file_name_queue)
x_val = tf.identity(x_train,'x_val')
y_val = tf.identity(y_train,'y_val')
# Create Latent Implemented Model
model = models.LatentResidualModel3Layer(x_train,y_train,g_weights,f_weights,g_biases,f_biases,phi_weights,phi_biases)
# Feeding Operation
feed_op = model.feed()
g_feed_op_save = tf.identity(feed_op[0],'deterministic_feed_op')
f_feed_op_save = tf.identity(feed_op[1],'latent_feed_op')
# Loss Operation
loss_op = tf.losses.mean_squared_error(labels=y_train,predictions=feed_op[1])
tf.summary.scalar('loss',loss_op)
# Optimization Operation (Optimize only Variables in 'trainable' list)
optimize_op = tf.train.AdamOptimizer(arg.lrt).minimize(loss_op,var_list=trainable)

## --------------- Training ---------------- ##

with tf.Session() as sess:
	# Initialize
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Merge all summary
	merged_summary = tf.summary.merge_all()

	# Saver Object to save all the variables
	saver = tf.train.Saver()
	# Set up Tensorboard summary writer
	summary_writer = tf.summary.FileWriter(arg.tensorboard_path)
	summary_writer.add_graph(sess.graph)


	# Start Coordinator to feed in data from batch shuffler
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for epochs in range(arg.epoch):
		print('epochs : {}'.format(epochs),end=' || ')
		# 1. Compute g_result and f_result
		g_result, f_result, z = sess.run(feed_op)
		# print G result shape and F result shape
		print('g_shape:{} f_shape:{} z_shape:{}'.format(g_result.shape,f_result.shape,z.shape),end=' || ')
		# 2. Compute Loss
		loss = sess.run(loss_op)
		# 3. Optimize
		sess.run(optimize_op)

		## Save summary data every epoch
		temp_summary = sess.run(merged_summary)
		summary_writer.add_summary(temp_summary,epochs)

		# Print Loss in console
		print('loss:{}'.format(loss))

		if epochs % 10 == 0:
			saver.save(sess,arg.model_save_path,global_step=epochs)

	# stop coordinator and join threads
	coord.request_stop()
	coord.join(threads)
