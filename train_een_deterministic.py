'''
Version 1.0 train_een_deterministic

[main functionality]
-used when training een deterministic model

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
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch', type=int, default=500, help='number of epochs')
parser.add_argument('-videopath', type=str, default='./data/flower.mp4', help='video folder')
parser.add_argument('-tfrecordspath', type=str, default='./data/dataset.tfrecords', help='tfrecords file path')
parser.add_argument('-tensorboard_path', type=str, default='./results/tensorboard/test1/deterministic', help='where tensorboard data is stored')
parser.add_argument('-model_save_path', type=str, default='./model/deterministic/deterministic_model', help='deterministic model path')
arg = parser.parse_args()

### Setup Training Environment ###
# Initialize dataloader
dataloader = dataloader.dataloader(arg)
videoInfo = dataloader.getVideoInfo()
print('original width: {0[0]} original height: {0[1]} number_of_Frame: {0[2]} FPS: {0[3]}'.format(videoInfo))
# if tfrecords doesn't exist --> make one
if not(os.path.isfile(arg.tfrecordspath)):
	dataloader.gen_tfrecords()
else:
	print('dataloader: {} exists'.format(arg.tfrecordspath))

## Variables ##

# Define weights and biases of Deterministic Model
weights={
	'wc1' : tf.get_variable("W1", shape=[7,7,dataloader.channel,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc2' : tf.get_variable("W2", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc3' : tf.get_variable("W3", shape=[5,5,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc4' : tf.get_variable("W4", shape=[4,4,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc5' : tf.get_variable("W5", shape=[4,4,arg.nfeature,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'wc6' : tf.get_variable("W6", shape=[4,4,dataloader.channel,arg.nfeature],initializer=tf.contrib.layers.xavier_initializer())
}
biases={
	'bc1' : tf.get_variable("B1", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc2' : tf.get_variable("B2", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc3' : tf.get_variable("B3", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc4' : tf.get_variable("B4", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc5' : tf.get_variable("B5", shape=[arg.nfeature],initializer=tf.contrib.layers.xavier_initializer()),
	'bc6' : tf.get_variable("B6", shape=[dataloader.channel],initializer=tf.contrib.layers.xavier_initializer())
}
# Store Weights and Biases as tf.summary
for i in range(1,7):
	tf.summary.histogram('wc{}'.format(i),weights['wc{}'.format(i)])
	tf.summary.histogram('bc{}'.format(i),biases['bc{}'.format(i)])

## Operations ##

# Make tfrecord filename queue
file_name_queue = tf.train.string_input_producer([arg.tfrecordspath])
# Decode tfrecord file to usable numpy array
x_train , y_train = dataloader.decode(file_name_queue)
# Create Deterministic Model
model = models.BaselineModel3Layer(x_train,weights,biases)
# Feeding Operation
feed_op = model.feed()
## Feeding Operation to store
feed_op_save = tf.identity(feed_op,'deterministic_feed_op')

# Define MSE Loss
loss = tf.losses.mean_squared_error(
	labels=y_train,
	predictions=feed_op
)
tf.summary.scalar('loss',loss)
# Train Operation
train_op = tf.train.AdamOptimizer(arg.lrt).minimize(loss)

# Initialization Operation
init_global_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

# Start Session
with tf.Session() as sess:
	# Initialize Variables
	sess.run(init_global_op)
	sess.run(init_local_op)

	# Merge all summary
	merged_summary = tf.summary.merge_all()

	# Saver Object to save all the variables
	saver = tf.train.Saver()
	# Set up Tensorboard summary writer
	summary_writer = tf.summary.FileWriter(arg.tensorboard_path)
	summary_writer.add_graph(sess.graph)

	# Start Coordinator and thread to feed in data
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	# Strat Training
	for epochs in range(arg.epoch):
		# Run Training Operation
		sess.run(train_op)
		## Save summary data every epoch
		temp_summary = sess.run(merged_summary)
		summary_writer.add_summary(temp_summary,epochs)
		# Print Loss Operation
		print('epochs: {} loss: {}'.format(epochs,loss.eval()))
		## Save weight every 10 epochs
		if epochs % 10 == 0:
			saver.save(sess,arg.model_save_path,global_step=epochs)


	# stop coordinator and join threads
	coord.request_stop()
	coord.join(threads)
