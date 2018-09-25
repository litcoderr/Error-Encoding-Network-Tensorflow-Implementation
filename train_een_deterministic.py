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

import dataloader
import models

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='baseline-3layer', help='type of model to use')
parser.add_argument('-width', type=int, default=480, help='video width')
parser.add_argument('-height', type=str, default=480, help='video height')
parser.add_argument('-pred_frame', type=int, default=5, help='number of frames to learn and predict')
parser.add_argument('-time_interval', type=int, default=2, help='time interval between frames in milliseconds')
parser.add_argument('-frame_interval', type=int, default=150, help='frame interval when generating datasets')
parser.add_argument('-batch_size', type=int, default=5, help='batch size')
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps in convnet')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch', type=int, default=500, help='number of epochs')
parser.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parser.add_argument('-videopath', type=str, default='./data/flower.mp4', help='video folder')
parser.add_argument('-tfrecordspath', type=str, default='./data/dataset.tfrecords', help='tfrecords file path')
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

# Variables
weights={
	'wc1' : tf.Variable(tf.random_normal([7,7,dataloader.channel,arg.nfeature])),
	'wc2' : tf.Variable(tf.random_normal([5,5,arg.nfeature,arg.nfeature])),
	'wc3' : tf.Variable(tf.random_normal([5,5,arg.nfeature,arg.nfeature])),
	'wc4' : tf.Variable(tf.random_normal([4,4,arg.nfeature,arg.nfeature])),
	'wc5' : tf.Variable(tf.random_normal([4,4,arg.nfeature,arg.nfeature])),
	'wc6' : tf.Variable(tf.random_normal([4,4,dataloader.channel,arg.nfeature]))
}
biases={
	'bc1' : tf.Variable(tf.random_normal([arg.nfeature])),
	'bc2' : tf.Variable(tf.random_normal([arg.nfeature])),
	'bc3' : tf.Variable(tf.random_normal([arg.nfeature])),
	'bc4' : tf.Variable(tf.random_normal([arg.nfeature])),
	'bc5' : tf.Variable(tf.random_normal([arg.nfeature])),
	'bc6' : tf.Variable(tf.random_normal([dataloader.channel]))
}

# Variables
init_global_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()

model = models.BaselineModel3Layer(x_train,weights,biases)
feed_op = model.feed()
# Train
with tf.Session() as sess:
	sess.run(init_global_op)
	sess.run(init_local_op)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	#Test
	result = sess.run(feed_op)

	print(result.shape)

	# stop coordinator and join threads
	coord.request_stop()
	coord.join(threads)







