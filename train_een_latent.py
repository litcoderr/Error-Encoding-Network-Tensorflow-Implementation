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

with tf.Session() as sess:
	saver = tf.train.import_meta_graph(arg.model_path)
	saver.restore(sess,tf.train.latest_checkpoint('./model/deterministic/'))
	graph = tf.get_default_graph()
	wc1 = graph.get_tensor_by_name('W1:0')
	print(wc1.eval())






