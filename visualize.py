import tensorflow as tf 
import numpy as np 
import argparse
import os
from matplotlib import pyplot as plt

import dataloader
import models

# Settings
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
#parser.add_argument('-deterministic_path', type=str, default='./model/deterministic/deterministic_model-10.meta', help='deterministic model path')
parser.add_argument('-latent_path', type=str, default='./model/latent/latent_model-30.meta', help='latent model path')
arg = parser.parse_args()

### Setup Testing Environment ###
dataloader = dataloader.dataloader(arg)
videoInfo = dataloader.getVideoInfo()
print('original width: {0[0]} original height: {0[1]} number_of_Frame: {0[2]} FPS: {0[3]}'.format(videoInfo))
# if tfrecords doesn't exist make one
if not(os.path.isfile(arg.tfrecordspath)):
	dataloader.gen_tfrecords()
else:
	print('dataloader: {} exists'.format(arg.tfrecordspath))

sess = tf.Session('', tf.Graph())
with sess.graph.as_default():
    # Read meta graph and checkpoint to restore tf session
    saver = tf.train.import_meta_graph(arg.latent_path)
    saver.restore(sess, "./model/latent/latent_model-30")

    coord = tf.train.Coordinator()
    threads = []
    for qr in sess.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    while(True):
	    x_val,y_val,g_result, f_result = sess.run(['x_val:0','y_val:0','deterministic_feed_op:0','latent_feed_op:0'])
	    
	    ## Make Subplot
	    fig=plt.figure(figsize=(8, 8))
	    rows = 4
	    columns = 5
	    index = 1
	    # x_val
	    for i in range(columns):
	    	img = x_val[0,:,:,i*3:(i+1)*3]
	    	fig.add_subplot(rows, columns, index)
	    	plt.imshow(img)
	    	index = index + 1

	    for i in range(columns):
	    	img = y_val[0,:,:,i*3:(i+1)*3]
	    	fig.add_subplot(rows, columns, index)
	    	plt.imshow(img)
	    	index = index + 1

	    for i in range(columns):
	    	img = g_result[0,:,:,i*3:(i+1)*3]
	    	fig.add_subplot(rows, columns, index)
	    	plt.imshow(img)
	    	index = index + 1

	    for i in range(columns):
	    	img = f_result[0,:,:,i*3:(i+1)*3]
	    	fig.add_subplot(rows, columns, index)
	    	plt.imshow(img)
	    	index = index + 1

	    ## Show Image
	    plt.show()

