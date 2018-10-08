'''
Version 1.0 dataloader

[main functionality]
-load video data
-feeding data

Developed By James Youngchae Chee @Litcoderr
You are welcome to contribute
'''

import cv2
import tensorflow as tf
import numpy as np

class dataloader:
	def __init__(self,arg):
		self.arg = arg
		self.loadData()

		# original width and height(Not used in operation --> converted to different size)
		# nframe : Number of frames
		# fps: Frames per Seconds
		self.original_width,self.original_height,self.nframe,self.fps = self.getVideoInfo()
		# Number of Channel for each dataset
		self.channel = 3 * self.arg.pred_frame
		# Number of frames between each frame in a dataset
		self.frame_interval = int((self.arg.time_interval/10)*self.fps)
		print('dataloader: done initializing')

	# Loading Data from arg.videopath --> Video File Path
	def loadData(self):
		self.cap = cv2.VideoCapture(self.arg.videopath)
		print('dataloader: video loaded')

	# Get Video Meta Data (Basic Information)
	def getVideoInfo(self):
		width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = self.cap.get(cv2.CAP_PROP_FPS)
		nframe = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		return (width,height,nframe,fps)

	# Returns one frame's numpy array based on "frame_index"
	def getFrame(self,frame_index):
		# If video data is opened(loaded)
		if self.cap.isOpened():
			# Read Frame
			self.cap.set(1,frame_index)
			ret, frame = self.cap.read()
			# If video is not correctly oriented transpose to right shape
			if self.original_height < self.original_width:
				frame = np.transpose(frame,(1,0,2))
			# Resize to wanted shape (which will be used in training)
			# Want smaller size if computing power is lacking (Me...sad)
			frame = cv2.resize(frame, (self.arg.width, self.arg.height))
			return ret,frame
		else:
			# If Video is not loaded, load data first
			print('dataloader: please load data first')
			return 0

	# Returns One Data([height,width,num_channel])
	# --> will be stored in one big dataset
	def gen_Data(self,start_frame_index):
		# Numpy array to store X data , Y data
		temp_x = np.array([])
		temp_y = np.array([])

		# End index of X data , Y data
		endof_x = start_frame_index+self.frame_interval*(self.arg.pred_frame-1)
		endof_y = start_frame_index+self.frame_interval*(2*self.arg.pred_frame-1)
		
		# Index of each frame used to getFrame
		j = start_frame_index

		# Get Frame for training input (X data)
		while j <= endof_x:
			# Get Frame of index j
			_, frame = self.getFrame(j)
			# Preprocess image for it to range between 0 and 1
			frame = frame / 255
			# If start of a data --> append to empty temp_x
			if j==start_frame_index:
				temp_x = frame
			# If not start of a data --> append to existing temp_x
			else:
				temp_x = np.concatenate((temp_x,frame),2)
			# Update index of a frame based on calculated frame_interval
			j = j+self.frame_interval

		#Get Frame for training output (Y data)
		while j <= endof_y:
			# Get Frame of index j
			_, frame = self.getFrame(j)
			# Preprocess image for it to range between 0 and 1
			frame = frame / 255
			# If start of a data --> append to empty temp_y
			if j == endof_x+self.frame_interval:
				temp_y = frame
			# If not start of a data --> append to existing temp_y
			else:
				temp_y = np.concatenate((temp_y,frame),2)
			# Update index of a frame based on calculated frame_interval
			j = j+self.frame_interval
		# Return X data and Y data
		return (temp_x,temp_y)

	# Returns end of y data (target) index
	def endof_y(self,start_frame_index):
		return start_frame_index+self.frame_interval*(2*self.arg.pred_frame-1)

	# Generate TFRecord file for training
	def gen_tfrecords(self):
		print('dataloader: Generating TFRecords file-->{}'.format(self.arg.tfrecordspath))

		# Create tfrecord writer with destination file name
		filename = self.arg.tfrecordspath
		writer = tf.python_io.TFRecordWriter(filename)

		# index : starting frame index
		index = 0
		while self.endof_y(index) < self.nframe:
			# Get data for X and Y
			temp_x , temp_y = self.gen_Data(index)

			# Get dimensions of X and Y data
			height_x,width_x,channel_x = temp_x.shape
			height_y,width_y,channel_y = temp_y.shape

			# Cast to float32 (Optimal for Tensorflow)
			temp_x = np.float32(temp_x)
			temp_y = np.float32(temp_y)

			# Convert numpy array to raw string
			raw_x = temp_x.tostring()
			raw_y = temp_y.tostring()

			# Define a tensorflow train example
			example = tf.train.Example(features=tf.train.Features(feature={
				'height_x' : self._int64_feature(height_x),
				'width_x' : self._int64_feature(width_x),
				'channel_x' : self._int64_feature(channel_x),
				'raw_x' : self._bytes_feature(raw_x),
				'height_y' : self._int64_feature(height_y),
				'width_y' : self._int64_feature(width_y),
				'channel_y' : self._int64_feature(channel_y),
				'raw_y' : self._bytes_feature(raw_y)
				}))
			# Write example to tfrecords file
			writer.write(example.SerializeToString())
			# increase index to fetch next dataset
			index = index + self.arg.data_interval
		# close writer when done using
		writer.close()

	# Returns byte list to tf.train.Feature
	def _bytes_feature(self,value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	# Returns int64 list to tf.train.Feature
	def _int64_feature(self,value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	# decode tfrecords data and return numpy array data
	def decode(self,file_name_queue):
		# Create TFRecord Reader
		reader = tf.TFRecordReader()
		# Read an Example from file_name_queue
		_, example = reader.read(file_name_queue)
		# Parse Example
		features = tf.parse_single_example(example,features={
			'height_x' : tf.FixedLenFeature([], tf.int64),
			'width_x' : tf.FixedLenFeature([], tf.int64),
			'channel_x' : tf.FixedLenFeature([], tf.int64),
			'raw_x' : tf.FixedLenFeature([], tf.string),
			'height_y' : tf.FixedLenFeature([], tf.int64),
			'width_y' : tf.FixedLenFeature([], tf.int64),
			'channel_y' : tf.FixedLenFeature([], tf.int64),
			'raw_y' : tf.FixedLenFeature([], tf.string)
			})
		# Extract Feature
		X = tf.decode_raw(features['raw_x'],tf.float32)
		Y = tf.decode_raw(features['raw_y'],tf.float32)
		height_x = tf.cast(features['height_x'],tf.int32)
		width_x = tf.cast(features['width_x'],tf.int32)
		channel_x = tf.cast(features['channel_x'],tf.int32)
		height_y = tf.cast(features['height_y'],tf.int32)
		width_y = tf.cast(features['width_y'],tf.int32)
		channel_y = tf.cast(features['channel_y'],tf.int32)

		## Remake image
		# Define x and y data shape
		x_shape = tf.stack([height_x,width_x,channel_x])
		y_shape = tf.stack([height_y,width_y,channel_y])
		# Reshape X and Y data to wanted shape
		X = tf.reshape(X,x_shape)
		Y = tf.reshape(Y,y_shape)
		# Setting tensor's shape (Weird Tensorflow Stuff)
		X.set_shape([self.arg.height,self.arg.width,self.channel])
		Y.set_shape([self.arg.height,self.arg.width,self.channel])

		# Generate shuffled batch data (with wanted batch_size)
		X,Y = tf.train.shuffle_batch([X,Y],
			batch_size = self.arg.batch_size,
			capacity = 30,
			num_threads=2,
			min_after_dequeue=10)
		# Return final X and Y data
		# Shape: [batch_size,height,width,num_channel] (for each X and Y)
		return X,Y

	# Show frame based on frame_index
	def showFrame(self,frame_index):
		# Get Frame as Numpy array
		ret,frame = self.getFrame(frame_index)
		# If retrieved successfully
		if ret == True:
			# Show image using CV2 Library (CV2 Stuff)
			cv2.imshow('showFrame',frame)
			cv2.waitKey(0)
			cv2.destroyWindow('showFrame')

	# Play video (Slow --> because of fetching and resizing process)
	def playVideo(self):
		# Check if data is loaded
		if self.cap.isOpened():
			print('dataloader: playing video')
			# For Every Frame
			for i in range(self.nframe):
				# Get each frame's numpy array (resized)
				ret,frame = self.getFrame(i)
				# If retrieved Successfully
				if ret == True:
					# Show frame using CV2 Library
					cv2.imshow('playFrame',frame)
					# If any key is pressed --> break
					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
				else:
					break
			# Destroy Window
			cv2.destroyWindow('playFrame')
		else:
			print('dataloader: please load data first')
			