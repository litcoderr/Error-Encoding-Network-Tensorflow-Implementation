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

		# Needed Variables
		self.original_width,self.original_height,self.nframe,self.fps = self.getVideoInfo()
		# Wanted frame interval based on wanted time_interval
		self.frame_interval = int((self.arg.time_interval/10)*self.fps)
		print('dataloader: done initializing')

	# Loading Data from arg.videopath
	def loadData(self):
		self.cap = cv2.VideoCapture(self.arg.videopath)
		print('dataloader: video loaded')

	# Get Video Data
	def getVideoInfo(self):
		width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = self.cap.get(cv2.CAP_PROP_FPS)
		nframe = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		return (width,height,nframe,fps)

	# Get frame of index in numpy array
	def getFrame(self,frame_index):
		if self.cap.isOpened():
			self.cap.set(1,frame_index)
			ret, frame = self.cap.read()
			if self.original_height < self.original_width:
				frame = np.transpose(frame,(1,0,2))
			frame = cv2.resize(frame, (self.arg.width, self.arg.height))
			return ret,frame
		else:
			print('dataloader: please load data first')
			return 0

	# Manufacture and return data to make a Trainable Dataset
	def gen_Data(self,start_frame_index):
		temp_x = np.array([])
		temp_y = np.array([])
		endof_x = start_frame_index+self.frame_interval*(self.arg.pred_frame-1)
		endof_y = start_frame_index+self.frame_interval*(2*self.arg.pred_frame-1)
		j = start_frame_index

		# Get Frame for training input
		while j <= endof_x:
			ret, frame = self.getFrame(j)
			frame = frame / 255 # preprocess image to range 0 and 1
			if j==start_frame_index:
				temp_x = frame
			else:
				temp_x = np.concatenate((temp_x,frame),2)
			j = j+self.frame_interval

		#Get Frame for training output
		while j <= endof_y:
			ret, frame = self.getFrame(j)
			frame = frame / 255 # preprocess image to range 0 and 1
			if j == endof_x+self.frame_interval:
				temp_y = frame
			else:
				temp_y = np.concatenate((temp_y,frame),2)
			j = j+self.frame_interval
		
		return (temp_x,temp_y)

	# End index of y(target) data
	def endof_y(self,start_frame_index):
		return start_frame_index+self.frame_interval*(2*self.arg.pred_frame-1)

	# Generate TFRecord file for training
	def gen_tfrecords(self):
		print('dataloader: Generating TFRecords file...')

		filename = self.arg.tfrecordspath # tfrecords filename
		writer = tf.python_io.TFRecordWriter(filename)

		index = 0 # starting frame index
		while self.endof_y(index) < self.nframe:
			temp_x , temp_y = self.gen_Data(index)

			height_x,width_x,channel_x = temp_x.shape
			height_y,width_y,channel_y = temp_y.shape

			raw_x = temp_x.tostring()
			raw_y = temp_y.tostring()

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

			index = index + self.arg.frame_interval
		# close writer when done using
		writer.close()

	# make byte list to tf.train.Feature
	def _bytes_feature(self,value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	# make int64 list to tf.train.Feature
	def _int64_feature(self,value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	# Show frame
	def showFrame(self,frame_index):
		ret,frame = self.getFrame(frame_index)
		if ret == True:
			cv2.imshow('showFrame',frame)
			cv2.waitKey(0)
			cv2.destroyWindow('showFrame')

	# Play video
	def playVideo(self):
		# check if data is loaded
		if self.cap.isOpened():
			print('dataloader: playing video')
			for i in range(self.nframe):
				ret,frame = self.getFrame(i)
				if ret == True:
					cv2.imshow('playFrame',frame)
					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
				else:
					break
			cv2.destroyWindow('playFrame')
		else:
			print('dataloader: please load data first')

	# Print Video Data
	def printVideoData(self):
		print('video data: {}'.format(self.getVideoData()))