'''
Version 1.0 dataloader

[main functionality]
-load video data
-feeding data

Developed By James Youngchae Chee @Litcoderr
You are welcome to contribute
'''

import cv2
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
			frame = cv2.resize(frame, (self.arg.width, self.arg.height))
			return ret,frame
		else:
			print('dataloader: please load data first')
			return 0

	def endof_y(self,first_frame_index):
		return first_frame_index+self.frame_interval*(2*self.arg.pred_frame-1)

	# Manufacture and return data to make a Trainable Dataset
	def makeDataset(self):
		dataset=np.array([]) # Empty dataset to be filled
		first_frame_index = 0
		while self.endof_y(first_frame_index) < self.nframe:
			temp_x = np.array([])
			temp_y = np.array([])
			endof_x = first_frame_index+self.frame_interval*(self.arg.pred_frame-1)
			endof_y = first_frame_index+self.frame_interval*(2*self.arg.pred_frame-1)
			j = first_frame_index

			# Get Frame for training input
			while j <= endof_x:
				ret, frame = self.getFrame(j)
				if j==first_frame_index:
					temp_x = frame
				else:
					temp_x = np.concatenate((temp_x,frame),2)
				j = j+self.frame_interval

			#Get Frame for training output
			while j <= endof_y:
				ret, frame = self.getFrame(j)
				if j == endof_x+self.frame_interval:
					temp_y = frame
				else:
					temp_y = np.concatenate((temp_y,frame),2)
				j = j+self.frame_interval

			temp = np.expand_dims(np.stack((temp_x,temp_y)),axis=0)
			# Construct a full dataset
			if first_frame_index == 0:
				dataset = temp
			else:
				dataset = np.concatenate((dataset,temp),0)
			
			print(dataset.shape)
			first_frame_index = first_frame_index+300
		return dataset

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