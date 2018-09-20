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
		self.width,self.height,self.nframe,self.fps = self.getVideoInfo()
		print('dataloader: done initializing')

	# Get Video Data
	def getVideoInfo(self):
		width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = self.cap.get(cv2.CAP_PROP_FPS)
		nframe = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		return (width,height,nframe,fps)

	# Loading Data from arg.videopath
	def loadData(self):
		self.cap = cv2.VideoCapture(self.arg.videopath)
		print('dataloader: video loaded')

	# Get frame of index in numpy array
	def getFrame(self,frame_index):
		if self.cap.isOpened():
			self.cap.set(1,frame_index)
			ret, frame = self.cap.read()
			return ret,frame
		else:
			print('dataloader: please load data first')
			return 0

	# Get data for training or prediction
	# Ouputs : [batch_size,nframe,in_w,in_h,3]
	#TODO --> finish this method
	def getData(self,in_w,in_h):
		return self.getFrame(300)

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