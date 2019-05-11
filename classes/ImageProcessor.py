import imutils
import cv2
import pickle
import numpy as np

class ImageProcessor:

	THRESHOLD_PATH = "pickles/hsv"

	def __init__(self):
		self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(0,0))
		self.unpack_threshold_values()


	def threshold(self, img, thresh_low=None, thresh_high=None):  # Apply thresholding to image
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert into HSV readable

		if (thresh_low and thresh_high):
			mask = cv2.inRange(hsv, thresh_low, thresh_high)  # Filter HSV range
		else:
			mask = cv2.inRange(hsv, self.THRESH_LOW, self.THRESH_HIGH)  # Filter HSV range

		#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)  # Apply noise reduction
		return mask  # Return mask


	def save_threshold_values(self, thresh_low, thresh_high):  # Function that will update the pickle values for the HSV thresholds
		
		# Create and update two seperate pickle files
		with open(self.THRESHOLD_PATH + "_low.pickle", 'wb') as f:
			pickle.dump(thresh_low, f)
			self.THRESH_LOW = thresh_low

		with open(self.THRESHOLD_PATH + "_high.pickle", 'wb') as f:
			pickle.dump(thresh_high, f)
			self.THRESH_HIGH = thresh_high


	def unpack_threshold_values(self):  # Function to unpickle the two threshold variables

		# Access and fetch two seperate pickle files
		with open(self.THRESHOLD_PATH + "_low.pickle", 'rb') as f:
			self.THRESH_LOW = pickle.load(f)

		with open(self.THRESHOLD_PATH + "_high.pickle", 'rb') as f:
			self.THRESH_HIGH = pickle.load(f)

		return self.THRESH_LOW, self.THRESH_HIGH


	def find_contours(self, img):

		mask = self.threshold(img)

		contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(contours, key=cv2.contourArea, reverse=True)