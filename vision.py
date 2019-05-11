import imutils
import cv2
import pickle
import math
import itertools
import numpy as np
from imutils.video import WebcamVideoStream, FPS

def midpoint(p1, p2):  # Basic function to calculate the midpoint of two points
	return (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

def distance(p1, p2):  # Calculates the distance between two points
	return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def find_angle(cnt):  # Basic function to calculate the angle of two points

	p1, p2 = get_midpoints(cnt)

	x = p2[0] - p1[0]  # Find the x coordinate
	y = p2[1] - p1[1]  # Find the y coordinate
	try:
		angle = math.degrees(math.atan(y/x))  # Inverse tangent (opposite over adjacent)
	except ZeroDivisionError:  # If angle is 90 degrees inverse tangent is undefined
		angle = 90.0
	return angle  # Return the angle measure

def order_points(box):  # Recieves dimensions of bounding box, and sorts through points to find two smallest points
	combinations = list(itertools.combinations(box, 2))  # Find all the possible combinations of points given bounding box

	points = []  # Define a new array to store all point combinations after for loop
	for point in combinations:  # Loop through all combinations
		d = distance(point[0], point[1])  # Find the distance of each combination
		points.append([point, d])  # Append the distance and actual point to multidimensional array

	points.sort(key=lambda x: x[1])  # After loop is finished sort array by second element in sub array which is distance (small to large)
	points = points[:2] # Chop everything except for the first two elements in array off
	return [points[0][0], points[1][0]]  # Return a multi dimensional array with the two points

def load_threshold():
	# Access and fetch two seperate pickle files
	with open(PICKLE_PATH + "hsv_low.pickle", 'rb') as f:
		HSV_LOW = pickle.load(f)

	with open(PICKLE_PATH + "hsv_high.pickle", 'rb') as f:
		HSV_HIGH = pickle.load(f)

	return HSV_LOW, HSV_HIGH

def get_midpoints(cnt):
	box = cv2.minAreaRect(cnt)  # Find the box dimensions of contour
	box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # Adjust for rotation
	box = np.array(box, dtype="int")  # Create a numpy array of box dimensions

	rect = order_points(box)  # Return the two smallest point combinations from box dimensions
	side1 = rect[0]  # Split the multidimensional array returned from "order_points" into two variables
	side2 = rect[1]

	midpoint1 = midpoint(side1[0], side1[1])  # Find the midpoint of each point combinations
	midpoint2 = midpoint(side2[0], side2[1])

	return midpoint1, midpoint2


PICKLE_PATH = "pickles/"  # Path of folders with pickles
AREA_LIMIT = 1000  # Limit to find contours with area bigger than limit

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(0,0))  # Define kernel

# Start the webcam feed, and start FPS counter
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

previous = None

while True:

	frame = vs.read()  # Fetch the frame from the camera
	display = frame.copy() # Create a copy of frame for clear display and drawing

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert into HSV readable

	hsv_low, hsv_high = load_threshold()  # Unpack the threshold values
	mask = cv2.inRange(hsv, hsv_low, hsv_high)  # Filter HSV range

	# Code that finds contours, filter out sizes, and sort by x position left to right
	cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours
	sorted_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= AREA_LIMIT]  # Filter out small contours
	sorted_cnts = sorted(sorted_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])[:4]

	for count, cnt in enumerate(sorted_cnts):

		if (count == 0):
			previous = cnt
			continue

		angle_prev = find_angle(previous)
		angle = find_angle(cnt)

		# TODO: add logic to determine if tape pair

		previous = cnt

	cv2.imshow('Mask', mask)
	cv2.imshow('Frame', display)

	k = cv2.waitKey(5) & 0xFF

	if k == 27:  # If ESC is clicked, end the loop 
		break

	fps.update()  # Update FPS

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
		


