import imutils
import cv2
import pickle
import math
import itertools
import argparse
import numpy as np
from networktables import NetworkTables
from imutils.video import WebcamVideoStream, FPS

# ------------------ FUNCTIONS ------------------

def midpoint(p1, p2):  # Basic function to calculate the midpoint of two points
	return (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))


def distance(p1, p2):  # Calculates the distance between two points
	return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def find_angle(p1, p2):  # Basic function to calculate the angle of two points
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


def load_threshold():  # Load pickle files for contour data
	# Access and fetch two seperate pickle files
	with open(PICKLE_PATH + "hsv_low.pickle", 'rb') as f:
		HSV_LOW = pickle.load(f)

	with open(PICKLE_PATH + "hsv_high.pickle", 'rb') as f:
		HSV_HIGH = pickle.load(f)

	return HSV_LOW, HSV_HIGH


def get_box_sides(cnt):
	box = cv2.minAreaRect(cnt)  # Find the box dimensions of contour
	box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # Adjust for rotation
	box = np.array(box, dtype="int")  # Create a numpy array of box dimensions

	rect = order_points(box)  # Return the two smallest point combinations from box dimensions
	side1 = rect[0]  # Split the multidimensional array returned from "order_points" into two variables
	side2 = rect[1]

	return side1, side2

# ------------------ INITIALIZE VARIABLES ------------------

ROBORIO_IP = "10.6.12.2"

PICKLE_PATH = "pickles/"  # Path of folders with pickles
AREA_LIMIT = 1000  # Limit to find contours with area bigger than limit
TOLERANCE = 15  # Angle tolerance to consider a "pair"

# Start the webcam feed, and start FPS counter
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2), anchor=(0,0))  # Define kernel

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
ap.add_argument("-t", "--table", type=str, required=True, help="Determine the name of the NetworkTable to push to")
args = vars(ap.parse_args())

NetworkTables.initialize(server=ROBORIO_IP)  # Initialize NetworkTable server
sd = NetworkTables.getTable(args["table"])  # Fetch the NetworkTable table

previous = None

# ------------------ MAIN LOOP ------------------

while True:

	frame = vs.read()  # Fetch the frame from the camera
	display = frame.copy() # Create a copy of frame for clear display and drawing

	height, width = frame.shape[:2]

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert into HSV readable

	hsv_low, hsv_high = load_threshold()  # Unpack the threshold values
	mask = cv2.inRange(hsv, hsv_low, hsv_high)  # Filter HSV range

	# Code that finds contours, filter out sizes, and sort by x position left to right
	cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours
	sorted_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= AREA_LIMIT]  # Filter out small contours
	sorted_cnts = sorted(sorted_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])[:4]  # Sort and isolate max of 4 contours

	for count, cnt in enumerate(sorted_cnts):  # Iterate through 4 countours

		if (count == 0):  # If its the first contour, just skip it
			previous = cnt  # Assign the previous with the current and skip to next iteration
			continue

		# Calculate the short sides of the tape
		side1c, side2c = get_box_sides(cnt)
		side1p, side2p = get_box_sides(previous)
		
		# Calculate the midpoints of the sides of the contour
		midpoint1c, midpoint2c = midpoint(side1c[0], side1c[1]), midpoint(side2c[0], side2c[1]) # Find the midpoint of each point combinations
		midpoint1p, midpoint2p = midpoint(side1p[0], side1p[1]), midpoint(side2p[0], side2p[1]) # Find the midpoint of each point combinations

		# Calculate the angle of the two midpoints
		angle_prev = find_angle(midpoint1p, midpoint2p)  # Calculate the midpoint
		angle = find_angle(midpoint1c, midpoint2c)
		
		# If the previous angle is negative and next angle positive
		if (angle_prev < 0 and angle > 0):
			# Check if angles are within tolerance to be pair
			if abs(abs(angle_prev) - abs(angle)) < 15:

				# Determine which side of tape is the uppermost
				current_box = side1c if side1c[0][1] < side2c[0][1] else side2c
				prev_box = side1p if side1p[0][1] < side2p[0][1] else side2p

				# Find the point closest to the other corner (upper corner) for each previous and current
				if (current_box[0][0] < current_box[1][0]):
					current_target = current_box[0]
				else:
					current_target = current_box[1]

				if (prev_box[0][0] > prev_box[1][0]):
					prev_target = prev_box[0]
				else:
					prev_target = prev_box[1]

				if args["display"] > 0:
					# Some drawing to display contour results
					cv2.circle(display, tuple(current_target), 5, (255,0,0), thickness=1)
					cv2.circle(display, tuple(prev_target), 5, (0,255,0), thickness=1)
					cv2.line(display, tuple(current_target), tuple(prev_target), (0,0,255), 4)
					cv2.line(display, (int(width/2), 0), (int(width/2), height), (0,0,0), 4)
					cv2.line(display, tuple(current_box[0]), tuple(current_box[1]), (255,0,255), 2)
					cv2.line(display, tuple(prev_box[0]), tuple(prev_box[1]), (255,255,0), 2)

				tape_target = midpoint(current_target, prev_target)
				offset = (width/2)+tape_target[0]
				print(offset)

		previous = cnt  # Re-assign the previous contour

	# Display the frames
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

		
