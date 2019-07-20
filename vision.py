import imutils
import cv2
import pickle
import math
import itertools
import argparse
import numpy as np
from networktables import NetworkTables
from imutils.video import WebcamVideoStream, FPS
from tuner import load_threshold


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


# Function courtesy of https://www.cs.hmc.edu/ACM/lectures/intersections.html
def intersectLines(ptA, ptB, ptC, ptD): # Returns intersection point of Line(ptA, ptB) and Line(ptD, ptC)

	DET_TOLERANCE = 0.00000001  # Define determent tolerance

	# the first line is pt1 + r*(pt2-pt1)
	# in component form:
	x1, y1 = ptA;   x2, y2 = ptB
	dx1 = x2 - x1;  dy1 = y2 - y1

	# the second line is ptA + s*(ptB-ptA)
	x, y = ptC;   xB, yB = ptD;
	dx = xB - x;  dy = yB - y;

	#    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
	#    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
	DET = (-dx1 * dy + dy1 * dx)  # Given matrix above solve for DET

	if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

	# Solve for the inverse of the determent
	DETinv = 1.0/DET

	# Find the scalar amount along the "self" segment
	r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

	# Find the scalar amount along the input line
	s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

	# Return the average of the two descriptions
	xi = (x1 + r*dx1 + x + s*dx)/2.0
	yi = (y1 + r*dy1 + y + s*dy)/2.0
	return (int(xi), int(yi))


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


def get_box_sides(cnt):
	box = cv2.minAreaRect(cnt)  # Find the box dimensions of contour
	box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # Adjust for rotation
	box = np.array(box, dtype="int")  # Create a numpy array of box dimensions

	rect = order_points(box)  # Return the two smallest point combinations from box dimensions
	side1 = rect[0]  # Split the multidimensional array returned from "order_points" into two variables
	side2 = rect[1]

	return side1, side2


def find_targets(current_box, prev_box):
	if (current_box[0][0] < current_box[1][0]):
		current_target = current_box[0]
	else:
		current_target = current_box[1]

	if (prev_box[0][0] > prev_box[1][0]):
		prev_target = prev_box[0]
	else:
		prev_target = prev_box[1]

	return tuple(current_target), tuple(prev_target)


def main():

	while True:

		frame = vs.read()  # Fetch the frame from the camera
		display = frame.copy() # Create a copy of frame for clear display and drawing

		height, width = frame.shape[:2]

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert into HSV readable

		hsv_low, hsv_high = load_threshold()  # Unpack the threshold values
		mask = cv2.inRange(hsv, hsv_low, hsv_high)  # Filter HSV range

		# Code that finds contours, filter out sizes, and sort by x position left to right
		image, cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours
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

					# Determine which side of tape is the uppermost and farthest to left
					if side1c[0][1] < side2c[0][1]:
						current_box_bottom = side2c
						current_box_top = side1c
					else:
						current_box_bottom = side1c
						current_box_top = side2c

					if side1p[0][1] > side2p[0][1]:
						prev_box_top = side2p
						prev_box_bottom = side1p
					else:
						prev_box_top = side1p
						prev_box_bottom = side2p

					# Determine the corner target points of each tape contour
					current_target_top, prev_target_top = find_targets(current_box_top, prev_box_top)
					current_target_bottom, prev_target_bottom = find_targets(current_box_bottom, prev_box_bottom)

					# Data to send to network tables length of X lines and offset from intersection point of X lines
					intersection = intersectLines(prev_target_bottom, current_target_top, current_target_bottom, prev_target_top)
					leftDistance = distance(prev_target_bottom, current_target_top)
					rightDistance = distance(current_target_bottom, prev_target_top)
					offset = intersection[0] - (width/2)  # Calculate the offset from the center of the intersection point

					# Pass the data through network tables
					sd.putNumber("offset", offset)  # Push data to table
					sd.putNumber("leftDistance", leftDistance)  # Push data to table
					sd.putNumber("rightDistance", rightDistance)  # Push data to table

					print("Offset: {}".format(offset))
					print("Left Tape Distance: {}".format(leftDistance))
					print("Right Tape Distance: {}".format(rightDistance))

					if args["display"] > 0:

						cv2.circle(display, prev_target_bottom, 5, (255,0,0), thickness=1)
						cv2.circle(display, current_target_top, 5, (0,255,0), thickness=1)
						cv2.circle(display, current_target_bottom, 5, (0,0,0), thickness=1)
						cv2.circle(display, prev_target_top, 5, (255,0,0), thickness=1)

						# Some drawing to display contour results
						cv2.circle(display, prev_target_bottom, 5, (0,255,0), thickness=1)
						cv2.circle(display, current_target_bottom, 5, (0,0,255), thickness=1)
						cv2.line(display, current_target_bottom, prev_target_bottom, (0,0,255), 4)

						cv2.circle(display, prev_target_top, 5, (0,255,0), thickness=1)
						cv2.circle(display, current_target_top, 5, (0,0,255), thickness=1)
						cv2.line(display, current_target_top, prev_target_top, (0,0,255), 4)

						cv2.line(display, prev_target_bottom, current_target_top, (255,255,255), 4)
						cv2.line(display, current_target_bottom, prev_target_top, (0,0,255), 4)

						cv2.line(display, (int(width/2), 0), (int(width/2), height), (0,0,0), 4)
						cv2.line(display, tuple(current_box_top[0]), tuple(current_box_top[1]), (255,0,255), 2)
						cv2.line(display, tuple(current_box_bottom[0]), tuple(current_box_bottom[1]), (255,255,255), 2)
						cv2.line(display, tuple(prev_box_top[0]), tuple(prev_box_top[1]), (255,255,255), 2)
						cv2.circle(display, intersection, 5, (255,255,255), thickness=1)

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


if __name__ == "__main__":
	main()
