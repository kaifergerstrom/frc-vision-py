from imutils.video import WebcamVideoStream, FPS
import cv2
import pickle


PICKLE_PATH = "pickles/"  # Path of folders with pickles


def save_threshold_values(thresh_low, thresh_high):  # Function that will update the pickle values for the HSV thresholds
		
		# Create and update two seperate pickle files
		with open(PICKLE_PATH + "hsv_low.pickle", 'wb') as f:
			pickle.dump(thresh_low, f)
			THRESH_LOW = thresh_low

		with open(PICKLE_PATH + "hsv_high.pickle", 'wb') as f:
			pickle.dump(thresh_high, f)
			THRESH_HIGH = thresh_high


def load_threshold():  # Load pickle files for contour data
	# Access and fetch two seperate pickle files
	with open(PICKLE_PATH + "hsv_low.pickle", 'rb') as f:
		HSV_LOW = pickle.load(f)

	with open(PICKLE_PATH + "hsv_high.pickle", 'rb') as f:
		HSV_HIGH = pickle.load(f)

	return HSV_LOW, HSV_HIGH


def create_hsv_slider():  # Create sliders for HSV values

	cv2.namedWindow("slider")  # Create a window for sliders

	# Lower slider values
	cv2.createTrackbar("H", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("S", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("V", "slider", 0, 255, lambda *args: None)

	# Upper slider values
	cv2.createTrackbar("H2", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("S2", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("V2", "slider", 0, 255, lambda *args: None)

	hsv_low, hsv_high = load_threshold()  # Load the current HSV thresholds

	# Set values of lower sliders
	cv2.setTrackbarPos("H", "slider", hsv_low[0])
	cv2.setTrackbarPos("S", "slider", hsv_low[1])
	cv2.setTrackbarPos("V", "slider", hsv_low[2])

	# Set values of higher sliders
	cv2.setTrackbarPos("H2", "slider", hsv_high[0])
	cv2.setTrackbarPos("S2", "slider", hsv_high[1])
	cv2.setTrackbarPos("V2", "slider", hsv_high[2])


def main():

	# Start the webcam feed, and start FPS counter
	vs = WebcamVideoStream(src=0).start()
	fps = FPS().start()

	create_hsv_slider()

	while True:

		frame = vs.read()  # Fetch the frame from the camera
		display = frame.copy() # Create a copy of frame for clear display and drawing

		# Fetch values from sliders
		h1 = cv2.getTrackbarPos('H', 'slider')
		s1 = cv2.getTrackbarPos('S', 'slider')
		v1 = cv2.getTrackbarPos('V', 'slider')

		h2 = cv2.getTrackbarPos('H2', 'slider')
		s2 = cv2.getTrackbarPos('S2', 'slider')
		v2 = cv2.getTrackbarPos('V2', 'slider')

		# Display the frames
		#cv2.imshow('Mask', mask)
		cv2.imshow('Frame', display)

		k = cv2.waitKey(5) & 0xFF

		if k == 27:  # If ESC is clicked, end the loop 
			break

		if k == ord('s'):  # If S is clicked, save the hsv values
			save_threshold_values((h1,s1,v1),(h2,s2,v2))  # Save the values
			print("Saved current HSV config: low: ({},{},{}), high: ({},{},{})".format(h1, s1, v1, h2, s2, v2))

		fps.update()  # Update FPS

	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()
	vs.stop()


if __name__ == "__main__":
	main()