from imutils.video import WebcamVideoStream, FPS
import imutils
import cv2
from classes.ImageProcessor import ImageProcessor

# Start the webcam feed, and start FPS counter
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

image_processor = ImageProcessor()  # Create an object of the ImageProcessor class

def slider_init():  # Set the value of the HSV sliders to the current values
	THRESH_LOW, THRESH_HIGH = image_processor.unpack_threshold_values()  # Unpack the thresholds
	h1, s1, v1 = THRESH_LOW  # Parse data into lower half
	h2, s2, v2 = THRESH_HIGH  # Parse data into upper half

	# Set values of lower sliders
	cv2.setTrackbarPos("H", "slider", h1)
	cv2.setTrackbarPos("S", "slider", s1)
	cv2.setTrackbarPos("V", "slider", v1)

	# Set values of higher sliders
	cv2.setTrackbarPos("H2", "slider", h2)
	cv2.setTrackbarPos("S2", "slider", s2)
	cv2.setTrackbarPos("V2", "slider", v2)

def slider_frame():  # Create sliders for HSV values
	cv2.namedWindow("slider")  # Create a window for sliders

	# Lower slider values
	cv2.createTrackbar("H", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("S", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("V", "slider", 0, 255, lambda *args: None)

	# Upper slider values
	cv2.createTrackbar("H2", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("S2", "slider", 0, 255, lambda *args: None)
	cv2.createTrackbar("V2", "slider", 0, 255, lambda *args: None)

	slider_init()

def main():

	slider_frame()

	while True:

		frame = vs.read()  # Fetch the frame from the camera

		# Fetch values from sliders
		h1 = cv2.getTrackbarPos('H', 'slider')
		s1 = cv2.getTrackbarPos('S', 'slider')
		v1 = cv2.getTrackbarPos('V', 'slider')

		h2 = cv2.getTrackbarPos('H2', 'slider')
		s2 = cv2.getTrackbarPos('S2', 'slider')
		v2 = cv2.getTrackbarPos('V2', 'slider')

		mask = image_processor.threshold(frame, (h1,s1,v1), (h2,s2,v2))  # Apply threshold

		# Display the frames
		cv2.imshow('Vision Tuner', frame)
		cv2.imshow('Vision Mask', mask)

		k = cv2.waitKey(5) & 0xFF

		if k == 27:  # If ESC is clicked, end the loop 
			break

		if k == ord('s'):  # If S is clicked, save the hsv values
			image_processor.save_threshold_values((h1,s1,v1),(h2,s2,v2))  # Save the values
			print("Saved current HSV config: low: ({},{},{}), high: ({},{},{})".format(h1, s1, v1, h2, s2, v2))

		fps.update()  # Update FPS

	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	cv2.destroyAllWindows()
	vs.stop()

if __name__ == '__main__':
	main()