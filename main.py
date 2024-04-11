import cv2
import os

# Load an image from file
image_path = "fastx.jpg"
image = cv2.imread(image_path)  # Reads the image from the specified path

# Uncomment below to see the image in a window
# cv2.imshow('Sample Image', image)  # Displays the loaded image in a window

# Uncomment below to resize the image
# image_resized = cv2.resize(image, (500, 400))  # Resizes the image to 500x400 pixels

# Uncomment below to load the image in grayscale
# img = cv2.imread('fastx.jpg', 0)  # Load the image in grayscale (0)

# Uncomment below to load the image with alpha channel if exists
# img1 = cv2.imread('fastx.jpg', -1)  # Load the image including alpha channel (-1)

# Uncomment below to display grayscale and color images
# cv2.imshow('Grayscale Image', img)
# cv2.imshow('Color Image', img1)

# Video capture from webcam
# abc = cv2.VideoCapture(0)  # Starts video capture from the default camera

# Read frames from the webcam
# while True:
#     ret, frame = abc.read()  # Read a single frame from the webcam
#     cv2.imshow("Video Feed", frame)  # Display the captured frame
#     if cv2.waitKey(1) & 0xff == ord('q'):  # Exit loop if 'q' is pressed
#         break
# abc.release()  # Release the video capture object

# Clean up windows
# cv2.destroyAllWindows()

# Read a video file
video_path = 'Captured_video.mp4'
cap = cv2.VideoCapture(video_path)  # Open the video file

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Display each frame of the video
while True:
    ret, frame = cap.read()  # Read frame from video

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow('frame', frame)  # Display the frame

    # Break the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
