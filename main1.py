import numpy as np
import cv2

img_file = 'fastx.jpg'
# checking the pixel values of images
px = img[100,100]
print(px) # helps to get pixel output of the given image

blue= img[100,100,0]
print(blue)  # helps to get blue pixel output of the given image
img[100,100] = [255,255,255]
print(img[100,100])

# Accessing the image properties

img_file = 'fastx.jpg'
img = cv2.imread(img_file, cv2.IMREAD_COLOR)  # reading RGB pixel of the image
alpha_img= cv2.imread(img_file, cv2.IMREAD_UNCHANGED) # reads the alphaRGB(ARGB) of image
grey_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) # reads the grey scale pixel present in image
print('RGB Shape', img.shape) # prints the RGB pixel of the image
print('ARGB Shape', alpha_img.shape) # prints the  alphaRGB(ARGB) of image
print('Gray scale Shape', grey_img.shape) # prints the grey scale pixel present in image

# data type
print('image datatype' , img.dtype) # prints the datatype of image

# size
print('image size' , img.size) # prints the size of  pixel present in (row & column) of the image

# Setting the Region of image

img_file= 'fastx.jpg'
img_raw= cv2.imread(img_file)
roi = cv2.selectROI(img_raw)
print(roi)

# cropping selected ROI from the raw(given) image
#
roi_cropped= img_raw[int(roi[1]): int(roi[1]+roi[3]),int(roi[0]): int(roi[0]+roi[2])]
cv2.imshow("ROI image", roi_cropped)  # after selecting the image press enter
cv2.imwrite("Cropped.jpeg", roi_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# splitting and merging images
img_file= 'fastx.jpg'
image = cv2.imread(img_file)
g,b,r=cv2.split(image)
cv2.imshow("Green Part of image", g)
cv2.imshow("Green Part of image", b)
cv2.imshow("Green Part of image", r)
img1= cv2.merge((g,b,r))
cv2.imshow("Image after merging", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()# # changing the color of the images
img_file = 'fastx.jpg'
img=cv2.imread(img_file)
color_change= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
cv2.imshow("changed color scheme image", color_change)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Blend two different images
img_path = "fastx.jpg"
img1_path = "west.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img1= cv2.imread(img1_path, cv2.IMREAD_COLOR)
imge = cv2.resize(img, (800,400))
imge1 = cv2.resize(img1, (800,400))
blended_img = cv2.addWeighted(imge, 0.5, imge1,1,1)
cv2.imshow("Blended image", blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Different filters on image

img1 = cv2.imread("fastx.jpg")
img = cv2.resize(img1, (800,400))
# here we can play with different values by changing the values in rows
k_sharped= np.array([[-1,-1,-1],
                    [-1,20,-1],
                    [-1,-1,-10]])
sharpened= cv2.filter2D(img, -1, k_sharped)
cv2.imshow("Original image", img)
cv2.imshow("filtered image", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
# image thresholding
img1 = cv2.imread("fastx.jpg" , cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img1, (800,400))
ret,thresh= cv2.threshold(img, 127,255,cv2.THRESH_BINARY)
cannyimage= cv2.Canny(img, 50,100)
cv2.imshow("Original image", img)
cv2.imshow("filtered image", thresh)
cv2.imshow("canny image", cannyimage)
cv2.waitKey(0)
cv2.destroyAllWindows()


# countour Detection and shape detection

import matplotlib.pyplot as plt
img = cv2.imread('fastx.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# setting threshold of the gray scale image

_, threshold= cv2.threshold(gray, 127,255,cv2.THRESH_BINARY)
# contours using find counters function
contours, _ =cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for contour in contours:
    if i==0:
        i=1
        continue
    appox = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0,(255, 0 , 255),5)

# finding the center of the different shapes
M= cv2.moments(contour)
if M['m00'] != 0.0:
    x = int(M['m10']/ M['m00'])
    y = int(M['m10']/M['m00'])

# i want to put names of the shapes inside the corresponding shapes

if len(appox) ==3:
    cv2.putText(img, 'Triangle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
elif len(appox) ==4:
    cv2.putText(img, 'Quadrilateral', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
elif len(appox) ==5:
    cv2.putText(img, 'Pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
elif len(appox) ==6:
    cv2.putText(img, 'Hexagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
else:
    cv2.putText(img, 'Circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Color detection in OPENCV
img = cv2.imread('fastx.jpg')

# HSV, hue saturatiom and Value. HSV is commonly used in color and paint softwares

hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0,50,50])
upper_blue= np.array([130, 255, 255])

lower_green = np.array([40,40,40])
upper_green= np.array([70, 255, 255])

# threshold the HSV image to get only blue color

mask_blue = cv2.inRange(hsv , lower_blue, upper_blue)
res= cv2.bitwise_and(img, img, mask=mask_blue)
cv2.imshow('res', res)
# threshold the HSV image to get only green color

mask_green = cv2.inRange(hsv , lower_green, upper_green)
res= cv2.bitwise_and(img, img, mask=mask_green)
cv2.imshow('res', res)

lower_white = np.array([40,40,40])
upper_white= np.array([70, 255, 255])

# threshold the HSV image to get only blue color

mask_white = cv2.inRange(hsv , lower_white, upper_white)
res= cv2.bitwise_and(img, img, mask=mask_white)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Object Replacing in 2D Image using OpenCV
img = cv2.imread('fastx.jpg', cv2.IMREAD_COLOR)
img1= img.copy()
mask=np.zeros((100, 300, 3))
print(mask.shape)

pos= (200,200)
var= img1[200:(200+mask.shape[0]), 200:(200+mask.shape[1])]= mask
cv2.imshow("Coloring", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()