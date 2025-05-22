import cv2
import numpy as np

# Load two images of the same size
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Resize to same dimensions if needed
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Arithmetic operations
add = cv2.add(img1, img2)
subtract = cv2.subtract(img1, img2)
multiply = cv2.multiply(img1, img2)
divide = cv2.divide(img1, img2)

# Display results
cv2.imshow('Addition', add)
cv2.imshow('Subtraction', subtract)
cv2.imshow('Multiplication', multiply)
cv2.imshow('Division', divide)

cv2.waitKey(0)
cv2.destroyAllWindows()
