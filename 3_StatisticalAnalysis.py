import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('image2.jpg', 0)

# Calculate statistics
mean = np.mean(image)
median = np.median(image)
mode = np.argmax(np.bincount(image.ravel()))
std_dev = np.std(image)

# Print statistics
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")

# Plot histogram
plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
