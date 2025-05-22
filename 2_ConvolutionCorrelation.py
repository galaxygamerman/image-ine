import cv2
import numpy as np
from scipy.signal import convolve2d, correlate2d
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('image1.jpg', 0)

# Define a kernel/filter
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Convolution
convolved = convolve2d(image, kernel, mode='same', boundary='symm')

# Correlation
correlated = correlate2d(image, kernel, mode='same', boundary='symm')

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Convolution')
plt.imshow(convolved, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Correlation')
plt.imshow(correlated, cmap='gray')

plt.tight_layout()
plt.show()
