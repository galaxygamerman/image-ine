import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
images = [
		("Image 1", cv2.imread('image1.jpg',0)),
		("Image 2", cv2.imread('image2.jpg',0))
]

figure,axis = plt.subplots(nrows=1,ncols=len(images),figsize=(9,4))
figure.suptitle('Histogram of Pixel Intensities')

for i,(title,image) in enumerate(images):
	# Calculate statistics
	mean = np.mean(image)
	median = np.median(image)
	mode = np.argmax(np.bincount(image.ravel()))
	std_dev = np.std(image)

	# Print statistics
	print(f"Statistics for {title}:")
	print(f"Mean: {mean}")
	print(f"Median: {median}")
	print(f"Mode: {mode}")
	print(f"Standard Deviation: {std_dev}")

	# cv2.imshow(title, image)
	# Plot histogram
	subplot = axis[i]
	subplot.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
	subplot.set_title(f'Histogram of Pixel Intensities for {title}')
	subplot.set_xlabel('Pixel Intensity')
	subplot.set_ylabel('Frequency')

plt.show()
