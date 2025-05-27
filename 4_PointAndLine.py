import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# ============================
# POINT DETECTION (Laplacian)
# ============================
laplacian = cv2.Laplacian(image, cv2.CV_64F)
point_detected = cv2.convertScaleAbs(laplacian)

# ============================
# LINE DETECTION (Hough Transform)
# ============================

# Step 1: Edge detection using Canny
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Step 2: Detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=475)

# Convert grayscale to BGR to draw colored lines
line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw detected lines
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ============================
# DISPLAY RESULTS
# ============================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Grayscale')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Point Detection (Laplacian)')
plt.imshow(point_detected, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Line Detection (Hough Transform)')
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
