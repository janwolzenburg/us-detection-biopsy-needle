from needle_detection.preprocessing import read_frame, get_ROI
from needle_detection.kernels import filter_kernel_parameters, build_gauss_kernel, build_sobel_kernel, convolution, gabor_kernel, canny_edge_detection
from needle_detection.line_detection import line_detector, build_probe_lines
import matplotlib.pyplot as plt
import numpy as np
import needle_detection.parameters as p
from skimage.draw import disk, line
import cv2 as cv
angle = p.angles[4]
print(angle)

#frame = read_frame('../needle_detection/resources/test/77.png', 0,255)
#ROI, rescale_factor = get_ROI(frame, angle)
frame = cv.imread('../needle_detection/resources/test/77.png', cv.IMREAD_GRAYSCALE)
cropped_frame = frame[100:600, 300:1000]

gabor_image = gabor_kernel(cropped_frame, 0)
edges =  canny_edge_detection(gabor_image, sigma=1, low_threshold=1, high_threshold=50, mask=None)

dmy = edges.copy()
thresL = 50
minL = 200
maxL = 50
lines = cv.HoughLinesP(edges, 1, np.pi/360, threshold=thresL, maxLineGap=maxL, minLineLength=minL)
# draw Hough lines
for line in lines:
	x1, y1, x2, y2 = line[0]
	cv.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)

f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10,6))
ax1.imshow(cropped_frame, cmap='gray')
ax2.imshow(gabor_image, cmap='gray')
ax3.imshow(edges, cmap='gray')
ax4.imshow(dmy, cmap='gray')

plt.show()
plt.close()