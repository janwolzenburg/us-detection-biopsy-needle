from needle_detection.preprocessing import read_frame, get_ROI, mask_roi
from needle_detection.kernels import filter_kernel_parameters, build_gauss_kernel, build_sobel_kernel, convolution, gabor_kernel, canny_edge_detection
from needle_detection.line_detection import line_detector, build_probe_lines, find_houghlinesP#, average_houghlines
import matplotlib.pyplot as plt
import numpy as np
import needle_detection.parameters as p
from skimage.draw import disk, line
import cv2 as cv

#frame = read_frame('../needle_detection/resources/test/77.png', 0,255)
#ROI, rescale_factor = get_ROI(frame, angle)
frame = cv.imread('../needle_detection/resources/test/77.png', cv.IMREAD_GRAYSCALE)
masked_frame = mask_roi(frame, p.vertices)
plt.imshow(masked_frame)
gabor_image = gabor_kernel(masked_frame, 0)
edges =  canny_edge_detection(gabor_image, sigma=30, low_threshold=1, high_threshold=50, mask=None)
cdst = edges.copy()
dmy = masked_frame.copy()

    
linesP = find_houghlinesP(edges)
#lines= find_houghlines(edges)
if linesP is not None:
    m = []
    theta = []
    rho = []
    slope = []
    y_int = []
    line_params = []

    for line in linesP:
        x1, y1, x2, y2 = line[0]
        m0 = (y2-y1)/(x2-x1)
        theta0 = np.arctan(m0)
        rho0 = np.sqrt((x2-x1)**2+(y2-y1)**2)
        m.append(m0)
        theta.append(theta0)
        rho.append(rho0)
        #x1, y1, x2, y2 = line.reshape(4)
        if 18 < np.rad2deg(theta0)<20:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope0 = parameters[0]
            y_int0 = parameters[1]
            slope.append(slope0)
            y_int.append(y_int0)
            line_params.append((slope, y_int))

        line_av = np.average(line_params, axis=0)
        cv.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # plot frame
    plt.figure(figsize=(10,10))
    plt.imshow(dmy, cmap= "gray")
    plt.show()


#    for i in range(0, len(lines)):
#        rho = lines[i][0][0]
#        theta = lines[i][0][1]
#        a = np.cos(theta)
#        b = np.sin(theta)
#        x0 = a * rho
#        y0 = b * rho
#        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a))#)
#        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)#
#	plt.imshow(cdst, cmap='gray')
#	plt.show()
#pt1, pt2 = average_houghlines(lines)

#cdst = np.copy(edges)
#cv.line(cdst, pt1, pt2, (255,0,0), 3, cv.LINE_AA)


#for line in linesP:#
#	x1, y1, x2, y2 = line[0]###
#	cv.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
#x = [x1, x2] 
#y = [y1, y2] 



#f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10,6))
#ax1.imshow(masked_frame, cmap='gray')
#ax2.imshow(gabor_image, cmap='gray')
#ax3.imshow(edges, cmap='gray')
#ax4.plot(x,y, color="white", linewidth=3) 
#ax4.imshow(masked_frame, cmap='gray')

#plt.show()
#plt.close()