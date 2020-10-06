import cv2 as cv
import numpy as np
import math as m
from astropy.convolution import Gaussian2DKernel

############################################

# Load frame
filename = '../resources/single_frames/30.png'
frame = cv.imread(filename, cv.IMREAD_GRAYSCALE)

# Crop frame to ROI
frame_roi = frame[111:701, 368:1015]

############################################

# Make anisotropic gaussian filter kernel. First two arguments are the std. devs. of the 2D Gaussian distribution. Third argument is the angle in radians the kernel is rotated counterclockwise. 
sigma_x = 10; sigma_y = 3; angle = 0.5236;
kernel = Gaussian2DKernel(sigma_x, sigma_y, angle)

# Filter frame with kernel. Second argument
frame_filtered = cv.filter2D(frame_roi, -1, kernel.array)

# Show filtered frame
cv.imshow("Filtered frame", frame_filtered)    

############################################

#Use Canny-Edge-Detector with the two hystersis thresholds
th1 = 3; th2 = 50;
frame_edges = cv.Canny(frame_filtered, th1, th2, None, 3)
cv.imshow("Edges in frame", frame_edges) 

############################################

# Use Houghline algorithm to find lines. 
dist_res = 5; angle_res = 5*np.pi/180; threshold_votes = 100; 
min_num_pts = 50; max_line_gap = 50;
linesP = cv.HoughLinesP(frame_edges, dist_res, angle_res, threshold_votes, None, min_num_pts, max_line_gap)

############################################

# Remove lines that are not in the expected angle range
max_angle_deviation = 0.085;
indices_to_del = [];

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        line_angle = m.atan((l[3]-l[1])/(l[2]-l[0]))
        
        if line_angle > angle + max_angle_deviation or line_angle < angle - max_angle_deviation:
        	indices_to_del.append(i)
 
linesP = np.delete(linesP, indices_to_del, axis=0)

############################################

# Print each line on frame
print(len(linesP))
frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_GRAY2RGB)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(frame_filtered, (l[0], l[1]), (l[2], l[3]), (255,0,0), 1, cv.LINE_AA)

cv.imshow("Filtered frame with lines", frame_filtered)     
cv.waitKey(30000)
