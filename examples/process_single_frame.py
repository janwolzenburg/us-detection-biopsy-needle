import cv2 as cv
import numpy as np
import math as m
import statistics as st
from astropy.convolution import Gaussian2DKernel

############################################

# Load frame
filename = './resources/single_frames/30.png'
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

# Use Canny-Edge-Detector with the two hystersis thresholds
th1 = 3; th2 = 50;
frame_edges = cv.Canny(frame_filtered, th1, th2, None, 3)
cv.imshow("Edges in frame", frame_edges) 

############################################

# Use Houghline algorithm to find lines. 
dist_res = 5; angle_res = 5*np.pi/180; threshold_votes = 100; 
min_num_pts = 50; max_line_gap = 80;
linesP = cv.HoughLinesP(frame_edges, dist_res, angle_res, threshold_votes, None, min_num_pts, max_line_gap)
lines = [];
if linesP is not None:
    for i in range(0, len(linesP)):
        lines.append(linesP[i][0])

############################################

# Remove lines that are not in the expected angle range
max_angle_deviation = 2*0.085;
indices_to_del = [];
line_angles = [];

if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i]
        line_angles.append(m.atan((l[1]-l[3])/(l[0]-l[2])))
        
        if line_angles[i] > angle + max_angle_deviation or line_angles[i] < angle - max_angle_deviation:
        	indices_to_del.append(i)
 
lines = np.delete(lines, indices_to_del, axis=0)
line_angles = np.delete(line_angles, indices_to_del)

############################################

# Remove lines which angles deviate too much from the rest when there are enough lines
if len(lines) > 5:
    quants = st.quantiles(line_angles, n=4)
    quartile_1 = quants[0]; quartile_2 = quants[2];
    iqr_1_5 = 1.5*(quartile_2-quartile_1)
    lower_bound = quartile_1 - iqr_1_5; upper_bound = quartile_2 + iqr_1_5;

    indices_to_del = [];
    
    if line_angles is not None:
        for i in range(0, len(line_angles)):
            if line_angles[i] < lower_bound or line_angles[i] > upper_bound:
                indices_to_del.append(i)

    lines = np.delete(lines, indices_to_del, axis=0)
    line_angles = np.delete(line_angles, indices_to_del)

############################################

# Remove the lines where their y-intersect deviates too much from the rest
b = []
height = frame_filtered.shape[0]
width = frame_filtered.shape[1]

if lines is not None:
    for i in range(0, len(line_angles)):
        l = lines[i]
        b.append(l[1]-m.tan(line_angles[i])*l[0])

if len(lines) > 3:
    quants = st.quantiles(b, n=4)
    quartile_1 = quants[0]; quartile_2 = quants[2];
    iqr_1_5 = 1.5*(quartile_2-quartile_1)
    lower_bound = quartile_1 - iqr_1_5; upper_bound = quartile_2 + iqr_1_5;

    indices_to_del = [];
    
    if b is not None:
        for i in range(0, len(b)):
            if b[i] < lower_bound or b[i] > upper_bound:
                indices_to_del.append(i)

    lines = np.delete(lines, indices_to_del, axis=0)
    line_angles = np.delete(line_angles, indices_to_del)
    b = np.delete(b, indices_to_del)


############################################
    
# Print mean line
B = st.mean(b)
M = m.tan(st.mean(line_angles)) 


start = (0, round(B))

if M*width + B > height:
    end = (round((height-B)/M), height)
else:
    end = (width, round(M*width + B))
    
frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_GRAY2RGB)
cv.line(frame_filtered, start, end, (0,128,255), 6, cv.LINE_AA)
# Print each line on frame

if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i]
        cv.line(frame_filtered, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1, cv.LINE_AA)
        cv.circle(frame_filtered, (0, round(b[i])), 4, (0,0,255), -1)



cv.imshow("Filtered frame with lines", frame_filtered)     
cv.waitKey(30000)
