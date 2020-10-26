import cv2 as cv
import numpy as np
import math as m
import statistics as st
from astropy.convolution import Gaussian2DKernel

# For the pocessing we assume that we know approximately the angle the needle enters the tissue. The angle is measured with respect to the tissue face.
# Furthermore we know the ROI inside the whole image the sonographic unit puts out. We assume that the needle enters the image at x=0.
# Also the needles angle is between 0 an pi/2 measured clockwise from the x-axis
############################################

# Load frame
filename = './resources/single_frames/30.png'
frame = cv.imread(filename, cv.IMREAD_GRAYSCALE)

# Crop frame to ROI
frame_roi = frame[111:701, 368:1015]

height = frame_roi.shape[0]
width = frame_roi.shape[1] 

print("ROI size (y, x): ", frame_roi.shape)

############################################

# Make anisotropic gaussian filter kernel. First two arguments are the std. devs. of the 2D Gaussian distribution. Third argument is the angle in radians the kernel is rotated clockwise. 
sigma_x = width/32; sigma_y = sigma_x/8; angle = 0.5236;
kernel = Gaussian2DKernel(sigma_x, sigma_y, angle)

print("Filterkernel size (y, x): ", kernel.array.shape)

cv.imshow("Filter kernel", 255*kernel.array/np.linalg.norm(kernel.array))  

# Filter frame with kernel
frame_filtered_raw = cv.filter2D(frame_roi, -1, kernel.array)
frame_filtered = frame_filtered_raw

# Show filtered frame
cv.imshow("Filtered frame", frame_filtered)    

############################################

# Use Canny-Edge-Detector with the two hystersis thresholds
th1 = 10; th2 = 50;
frame_edges = cv.Canny(frame_filtered, th1, th2, None, 3)
cv.imshow("Edges in frame", frame_edges) 

############################################

# Use Houghline algorithm to find lines. 
dist_res = 5; angle_res = 5*np.pi/180; threshold_votes = 80; 
min_num_pts = m.sqrt(width**2+height**2)/10; max_line_gap = 2*min_num_pts;

linesP = cv.HoughLinesP(frame_edges, dist_res, angle_res, threshold_votes, None, min_num_pts, max_line_gap)
lines = [];
if linesP is not None:
    for i in range(0, len(linesP)):
        lines.append(linesP[i][0])

print("Lines in HoughlinesP found: ", len(lines))

############################################

# Remove lines that are not in the expected angle range
max_angle_deviation = 0.0425;          # 5°
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

print("Lines after removing lines outside expected angle range: ", len(lines))

############################################

# Remove lines which angles deviate too much from the rest
if len(lines) > 0:
    quants = st.quantiles(line_angles, n=4)
    quartile_1 = quants[0]; quartile_2 = quants[2];
    iqr_1_2 = 1.2*(quartile_2-quartile_1)
    lower_bound = quartile_1 - iqr_1_2; upper_bound = quartile_2 + iqr_1_2;

    indices_to_del = [];
    
    if line_angles is not None:
        for i in range(0, len(line_angles)):
            if line_angles[i] < lower_bound or line_angles[i] > upper_bound:
                indices_to_del.append(i)

    lines = np.delete(lines, indices_to_del, axis=0)
    line_angles = np.delete(line_angles, indices_to_del)

print("Lines after removing angle outliers: ", len(lines))

############################################

# Remove the lines where their y-intersect deviates too much from the rest
b = []

if lines is not None:
    for i in range(0, len(line_angles)):
        l = lines[i]
        b.append(l[1]-m.tan(line_angles[i])*l[0])

if len(lines) > 0:
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

print("Lines after removing y-intersect outliers: ", len(lines))

############################################
    
# Print mean line and all remaining lines
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


############################################
        
# Find the endpoint of the needle 
absolute_max = 0
needle_tip_idx = 0
if lines is not None:
    for i in range(0, len(lines)):
        l = lines[i]
        absolute = m.sqrt(l[2]**2 + l[3]**2)
        if absolute > absolute_max:
            absolute_max = absolute
            needle_tip_idx = i
            
cv.circle(frame_filtered, (lines[i][2], lines[i][3]), 8, (255,128,0), -1)
cv.imshow("Filtered frame with lines", frame_filtered)

frame_filtered_raw = cv.cvtColor(frame_filtered_raw, cv.COLOR_GRAY2RGB)
cv.line(frame_filtered_raw, start, end, (0,128,255), 4, cv.LINE_8)
cv.circle(frame_filtered_raw, (lines[i][2], lines[i][3]), 8, (255,128,0), -1)
cv.imshow("Filtered frame with needle and needle tip", frame_filtered_raw)

cv.waitKey(30000)
