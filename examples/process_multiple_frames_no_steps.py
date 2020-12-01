#########################################################################################
#   Author:         Jan Wolzenburg @ Fachhochschule Südwestfalen (Lüdenscheid, Germany) #
#   Date:           01.12.2020                                                          #
#   Version:        0.1                                                                 #
#   Description:    Needle detection in continueous US-image stream                     #
#########################################################################################


################################################
# Modules

import math as m
import numpy as np

import scipy.ndimage as spimg

import skimage.draw as skdw
import skimage.transform as sktr

import matplotlib.image as matimg
import matplotlib.pyplot as matpl

import functions as f

import parameters as p

################################################
# Parameters

# Shape of ROI
width = p.roi_x_2 - p.roi_x_1; height = p.roi_y_2 - p.roi_y_1;      # width and height

# Needle Parameters----------------------------
dist_per_pixel = p.depth/height                                     # Distance in cm per Pixel
expected_angle = m.pi/2-p.expected_angle/360*2*m.pi 

# Resize Parameters----------------------------

if width <= height:
    new_height = p.processing_size;
    rescale_factor = new_height/height
    new_width = round(rescale_factor*width)
else:
    new_width = p.processing_size;
    rescale_factor = new_width/width
    new_height = round(rescale_factor*height)
    
width = new_width; height = new_height;
wdt_hgt_ref = m.sqrt(width**2+height**2)                            # Refernence for some parameters

# Processing----------------------------------
# Filter kernel parameters for smoothing and edge improvement
sigma_x = wdt_hgt_ref/75                                            # Sigma x of gaussian kernel
sigma_y = sigma_x/p.kernel_aspect_ratio                             # Sigma y
sob_kernel_size = int(12);                                          # Size of Sobel Kernel

# Probing lines
# b -> Pixel value where the needle enters the picture
expected_b = round(p.insertion_depth/dist_per_pixel*rescale_factor)
b_range = int(height/16)
angle_range = 5/360*m.pi*2


################################################
# Initialisation

# Build Gauss filter kernel
kernel = f.build_gauss_kernel(sigma_x, sigma_y, expected_angle)

# Build rotated Sobel
sob_kernel = f.build_sobel_kernel(sob_kernel_size, expected_angle);
kernel = spimg.convolve(kernel, sob_kernel)

# Build probing lines
prob_lines, num_prob_lines, all_bs, all_ms, delta_b, y_pts = f.build_probe_lines(expected_angle, angle_range, p.num_angles, expected_b, b_range, p.num_bs, p.line_wdt, width, height)
score = np.empty([num_prob_lines])


################################################
# Repeatet steps

for i in range(0, len(p.frames)):

    # Get image and crop to ROI
    filename = p.frames[i]
    frame = matimg.imread(p.image_path + filename)
    frame_raw = frame[p.roi_y_1:p.roi_y_2, p.roi_x_1:p.roi_x_2]
    frame_roi = sktr.resize(frame_raw, (height, width))
    
    # Apply filters
    frame_filtered = spimg.convolve(frame_roi, kernel)
    # Normalize
    frame_filtered = f.normal(frame_filtered, np.uint8)

    # Line probing
    for j in range(0, num_prob_lines):
        score[j] = np.sum(frame_filtered[prob_lines[j][1][:],prob_lines[j][0][:]])
    score_max_idx = np.argmax(score)

    n = int(wdt_hgt_ref/50)
    diff_min_x, diff_min_y, intensity_along_line, intensity_along_line_diff = f.find_tip(frame_filtered, prob_lines[score_max_idx], n, width, y_pts, delta_b)
    
    line_b = round(all_bs[score_max_idx]/rescale_factor)
    line_m = all_ms[score_max_idx]
    line_y, line_x = f.get_draw_line(line_b, line_m, frame_raw.shape)

    for t in range(-3, 3):
        frame_raw[line_y+t, line_x] = 1
        
    tip_x = int(round(diff_min_x/rescale_factor))
    tip_y = int(round(diff_min_y/rescale_factor))

    circle_y, circle_x = skdw.disk([tip_y, tip_x], 12)
    frame_raw[circle_y, circle_x] = 1
    matpl.figure()
    matpl.imshow(frame_raw, cmap='gray')
    
    
matpl.show()
