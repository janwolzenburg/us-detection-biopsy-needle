import numpy as np
from scipy.ndimage import convolve
from skimage.draw import disk
from skimage.transform import resize
import matplotlib.pyplot as plt
from needle_detection.tip_detection import find_tip
from needle_detection.plotting import get_draw_line 
from needle_detection.line_detection import build_probe_lines
import needle_detection.parameters as p
from needle_detection.kernels import build_gauss_kernel, build_sobel_kernel
from needle_detection.preprocessing import normal
import time

################################################
# Parameters

# Shape of ROI
width = p.roi_x_2 - p.roi_x_1; height = p.roi_y_2 - p.roi_y_1;      # width and height

# Needle Parameters----------------------------
dist_per_pixel = p.depth/height                                     # Distance in cm per Pixel
expected_angle = np.pi/2-p.expected_angle/360*2*np.pi 

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
wdt_hgt_ref = np.sqrt(width**2+height**2)                            # Refernence for some parameters

# Processing----------------------------------
# Filter kernel parameters for smoothing and edge improvement
sigma_x = wdt_hgt_ref/75                                            # Sigma x of gaussian kernel
sigma_y = sigma_x/p.kernel_aspect_ratio                             # Sigma y
sob_kernel_size = int(12);                                          # Size of Sobel Kernel

# Probing lines
# b -> Pixel value where the needle enters the picture
expected_b = round(p.insertion_depth/dist_per_pixel*rescale_factor)
b_range = int(height/16)
angle_range = 5/360*np.pi*2


################################################
# Initialisation

# Build Gauss filter kernel
kernel = build_gauss_kernel(sigma_x, sigma_y, expected_angle)

# Build rotated Sobel
sob_kernel = build_sobel_kernel(sob_kernel_size, expected_angle);
kernel = convolve(kernel, sob_kernel)

# Build probing lines
prob_lines, num_prob_lines, all_bs, all_ms, delta_b, y_pts = build_probe_lines(expected_angle, angle_range, p.num_angles, expected_b, b_range, p.num_bs, p.line_wdt, width, height)
score = np.empty([num_prob_lines])


################################################
# Repeatet steps

for i in range(0, len(p.frames)):

    # Get image and crop to ROI
    filename = p.frames[i]
    frame = plt.imread(p.image_path + filename)
    frame_raw = frame[p.roi_y_1:p.roi_y_2, p.roi_x_1:p.roi_x_2]
    frame_roi = resize(frame_raw, (height, width))
    
    # Apply filters
    frame_filtered = convolve(frame_roi, kernel)
    # Normalize
    frame_filtered = normal(frame_filtered, np.uint8)

    # Line probing
    for j in range(0, num_prob_lines):
        score[j] = np.sum(frame_filtered[prob_lines[j][1][:],prob_lines[j][0][:]])
    score_max_idx = np.argmax(score)

    n = int(wdt_hgt_ref/50)
    diff_min_x, diff_min_y, intensity_along_line, intensity_along_line_diff = find_tip(frame_filtered, prob_lines[score_max_idx], n, width, y_pts, delta_b)
    
    line_b = round(all_bs[score_max_idx]/rescale_factor)
    line_m = all_ms[score_max_idx]
    line_y, line_x = get_draw_line(line_b, line_m, frame_raw.shape)

    for t in range(-3, 3):
        frame_raw[line_y+t, line_x] = 1
        
    tip_x = int(round(diff_min_x/rescale_factor))
    tip_y = int(round(diff_min_y/rescale_factor))

    circle_y, circle_x = disk([tip_y, tip_x], 12)
    frame_raw[circle_y, circle_x] = 1
    plt.figure()
    plt.imshow(frame_raw, cmap='gray')
    
    
plt.show()
