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


# Parameters

# Shape of ROI
width = p.roi_x_2 - p.roi_x_1
height = p.roi_y_2 - p.roi_y_1      # width and height

# Needle Parameters----------------------------
dist_per_pixel = p.depth/height                                     # Distance in cm per Pixel
expected_angle = np.pi/2-p.expected_angle/360*2*np.pi 

# Resize Parameters----------------------------
if width <= height:
    new_height = p.processing_size
    rescale_factor = new_height/height
    new_width = round(rescale_factor*width)
else:
    new_width = p.processing_size
    rescale_factor = new_width/width
    new_height = round(rescale_factor*height)
    
width = new_width
height = new_height

wdt_hgt_ref = np.sqrt(width**2+height**2)                            # Refernence for some parameters

# Processing----------------------------------
# Filter kernel parameters for smoothing and edge improvement
sigma_x = wdt_hgt_ref/75                                            # Sigma x of gaussian kernel
sigma_y = sigma_x/p.kernel_aspect_ratio                             # Sigma y
sob_kernel_size = int(12)                                       # Size of Sobel Kernel

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
plt.figure()
plt.imshow(kernel)

# Build probing lines
prob_lines, num_prob_lines, all_bs, all_ms, delta_b, y_pts = build_probe_lines(expected_angle, angle_range, p.num_angles, expected_b, b_range, p.num_bs, p.line_wdt, width, height)
score = np.empty([num_prob_lines])

frame_empty = np.zeros([height, width])
for i in range(0, num_prob_lines):
    frame_empty[prob_lines[i][1][:],prob_lines[i][0][:]] = 1

fig_lines = plt.figure()
frame_lines_axes = fig_lines.add_axes([0,0,1,1])
frame_lines_axes.imshow(frame_empty, cmap='gray')

################################################
# Repeated steps

fig_frames = plt.figure()
frame_axes = fig_frames.subplots(4, len(p.frames))

times = np.empty([len(p.frames),5])

for i in range(0, len(p.frames)):

    t1 = time.time()
    # Get image and crop to ROI
    filename = p.frames[i]
    frame = plt.imread(p.image_path + filename)
    frame_raw = frame[p.roi_y_1:p.roi_y_2, p.roi_x_1:p.roi_x_2]
    frame_roi = resize(frame_raw, (height, width))
    times[i,0] = time.time()-t1
    t1 = time.time()
    
    # Apply filters
    frame_filtered = convolve(frame_roi, kernel)

    # Normalize
    frame_filtered = normal(frame_filtered, np.uint8)
    times[i,1] = time.time()-t1
    t1 = time.time()

    # Line probing
    for j in range(0, num_prob_lines):
        score[j] = np.sum(frame_filtered[prob_lines[j][1][:],prob_lines[j][0][:]])
        
    score_max_idx = np.argmax(score)

    n = int(wdt_hgt_ref/50)
    
    
    diff_min_x, diff_min_y, intensity_along_line, intensity_along_line_diff = find_tip(frame_filtered, prob_lines[score_max_idx], n, width, y_pts, delta_b)
    
       
    
    times[i,2] = time.time()-t1
    t1 = time.time()

    line_b = round(all_bs[score_max_idx]/rescale_factor)
    line_m = all_ms[score_max_idx]

    line_y, line_x = get_draw_line(line_b, line_m, frame_raw.shape)



    for t in range(-3, 3):
        frame_raw[line_y+t, line_x] = 1
        
    tip_x = int(round(diff_min_x/rescale_factor))
    tip_y = int(round(diff_min_y/rescale_factor))

    circle_y, circle_x = disk([tip_y, tip_x], 12)
    frame_raw[circle_y, circle_x] = 1

    frame_axes[0][i].imshow(frame_raw, cmap='gray')
    frame_axes[1][i].imshow(frame_filtered, cmap='gray')
    frame_axes[2][i].plot(range(0, width), intensity_along_line)
    frame_axes[3][i].plot(range(0, width), intensity_along_line_diff)
    times[i,3] = time.time()-t1
    t1 = time.time()

print(line_m, line_b)
print("Average execution times")
print("Loading and resize: ", np.mean(times[:,0]))
print("Filtering and normalization: ", np.mean(times[:,1]))
print("Line probing: ", np.mean(times[:,2]))
print("Showing ", np.mean(times[:,3]))
print("All ", np.mean(np.sum(times,1)))

plt.show()
