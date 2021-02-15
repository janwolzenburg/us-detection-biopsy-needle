from needle_detection.preprocessing import read_frame, get_ROI
from needle_detection.kernels import filter_kernel_parameters, build_gauss_kernel, build_sobel_kernel, convolution
from needle_detection.line_detection import line_detector, build_probe_lines
import matplotlib.pyplot as plt
import numpy as np
import needle_detection.parameters as p
from skimage.draw import disk, line
angle = p.angles[4]


frame = read_frame('../needle_detection/resources/test/30.png')
ROI, rescale_factor = get_ROI(frame, angle)

sigma_x, sigma_y, n = filter_kernel_parameters(frame)  

gauss_kernel = build_gauss_kernel(sigma_x, sigma_y, angle)
sobel_kernel =  build_sobel_kernel(n, angle)
convolved_kernels = convolution(sobel_kernel, gauss_kernel)



ROI, rescale_factor = get_ROI(frame, angle)
filtered_frame = convolution(ROI, convolved_kernels)


prob_lines, num_lines, all_bs, all_ms, delta_b, y_pts, line_lengths, x_limits = build_probe_lines(filtered_frame, angle, rescale_factor)

line_b, line_m, line_x, line_y, tip_x, tip_y, intensity_along_line, intensity_along_line_diff, diff_min_x, diff_min_y = line_detector(frame, num_lines, prob_lines, x_limits, line_lengths, y_pts, delta_b, rescale_factor, frame, all_bs, all_ms)


#tip_x = int(round(diff_min_x/rescale_factor))
#tip_y = int(round(diff_min_y/rescale_factor))
circle_y, circle_x = disk([tip_y, tip_x], 12)
frame[circle_y, circle_x] = 255

for t in range(-3, 3):
        frame[line_y+t, line_x] = 255
#rr, cc = line(line_y[0], line_x[0], line_y[len(line_y)-1],line_y[len(line_x)-1])
#frame[rr, cc] = 255



f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
ax1.imshow(frame, cmap='gray')
ax2.imshow(filtered_frame, cmap='gray')
plt.savefig('172_line.png', dpi=300)
plt.show()
plt.close()