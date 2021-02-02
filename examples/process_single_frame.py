from needle_detection.read_frames import read_frame
from needle_detection.preprocessing import normal, get_ROI
from needle_detection.kernels import filter_kernel_parameters, build_gauss_kernel, build_sobel_kernel, convolve_kernels, filtering
from needle_detection.line_detection import line_detector, build_probe_lines
import matplotlib.pyplot as plt
import numpy as np

angle = 60
frame = read_frame('../needle_detection/resources/34.png')

sigma_x, sigma_y, n = filter_kernel_parameters(frame)   ##hier noch umbauen, sodass das vor einlesen (jedes) frames geschieht

gauss_kernel = build_gauss_kernel(sigma_x, sigma_y, angle)
sobel_kernel =  build_sobel_kernel(n, angle)
convolved_kernels = convolve_kernels(sobel_kernel, gauss_kernel)


normalized_frame = normal(frame)
ROI, rescale_factor = get_ROI(normalized_frame, angle )

filtered_frame = filtering(ROI, convolved_kernels)
plt.imshow(filtered_frame, cmap='gray')
plt.show()

prob_lines, num_lines, all_bs, all_ms, delta_b, y_pts, line_lengths, x_limits = build_probe_lines(frame, angle, rescale_factor)

#line_detector(filtered_frame, num_lines, prob_lines, x_limits, line_lengths, y_pts, delta_b, all_bs, rescale_factor, frame)