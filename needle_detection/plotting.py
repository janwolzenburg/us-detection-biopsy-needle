import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw


"""
description
    gets the x- and y-values representing the line to be drawn in the raw frame
    
arguments
    line_b  y-intersect of the line
    line_m  incline of the line
    shape   shape of the frame the line will be drawn in
    
returns
    line_y  array with y-values
    line_x  array with x-values
    score = np.empty([num_prob_lines])

frame_empty = np.zeros([height, width])
for i in range(0, num_prob_lines):
    frame_empty[prob_lines[i][1][:],prob_lines[i][0][:]] = 1

fig_lines = plt.figure()
frame_lines_axes = fig_lines.add_axes([0,0,1,1])
frame_lines_axes.imshow(frame_empty, cmap='gray')
"""
def get_draw_line(line_b, line_m, shape):

    if line_b < 0:
        x_start = int(round(-line_b/line_m))
        y_start = int(0)
    else:
        x_start = int(0)
        y_start = int(line_b)

    if line_m*(shape[1]-x_start) + y_start > shape[0] - 3:
        y_end = shape[0]-3-1
        x_end = int(round((y_end - y_start)/line_m + x_start))
    else:
        x_end = shape[1]-1
        y_end = int(round(line_m*(x_end-x_start) + y_start))
    
    line_y, line_x = draw.line(y_start, x_start, y_end, x_end)
    
    return line_y, line_x



def plotting_tip(diff_min_x, diff_min_y):
    tip_x = int(round(diff_min_x/rescale_factor))
    tip_y = int(round(diff_min_y/rescale_factor))

    circle_y, circle_x = disk([tip_y, tip_x], 12)
    frame_raw[circle_y, circle_x] = 1
    return 

    def plot_lines()