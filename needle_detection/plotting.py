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

