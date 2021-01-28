import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw


"""
description
    find the needle tip
    
arguments
    frame           the 2D frame
    prob_line       one probing line
    window_size     half the length of one side the moving window (=length/2-1)
    width           frame_width
    y_pts           projection of the line width to the y-axis
    delta_b         y_pts/2-1
    
returns
    diff_min_x                  x-value of needle tip
    diff_min_y                  y-value of needle tip

   
"""
def find_tip(frame, prob_line, window_size, width, y_pts, delta_b):
    
    intensity_along_line = np.zeros([width], float)
    
    # Iterate over each x-value in range given by moving window size
    for k in range(window_size, width-window_size):
        # Calculate indices. Necessary because the last index of probing lines does not select the x-value.
        # To select a rectengualar area of the line the window size and width of the line has to be considered
        i_start = y_pts*(k-window_size)
        i_end = y_pts*(k+window_size) + 2*delta_b
        # Calculate for each x-value the sum of pixel values inside the line selection
        intensity_along_line[k] = np.sum(frame[prob_line[1][i_start:i_end], prob_line[0][i_start:i_end]])

    # Side length of window
    m = 2*(2*window_size+1)
    # Moving average
    intensity_along_line = signal.convolve(intensity_along_line, np.full([m],1/m,dtype=float), mode='same')
    
    
    # Fill start and end with nearest value
    if m > window_size:
            window_size = m
    for l in range(0, window_size):
        intensity_along_line[l] = intensity_along_line[window_size]  
    for l in range(width-window_size-1, width):
       intensity_along_line[l] = intensity_along_line[width-window_size-1]  
    
    # Differenciate and locate needle tip. The minimum in der derivative of the intesity along the line is considered to be the needle tip
    difference = np.diff(intensity_along_line, 1)
    intensity_along_line_diff = np.append(difference[0], difference)
    diff_min_idx = np.argmin(intensity_along_line_diff)

    diff_min_x = prob_line[0][y_pts*diff_min_idx+delta_b]
    diff_min_y = prob_line[1][y_pts*diff_min_idx+delta_b]
    
    return diff_min_x, diff_min_y