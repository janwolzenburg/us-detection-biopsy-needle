import math as m
import numpy as np

import astropy.convolution as ascon

from scipy import ndimage, signal

from skimage import draw

##############################################################################################################
##############################################################################################################

"""
description
    Build the roteted anisotropic gaussian filter kernel
    
arguments
    sigma_x     sigma in x-direction
    sigma_y     sigma in y-direction
    angle       clockwise rotation angle in radians
     
returns
    kernel      roteted filter kernel
"""
def build_gauss_kernel(sigma_x, sigma_y, angle):
    
    # Calculate gaussian kernel
    kernel = ascon.Gaussian2DKernel(sigma_x, sigma_y, 0)
    # Extract size and kernel values
    x_size = kernel.shape[0]; y_size = kernel.shape[1]
    kernel = kernel.array
    # Rotate
    kernel = ndimage.rotate(kernel, -angle/(2*m.pi)*360, reshape=False)

    # Parameters for cropping
    max_in_kernel = np.amax(abs(kernel))
    threshold = 0.05*max_in_kernel 

    # Crop the kernel to reduce its size
    x_start = 0;
    for i in range(0, x_size, 1):
        if abs(max(kernel[i,:])) > threshold:
            x_start = i
            break
    x_end = (x_size-1)-x_start

    y_start = 0;
    for i in range(0, y_size, 1):
        if abs(max(kernel[:,i])) > threshold:
            y_start = i
            break
    y_end = (y_size-1)-y_start

    kernel = kernel[x_start:x_end, y_start:y_end]

    return kernel

##############################################################################################################
##############################################################################################################

"""
description
    Build rotated sobel kernel
    
arguments
    n       size of sobel kernel. Must be a multiple of three
    angle   clockwise rotation angle in radians
returns
    sob_kernel  sobel filter kernel
    
"""
def build_sobel_kernel(n, angle):
    
    n_3 = int(n/3)
    sob_kernel = np.zeros([n,n])
    sob_kernel[0:n_3,0:n_3] = 1; sob_kernel[0:n_3,n_3:2*n_3] = 2; sob_kernel[0:n_3,2*n_3:3*n_3] = 1
    sob_kernel[2*n_3:3*n_3,0:n_3] = -1; sob_kernel[2*n_3:3*n_3,n_3:2*n_3] = -2; sob_kernel[2*n_3:3*n_3,2*n_3:3*n_3] = -1 
    sob_kernel = ndimage.rotate(sob_kernel, -angle/(2*m.pi)*360, reshape=False)

    return sob_kernel
    
##############################################################################################################
##############################################################################################################

"""
description
    builds the lines for probing.
    
arguments
    expected angle      the expected insertion angle with respect to x-axis
    angle range         the angle range in which lines will be build
    num_angles          the amount of lines in range
    
    expected_b          the expected y-position the needle enteres the picture (x = 0)
    b_range             the range of bs
    num_bs              the amount of bs in range
    
    line_wdt            the width of the probing lines 
    frame_width         the width (x-size) of the frame
    frame_height        the height (y-size) of the frame
returns
    prob_lines          the multidimensional array that holds the lines. Indexing is: [line number][x(0) or y(1) value][pixel index]
    num_lines           amount of lines
    all_bs              array that holds all y-intersects
    all_ms              array that holds all inclines
    delta_b             projection of half line width to the y-axis
    y_pts               number of pixels in projection of line width
    line_lengths        lengths of all lines
    x_limits            limits where the line enteres and exits the
    
"""
def build_probe_lines(expected_angle, angle_range, num_angles, expected_b, b_range, num_bs, line_wdt, frame_width, frame_height):
    
    m_stg = np.linspace(m.tan(expected_angle-angle_range),m.tan(expected_angle+angle_range), num_angles)
    b = np.linspace(expected_b-b_range, expected_b+b_range, num_bs, dtype=int)
    delta_b = m.floor(line_wdt/2/m.cos(2*m.pi*expected_angle/360))
    y_pts = (2*delta_b+1)
    
    num_lines = len(b)*len(m_stg)
    score = np.empty([num_lines])
    prob_lines = np.empty([num_lines,2, (2*delta_b+1)*frame_width], int)
    all_bs = np.empty([num_lines])
    all_ms =np.empty([num_lines])
    line_lengths = np.empty([num_lines])
    x_limits= np.zeros([num_lines, 2], int)
    
    # Iterate over every y-intersect
    for i in range(0, len(b)):
        # Iterate over every incline
        for a in range(0, len(m_stg)):
            line_idx = a + len(m_stg)*i
            # Iterate over every pixel in x-size
            for j in range(0, frame_width):
                # Iterate over every y-point for given x-value
                for k in range(0, y_pts):
                    j_idx = y_pts*j + k
                    
                    # Save cuurent x-value und y-value
                    prob_lines[line_idx][0][j_idx] = j
                    prob_lines[line_idx][1][j_idx] = round(m_stg[a]*j + b[i]) + k - delta_b       
                    
                    # Check line entry and exit point
                    if prob_lines[line_idx][1][j_idx] < 0:
                        prob_lines[line_idx][1][j_idx] = 0
                        x_limits[line_idx][0] = j_idx
                    if prob_lines[line_idx][1][j_idx] > frame_height-1:
                        prob_lines[line_idx][1][j_idx] = frame_height-1
                        if x_limits[line_idx][1] == 0:
                            x_limits[line_idx][1] = j_idx
            
            # If exit point is has not been defined
            if x_limits[line_idx][1] == 0:
                x_limits[line_idx][1] = j_idx
            
            # Store y-intersect and incline
            all_bs[line_idx] = b[i]
            all_ms[line_idx] = m_stg[a]
            
            # Calculate line lenght
            line_lengths[line_idx] = m.sqrt((prob_lines[line_idx][0][x_limits[line_idx][1]]-
                                             prob_lines[line_idx][0][x_limits[line_idx][0]])**2+
                                            (prob_lines[line_idx][1][x_limits[line_idx][1]]-
                                             prob_lines[line_idx][1][x_limits[line_idx][0]])**2)
    
    return prob_lines, num_lines, all_bs, all_ms, delta_b, y_pts, line_lengths, x_limits
    
##############################################################################################################
##############################################################################################################

"""
description
    Normalizes a frame to a range from 0 to 255
        
arguments
    frame       2D array
    out_type    type of returned normalized frame

returns
    normalized frame with given type
    
"""
def normal(frame, out_type):
    frame_min = np.amin(frame)
    frame_max = np.amax(frame)
    k = 255/(frame_max - frame_min)
    frame = (frame - frame_min)*k
    return frame.astype(out_type)

##############################################################################################################
##############################################################################################################

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
    
##############################################################################################################
##############################################################################################################

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

