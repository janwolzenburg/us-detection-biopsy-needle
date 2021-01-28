import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw


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
    
    m_stg = np.linspace(np.tan(expected_angle-angle_range),np.tan(expected_angle+angle_range), num_angles)
    b = np.linspace(expected_b-b_range, expected_b+b_range, num_bs, dtype=int)
    delta_b = np.floor(line_wdt/2/np.cos(2*np.pi*expected_angle/360))
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
            line_lengths[line_idx] = np.sqrt((prob_lines[line_idx][0][x_limits[line_idx][1]]-
                                             prob_lines[line_idx][0][x_limits[line_idx][0]])**2+
                                            (prob_lines[line_idx][1][x_limits[line_idx][1]]-
                                             prob_lines[line_idx][1][x_limits[line_idx][0]])**2)
    
    return prob_lines, num_lines, all_bs, all_ms, delta_b, y_pts, line_lengths, x_limits
    
##############################################################################################################
##############################################################################################################


