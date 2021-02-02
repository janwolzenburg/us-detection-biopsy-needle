import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw
import needle_detection.parameters as p

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
    x_limits            limits where the line enters and exits the ...
    
"""
def build_probe_lines(frame, expected_angle, rescale_factor):
    # Probing lines
    # b -> Pixel value where the needle enters the picture
    angle_range = np.deg2rad(p.angle_range)
    width = int(np.shape(frame)[1])##das hier nochmal checken
    height = int(np.shape(frame)[0]) ##das hier nochmal checken
    dist_per_pixel = p.depth/height   
    expected_b = round(p.insertion_depths[0]/dist_per_pixel*rescale_factor)
    b_range = int(height/16) 
    angle_range = np.deg2rad(p.angle_range)
    m_stg = np.linspace(np.tan(expected_angle-angle_range),np.tan(expected_angle+angle_range), p.num_angles)
    b = np.linspace(expected_b-b_range, expected_b+b_range, p.num_bs, dtype=int)
    delta_b = int(np.floor(p.line_wdt/2/np.cos(2*np.pi*expected_angle/360))) ##np.rad..
    y_pts = (2*delta_b+1)


    num_lines = len(b)*len(m_stg)
    score = np.empty([num_lines])


    prob_lines = np.empty([num_lines,2, (2*delta_b+1)*width], int) 
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
            for j in range(0, width):
                # Iterate over every y-point for given x-value
                for k in range(0, y_pts):
                    j_idx = y_pts*j + k
                    
                    # Save curent x-value und y-value
                    prob_lines[line_idx][0][j_idx] = j
                    prob_lines[line_idx][1][j_idx] = round(m_stg[a]*j + b[i]) + k - delta_b       
                    
                    # Check line entry and exit point
                    if prob_lines[line_idx][1][j_idx] < 0:
                        prob_lines[line_idx][1][j_idx] = 0
                        x_limits[line_idx][0] = j_idx
                    if prob_lines[line_idx][1][j_idx] > height-1:
                        prob_lines[line_idx][1][j_idx] = height[1]-1
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
            print(prob_lines[line_idx][0][x_limits[line_idx][1]], prob_lines[line_idx][0][x_limits[line_idx][0]])                               
                                 
    return prob_lines, num_lines, all_bs, all_ms, delta_b, y_pts, line_lengths, x_limits
    
##############################################################################################################
##############################################################################################################


  # Line probing

def line_detector(frame_filtered, num_lines, prob_lines, x_limits, line_lengths, y_pts, delta_b, all_bs, rescale_factor, frame_raw):
    score = np.empty([num_lines])
    for j in range(0, num_lines):
        score[j] = np.sum(frame_filtered[prob_lines[j][1][x_limits[j][0]:x_limits[j][1]],prob_lines[j][0][x_limits[j][0]:x_limits[j][1]]])/line_lengths[j]           

    # Calc minimum score for a line to be drawn. The score reference is based on the average of the values bewteen the second and fourth quintile.
    q = np.quantile(score, [0.2, 0.4, 0.6, 0.8, 1])
    middle_q = score[(score >= q[1]) & (score <= q[3])]
    score_ref = p.score_thres*np.mean(middle_q)

    score_max_idx = np.argmax(score)

    if score[score_max_idx] >= score_ref:
        print('yeay')

        # Find needle tip
     #   diff_min_x, diff_min_y = find_tip(frame_filtered, prob_lines[score_max_idx], window_size, width, y_pts, delta_b)
        # line
     #   line_b = round(all_bs[score_max_idx]/rescale_factor)
     #   line_m = all_ms[score_max_idx]
     #   line_y, line_x = get_draw_line(line_b, line_m, frame_raw.shape)

        # tip
     #   tip_x = int(round(diff_min_x/rescale_factor))
     #   tip_y = int(round(diff_min_y/rescale_factor))
     #   circle_y, circle_x = draw.disk([tip_y, tip_x], 12)

        # Draw
     #   for t in range(-3, 3):
     #       frame_raw[line_y+t, line_x] = 255
     #   frame_raw[circle_y, circle_x] = 255
     #   print("Found axis and tip in frame", loop_ctr,"with line score of", round(score[score_max_idx],2))
    #else:
     #   print("No needle found")