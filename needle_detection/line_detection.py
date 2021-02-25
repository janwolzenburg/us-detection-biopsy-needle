import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw
import needle_detection.parameters as p
from needle_detection.plotting import get_draw_line
from needle_detection.tip_detection import find_tip
import time
import cv2 as cv
def build_probe_lines(frame, angle, rescale_factor):
    """
    Build probing lines for needle detection

    Parameters
    ----------
    frame : numpy.ndarray
        filtered ROI of original frame
    rescale_factor: float
       x
    angle: int
         angle in degrees of the needle holder measuered with respect to 'vertical' transducer axis

    Returns
    -------
    prob_lines: numpy.ndarray #(64, 2, 600) 
        the multidimensional array that holds the lines. Indexing is: [line number][x(0) or y(1) value][pixel index]
    num_lines: int 
        amount of lines
    all_bs: numpy.ndarray #(64,)
        array that holds all y-intersects
    all_ms: numpy.ndarray #(64,)
        array that holds all inclines
    delta_b: int
        projection of half line width to the y-axis
    y_pts: int
        number of pixels in projection of line width
    line_lengths: numpy.ndarray #(64,)
        lengths of all lines
    x_limits: numpy.ndarray #(64, 2)
        limits where the line enters and exits the ...
    """


    # Probing lines
    # b -> Pixel value where the needle enters the picture
    expected_angle = np.pi/2-np.deg2rad(angle) 
    angle_range = np.deg2rad(p.angle_range)
    width = int(np.shape(frame)[1])##das hier nochmal checken
    height = int(np.shape(frame)[0]) ##das hier nochmal checken
    dist_per_pixel = p.depth/height   
    expected_b = round(p.insertion_depths[0]/dist_per_pixel*rescale_factor)
    b_range = int(height/16) 
    angle_range = np.deg2rad(p.angle_range)
    m_stg = np.linspace(np.tan(expected_angle-angle_range),np.tan(expected_angle+angle_range), p.num_angles)
    b = np.linspace(expected_b-b_range, expected_b+b_range, p.num_bs, dtype=int)

   # delta_b = int(abs(np.floor((p.line_wdt/2)/np.cos(expected_angle)))) #delta sollte positiv sein deswegen mal abs
    delta_b = int(np.floor((p.line_wdt/2)/np.cos(expected_angle)))
    
    y_pts = (2*delta_b+1)

    num_lines = len(b)*len(m_stg)
    score = np.empty([num_lines])

    prob_lines = np.empty(shape=[num_lines,2 ,y_pts*width], dtype=int) 
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
                        prob_lines[line_idx][1][j_idx] = height-1
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
    


def line_detector(frame_filtered, num_lines, prob_lines, x_limits, line_lengths, y_pts, delta_b, rescale_factor, frame_raw, all_bs, all_ms):
    """
    Build probing lines for needle detection

    Parameters
    ----------
    frame_filtered : numpy.ndarray
        filtered ROI of original frame
    frame_raw : numpy.ndarray
        original frame
    prob_lines: numpy.ndarray #(64, 2, 600) 
        the multidimensional array that holds the lines. Indexing is: [line number][x(0) or y(1) value][pixel index]
    num_lines: int 
        amount of lines
    all_bs: numpy.ndarray #(64,)
        array that holds all y-intersects
    all_ms: numpy.ndarray #(64,)
        array that holds all inclines
    delta_b: int
        projection of half line width to the y-axis
    y_pts: int
        number of pixels in projection of line width
    line_lengths: numpy.ndarray #(64,)
        lengths of all lines
    x_limits: numpy.ndarray #(64, 2)
        limits where the line enters and exits the ...
    rescale_factor: 

    Returns
    -------
    line_b: int
        the multidimensional array that holds the lines. Indexing is: [line number][x(0) or y(1) value][pixel index]
    line_m: numpy.float64 
        amount of lines
    line_x: numpy.ndarray 
        array that holds all y-intersects
    line_y: numpy.ndarray 
        array that holds all inclines
    tip_x: int
        projection of half line width to the y-axis
    tip_y: int
        number of pixels in projection of line width
    intensity_along_line: numpy.ndarray
        lengths of all lines
    intensity_along_line_diff: numpy.ndarray 
        limits where the line enters and exits the ...
    diff_min_x: numpy.int64
        lengths of all lines
    diff_min_y: numpy.int64
        limits where the line enters and exits the ...
    """

    score  = [np.sum(frame_filtered[prob_lines[j][1][x_limits[j][0]:x_limits[j][1]], prob_lines[j][0][x_limits[j][0]:x_limits[j][1]]])/line_lengths[j]  for j in range(0, num_lines)]
    score = np.asarray(score)

    q = np.quantile(score, [0.2, 0.4, 0.6, 0.8, 1])
    middle_q = score[(score >= q[1]) & (score <= q[3])]
    score_ref = p.score_thres*np.mean(middle_q)-20

    score_max_idx = np.argmax(score)

    if score[score_max_idx] >= score_ref-50:
        
        # Find needle tip
        diff_min_x, diff_min_y, intensity_along_line, intensity_along_line_diff= find_tip(frame_filtered, prob_lines[score_max_idx], y_pts, delta_b)
        # line
        line_b = round(all_bs[score_max_idx]/rescale_factor)
        line_m = all_ms[score_max_idx]
        line_y, line_x = get_draw_line(line_b, line_m, frame_raw.shape)

        # tip
        tip_x = int(round(diff_min_x/rescale_factor))
        tip_y = int(round(diff_min_y/rescale_factor))   

    return line_b, line_m, line_x, line_y, tip_x, tip_y, intensity_along_line, intensity_along_line_diff, diff_min_x, diff_min_y
    #else:
     #   print("No needle found")

     
def find_houghlinesP(frame):
    """ 
    Build probing lines for needle detection

    Parameters
    ----------
    frame : numpy.ndarray
        Image after preprocessing and edge detection
    threshold: int
        ..
    maxLineGap: int
        maximum allowed distance between two line segments to be recognized as one line
    minLineLength: int
        Minimum length that the line must have in order to be recognized as a line.
    Returns
    -------
    lines: numpy.ndarray
        Array holding starting and ending points of all detected line segments [x1, y1, x2, y2]

    """

    thresL = 100
    minL = 50
    maxL = 50

    lines = cv.HoughLinesP(frame, 1, np.pi/360, threshold=thresL, maxLineGap=maxL, minLineLength=minL)
    #lines = cv.HoughLinesP(image, rho, theta, threshold, maxLineGap, minLineLength)
    return lines

  
#def find_houghlines(frame):
    """ 
    Build probing lines for needle detection

    Parameters
    ----------
    frame : numpy.ndarray
        Image after preprocessing and edge detection
    threshold: int
        ..

    Returns
    -------
    lines: numpy.ndarray
        Array holding starting and ending points of all detected line segments [x1, y1, x2, y2]

    """


#    lines = cv.HoughLines(frame, 1,  np.pi/360, threshold = int(3*np.mean(frame)))
    #lines = cv.HoughLinesP(image, rho, theta, threshold, maxLineGap, minLineLength)
#    print(lines)
#    return lines

def average_houghlines(lines):
    x = []
    y = []
    m = []
    b = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if 105 < np.rad2deg(theta) < 110:
                m0 = np.cos(theta)
                b0 = np.sin(theta)
                x0 = m0 * rho
                y0 = b0 * rho
                x.append(x0)
                y.append(y0)
                m.append(m0)
                b.append(b0)
        xx = np.mean(x)
        yy = np.mean(y)
        mm = np.mean(m)
        bb = np.mean(b)
        pt1 = (int(xx + 1000*(-bb)), int(yy + 1000*(mm)))
        pt2 = (int(xx - 1000*(-bb)), int(yy - 1000*(mm)))   
    return pt1, pt2


    