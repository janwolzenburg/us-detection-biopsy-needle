#########################################################################################
#   Author:         Jan Wolzenburg @ Fachhochschule Südwestfalen (Lüdenscheid, Germany) #
#   Date:           01.12.2020                                                          #
#   Version:        2.3                                                                 #
#   Description:    Needle detection in continueous US-image stream                     #
#########################################################################################

import math as m
import numpy as np

import astropy.convolution as ascon

import scipy.ndimage as spimg
import scipy.signal as spsi

import skimage.draw as skdw

##############################################################################################################
##############################################################################################################

"""
description
    
    
arguments
    

returns
    
"""
def build_gauss_kernel(sigma_x, sigma_y, angle):
    
    kernel = ascon.Gaussian2DKernel(sigma_x, sigma_y, 0)
    x_size = kernel.shape[0]; y_size = kernel.shape[1]
    kernel = kernel.array
    kernel = spimg.rotate(kernel, -angle/(2*m.pi)*360, reshape=False)

    max_in_kernel = np.amax(abs(kernel))
    threshold = 0.05*max_in_kernel 

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
    
    
arguments
    

returns
    
"""
def build_sobel_kernel(n, angle):
    
    n_3 = int(n/3)
    sob_kernel = np.zeros([n,n])
    sob_kernel[0:n_3,0:n_3] = 1; sob_kernel[0:n_3,n_3:2*n_3] = 2; sob_kernel[0:n_3,2*n_3:3*n_3] = 1
    sob_kernel[2*n_3:3*n_3,0:n_3] = -1; sob_kernel[2*n_3:3*n_3,n_3:2*n_3] = -2; sob_kernel[2*n_3:3*n_3,2*n_3:3*n_3] = -1 
    sob_kernel = spimg.rotate(sob_kernel, -angle/(2*m.pi)*360, reshape=False)

    return sob_kernel
    
##############################################################################################################
##############################################################################################################

"""
description
    
    
arguments
    

returns
    
"""
def build_probe_lines(expexted_angle, angle_range, num_angles, expected_b, b_range, num_bs, line_wdt, frame_width, frame_height):
    
    m_stg = np.linspace(m.tan(expexted_angle-angle_range),m.tan(expexted_angle+angle_range), num_angles)
    b = np.linspace(expected_b-b_range, expected_b+b_range, num_bs, dtype=int)
    delta_b = m.floor(line_wdt/2/m.cos(2*m.pi*expexted_angle/360))
    y_pts = (2*delta_b+1)
    
    score = np.empty([len(b)*len(m_stg)])
    prob_lines = np.empty([len(b)*len(m_stg),2, (2*delta_b+1)*frame_width], int)
    all_bs = np.empty([len(b)*len(m_stg)])
    all_ms =np.empty([len(b)*len(m_stg)])

    for i in range(0, len(b)):
        for a in range(0, len(m_stg)):
            line_idx = a + len(m_stg)*i
            for j in range(0, frame_width):
                for k in range(0, y_pts):
                    j_idx = y_pts*j + k
                    prob_lines[line_idx][0][j_idx] = j
                    prob_lines[line_idx][1][j_idx] = round(m_stg[a]*j + b[i]) + k - delta_b       
                    
                    if prob_lines[line_idx][1][j_idx] < 0:
                        prob_lines[line_idx][1][j_idx] = 0
                    if prob_lines[line_idx][1][j_idx] > frame_height-1:
                        prob_lines[line_idx][1][j_idx] = frame_height-1

                    all_bs[line_idx] = b[i]
                    all_ms[line_idx] = m_stg[a]
    
    return prob_lines, len(b)*len(m_stg), all_bs, all_ms, delta_b, y_pts
    
##############################################################################################################
##############################################################################################################

"""
description
    
    
arguments
    

returns
    
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
    
    
arguments
    

returns
    
"""
def find_tip(frame_filtered, prob_line, n, width, y_pts, delta_b):
    
    intensity_along_line = np.zeros([width], float)
    for k in range(n, width-n):
        i_start = y_pts*(k-n)
        i_end = y_pts*(k+n) + 2*delta_b
        intensity_along_line[k] = np.sum(frame_filtered[prob_line[1][i_start:i_end], prob_line[0][i_start:i_end]])

    m = 2*n+1
    intensity_along_line = spsi.convolve(intensity_along_line, np.full([m],1/m,dtype=float),mode='same')
        
    if m > n:
            n = m
        
    for l in range(0, n):
        intensity_along_line[l] = intensity_along_line[n]  
    for l in range(width-n-1, width):
       intensity_along_line[l] = intensity_along_line[width-n-1]  
        
    difference = np.diff(intensity_along_line, 1)
    intensity_along_line_diff = np.append(difference[0], difference)

    diff_min_idx = np.argmin(intensity_along_line_diff)

    diff_min_x = prob_line[0][y_pts*diff_min_idx+delta_b]
    diff_min_y = prob_line[1][y_pts*diff_min_idx+delta_b]
    
    return diff_min_x, diff_min_y, intensity_along_line, intensity_along_line_diff
    
##############################################################################################################
##############################################################################################################

"""
description
    
    
arguments
    

returns
    
"""
def get_draw_line(line_b, line_m, shape):

    if line_b < 0:
        x_start = int(round(-line_b/line_m))
        y_start = int(0)
    else:
        x_start = int(0)
        y_start = int(line_b)

    if line_m*(shape[1]-x_start) + y_start > shape[0]:
        y_end = shape[0]-1
        x_end = int(round((y_end - y_start)/line_m + x_start))
    else:
        x_end = shape[1]-1
        y_end = int(round(line_m*(x_end-x_start) + y_start))
    
    line_y, line_x = skdw.line(y_start, x_start, y_end, x_end)
    
    return line_y, line_x

##############################################################################################################
##############################################################################################################

"""
description
    draw a dashed line in an image
    
arguments
    image: the image
    pt1, pt2: start and endpoint
    color: line color
    thickness: line thickness
    gap: gap between segments
    
returns
    

Taken from: https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines (viewed on 28. Oct. 2020)
"""
def drawline(img,pt1,pt2,color,thickness=1, gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    s=pts[0]
    e=pts[0]
    i=0
    for p in pts:
        s=e
        e=p
        if i%2==1:
            #cv.line(img,s,e,color,thickness,cv.LINE_AA)
            b = 0
        i+=1
    return
