#########################################################################################
#   Author:         Jan Wolzenburg @ Fachhochschule Südwestfalen (Lüdenscheid, Germany) #
#   Date:           17.01.2020                                                          #
#   Version:        2.2                                                                 #
#   Description:    Needle detection in continueous US-image stream                     #
#########################################################################################


################################################
# Modules
import RPi.GPIO as gpio

import math as m
import numpy as np

import cv2

import scipy.ndimage as spimg

import skimage.draw as skdw
import skimage.transform as sktr

import matplotlib.image as matimg
import matplotlib.pyplot as mplpp
import matplotlib as mpl

import functions as f
import parameters as p

import time

print("Modules loaded!");

try:
    expected_angle = p.angles[1]
    loop_ctr = 0
    frame_ptr = 0
    new_angle = 1
    
    # Stup GPIO for holder angles
    gpio.setmode(GPIO.BOARD)
    gpio.setup(p.channels, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    gpio.add_event_detect(p.channels[0], GPIO.FALLING,  bouncetime=200)
    print("Set up input channels")
    
    ################################################
    # Set up capture and picture output

    cap = cv2.VideoCapture(0)
    cap.open(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, p.image_size_x)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p.image_size_y)
    print("Capture device opened!")
    
    while True:
        
        # Angle change button pressed ? 
        if gpio.event_detected(p.channels[0]):
            print("Button pressed!")
            # Check each input
            for c in range(2, len(p.channels)):
                # If input is low the expected angle is updated
                if gpio.input(p.channels[c]) == gpio.LOW:
                    expected_angle = p.angles[c - 1]
                    insertion_depth = p.insertion_depths[c - 1]
                    print("Angle set to ", expected_angle)
                    new_angle = 1
                    break
           
            
        if new_angle == 1:
                
            ################################################
            # Parameters

            # Shape of ROI---------------------------------
            width = p.roi_x_2 - p.roi_x_1; height = p.roi_y_2 - p.roi_y_1;      # width and height

            # Needle Parameters----------------------------
            dist_per_pixel = p.depth/height                                     # Distance in cm per Pixel
            expected_angle = m.pi/2-expected_angle/360*2*m.pi                 # Recalc expected angle so it is measured with respect to the x-axis

            # Resize Parameters----------------------------
            # Calculate new image size 
            if width <= height:
                new_height = p.processing_size;
                rescale_factor = new_height/height
                new_width = round(rescale_factor*width)
            else:
                new_width = p.processing_size;
                rescale_factor = new_width/width
                new_height = round(rescale_factor*height)
                
            width = new_width; height = new_height;
            wdt_hgt_ref = m.sqrt(width**2+height**2)                            # Refernence for some parameters

            # Processing----------------------------------
            # Filter kernel parameters for smoothing and edge improvement
            sigma_x = wdt_hgt_ref/p.sigma_ref_div                                            # Sigma x of gaussian kernel
            sigma_y = sigma_x/p.kernel_aspect_ratio                             # Sigma y
            sob_kernel_size = int(12);                                          # Size of Sobel Kernel

            # Probing lines
            # b -> Pixel value where the needle enters the cropped picture
            expected_b = round(insertion_depth/dist_per_pixel*rescale_factor) # expected pixel value where the needle enters the frame
            b_range = int(height/16)                                            # pixel range for probing lines
            angle_range = 5/360*m.pi*2                                          # angle range for probing lines

            # Tip detection
            window_size = int(wdt_hgt_ref/p.window_size_div)

            print("Parameters set! Width x Height:", width, "x", height, ";\t Expected b, angle:", expected_b, ",", expected_angle*360/m.pi/2, ";\t b, angle range:", b_range, ", ", angle_range*360/m.pi/2)

            ################################################
            # Initialisation

            # Build Gauss filter kernel
            kernel = f.build_gauss_kernel(sigma_x, sigma_y, expected_angle)
            print("Gaussian filter kernel build!")

            # Build rotated Sobel
            sob_kernel = f.build_sobel_kernel(sob_kernel_size, expected_angle);
            print("Sobel filter build!")
            kernel = spimg.convolve(kernel, sob_kernel)                         # Convolve Kernels
            print("Kernels convolved! Size:", kernel.shape[1], " x ", kernel.shape[0])

            # Build probing lines
            prob_lines, num_prob_lines, all_bs, all_ms, delta_b, y_pts, line_lengths, x_limits = f.build_probe_lines(expected_angle, angle_range, p.num_angles, expected_b, b_range, p.num_bs, p.line_wdt, width, height)
            score = np.empty([num_prob_lines])
            print("Probing lines build! Amount:", num_prob_lines)
            
            new_angle = 0;

        else:
            ################################################
            # Repeatet steps

            # Get image and crop to ROI
            ret, frame = cap.read()
            if ret == False:
                print("Video grabber disconnected!")
                exit
    
            ts=time.time();
            #frame = cv2.imread(p.image_path + "/" + str(loop_ctr) + ".png")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame_raw = frame[p.roi_y_1:p.roi_y_2, p.roi_x_1:p.roi_x_2]
            frame_roi = sktr.resize(frame_raw, (height, width))
            
            # Apply filters
            frame_filtered = spimg.convolve(frame_roi, kernel)
            # Normalize
            frame_filtered = f.normal(frame_filtered, np.uint8)
            
            # Line probing
            for j in range(0, num_prob_lines):
                score[j] = np.sum(frame_filtered[prob_lines[j][1][x_limits[j][0]:x_limits[j][1]],prob_lines[j][0][x_limits[j][0]:x_limits[j][1]]])/line_lengths[j]           
            
            # Calc minimum score for a line to be drawn. The score reference is based on the average of the values bewteen the second and fourth quintile.
            q = np.quantile(score, [0.2, 0.4, 0.6, 0.8, 1])
            middle_q = score[(score >= q[1]) & (score <= q[3])]
            score_ref = p.score_thres*np.mean(middle_q)
            
            score_max_idx = np.argmax(score)
            
            if score[score_max_idx] >= score_ref:
            
                # Find needle tip
                diff_min_x, diff_min_y = f.find_tip(frame_filtered, prob_lines[score_max_idx], window_size, width, y_pts, delta_b)

                # line
                line_b = round(all_bs[score_max_idx]/rescale_factor)
                line_m = all_ms[score_max_idx]
                line_y, line_x = f.get_draw_line(line_b, line_m, frame_raw.shape)
                
                # tip
                tip_x = int(round(diff_min_x/rescale_factor))
                tip_y = int(round(diff_min_y/rescale_factor))
                circle_y, circle_x = skdw.disk([tip_y, tip_x], 12)
                          
                # Draw
                for t in range(-3, 3):
                    frame_raw[line_y+t, line_x] = 255
                frame_raw[circle_y, circle_x] = 255

                print("Found axis and tip in frame", loop_ctr,"with line score of", round(score[score_max_idx],2))
            else:
                print("No needle found")
            
            # Save array
            np.save("./saved_frames/"+str(frame_ptr)+".npy", frame_raw)
            
            file = open(p.status_path, 'w+')
            file.seek(0)
            file.write(str(frame_ptr))
            file.close()
            
            if frame_ptr == 15:
                frame_ptr = 0
            else:
                frame_ptr += 1  
        
            print("Elapsed time: ", round(time.time()-ts,2),"s!")
            
            loop_ctr = loop_ctr + 1
except KeyboardInterrupt:
    print("Exiting!")
    #cap.release()
    pass

time.sleep(2)
#cap.release()
