#########################################################################################
#   Author:         Jan Wolzenburg @ Fachhochschule Südwestfalen (Lüdenscheid, Germany) #
#   Date:           12.01.2020                                                          #
#   Version:        2.1                                                                 #
#   Description:    Needle detection in continueous US-image stream                     #
#########################################################################################

# Image properties
#image_path = './resources/single_frames/';                  
#frames = ['40.png','35.png','45.png','63.png','69.png','74.png']
status_path = "./saved_frames/status"

image_size_x = 960; image_size_y = 1280                     # size of grabbed frames
roi_x_1 = 368; roi_x_2 = 1015                               # Region of interest in grabbed frame
roi_y_1 = 111; roi_y_2 = 701
depth = 3.5                                                 # Depth of picture in cm. Physical length covered by the ROIs height


# Needle -> processing only valid for angles which enter the picture at x = 0 and has positive incline (y value rises with x value)
expected_angle = 60                                         # expected_angle of the kernel rotation with respect to y-axis
angles =            [   20, 30, 40, 50, 60, 70]           # Angels of the needle holder measuered with respect to 'vertical' transducer axis
insertion_depths =  [    1,  1,  1,  1,  1,  1]           # Depth (in cm) at which the insterted needle appears in the image. May be depending on the selected angle
channels =          [7, 11, 12, 13, 15, 16, 18]           # Channels of the 'angle-change' button and conducting loops


# Processing
processing_size = 200                                       # Size of processed image. ROI will be cropped to match this size
kernel_aspect_ratio = 4                                     # Ratio of the two gaussian kernel sides
sigma_ref_div = 50

# probing lines                                        
num_bs = 8                                                  # amount of bs in probing lines  
num_angles = 8                                              # amount of angles in probing lines
line_wdt = 2                                                # width of the probing lines

score_thres = 1.2                                           # Threshold factor for line scoring. Decides wether a line is prominent enough to be considered  the needle

# tip detection
window_size_div = 50                                        # divider for side length of moving window
