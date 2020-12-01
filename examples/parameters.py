# Image properties
image_path = './resources/single_frames/';
frames = ['40.png','35.png','45.png','63.png','69.png','74.png']
image_size_x = 960; image_size_y = 1280                     # for pre-allocation
roi_x_1 = 368; roi_x_2 = 1015                               # Region of interest
roi_y_1 = 111; roi_y_2 = 701
depth = 3.5                                                 # Depth of picture in cm


# Needle
insertion_depth = 1                                         # Depth (in cm) at which the insterted needle appears in the image. Is constant for one physical setup (transducer, needle holder and insertion expected_angle)
expected_angle = 60                                      # expected_angle of the kernel rotation with respect to x-axis measured clockwise


# Processing
processing_size = 200                                       # Size of processed image
kernel_aspect_ratio = 3                                     # Ratio of the two kernel sides


# probing lines                                        
num_bs = 8                                                  # bs of probing lines  
num_angles = 8                                              # angles of probing lines
line_wdt = 2