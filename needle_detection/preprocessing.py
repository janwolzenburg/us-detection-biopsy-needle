import numpy as np
#import RPi.GPIO as gpio
import needle_detection.parameters as p
from skimage import transform
import matplotlib.pyplot as plt
import cv2 as cv

#def get_angle_from_pi():
#    gpio.setmode(gpio.BOARD)
#    gpio.setup(p.channels, gpio.IN, pull_up_down=gpio.PUD_UP)
#    gpio.add_event_detect(p.channels[0], gpio.FALLING,  bouncetime=200)
#    print("Set up input channels")
#    while True:
         
     # Angle change button pressed ? 
#     if gpio.event_detected(p.channels[0]):
#         print("Button pressed!")
#         # Check each input
#         for c in range(2, len(p.channels)):
#             # If input is low the expected angle is updated
#             if gpio.input(p.channels[c]) == gpio.LOW:
#                 expected_angle = p.angles[c - 1]
#                 insertion_depth = p.insertion_depths[c - 1] 
#                 print("Angle set to ", expected_angle)
#                 new_angle = 1
#                 break
#    return


#def read_cam(channel=0):
#    cap = cv2.VideoCapture(channel)
#    cap.open(channel)
#    cap.set(cv2.CAP_PROP_FRAME_WIDTH, p.image_size_x)
#    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p.image_size_y)
#    print("Capture device opened!")#

                # Get image and crop to ROI
#            ret, frame = cap.read()
#            if ret == False:
#                print("Video grabber disconnected!")
#                exit
#    return

#uhrzeigersinn, startet oben links
def mask_roi(img, vertices):
    """
    Get roi using a mask

    Parameters
    ----------
    img : numpy.ndarray
        Frame of us image
    vertices: numpy.ndarray
        Vertices characterize the polynomial, defined in parameters.py
    Returns
    -------
    numpy.ndarray
        masked input image
    """
    #blank mask:
    mask = np.zeros_like(img)

    # fill the mask
    #cv.fillPoly(mask, vertices, 255)
    cv.fillConvexPoly(mask, np.array(vertices, 'int32'), 255)
    # now only show the area that is the mask
    masked = cv.bitwise_and(img, mask)

    return masked

def read_frame(path, alpha=0, beta=255):
    """
    Reads in and normalizes single frame in range(0,255)

    Parameters
    ----------
    path : str
        The file location of the frame 

    Returns
    -------
    numpy.ndarray
        normalized gray-scale image
    """

    frame = cv.imread(path, cv.IMREAD_GRAYSCALE)
    norm = cv.normalize(frame, None, alpha, beta, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    return norm


#def roi(frame, vertices):
    #blank mask:
 #   mask = np.zeros_like(img)
    # fill the mask
 #   cv.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
#    masked = cv.bitwise_and(img, mask)
#    return masked


def get_ROI(frame, angle):
    """
    Defines ROI in given frame

    Parameters
    ----------
    frame : numpy.ndarray
    angle: int
         Angel in degrees of the needle holder measuered with respect to 'vertical' transducer axis

    Returns
    -------
    frame_roi : numpy.ndarray
        ROI of given frame 
    rescale_factor: float
        xxx
    """

    width = p.roi_x_2 - p.roi_x_1
    height = p.roi_y_2 - p.roi_y_1

    dist_per_pixel = p.depth/height     # Distance in cm per Pixel
    angle = np.pi/2-np.deg2rad(angle)   # Recalc expected angle so it is measured with respect to the x-axis   
    
    # Calculate new image size 
    if width <= height:
        new_height = p.processing_size
        rescale_factor = new_height/height
        new_width = round(rescale_factor*width)
    else:
        new_width = p.processing_size
        rescale_factor = new_width/width
        new_height = round(rescale_factor*height)
    width = new_width
    height = new_height

    wdt_hgt_ref = np.sqrt(width**2+height**2) 
    frame_raw = frame[p.roi_y_1:p.roi_y_2, p.roi_x_1:p.roi_x_2]
    frame_roi = transform.resize(frame_raw, (height, width))
    
    return frame_roi, rescale_factor