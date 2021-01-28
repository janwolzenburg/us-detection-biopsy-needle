import numpy as np

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