import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw

"""
description
    Build the rotated anisotropic gaussian filter kernel
    
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
    kernel = ndimage.rotate(kernel, -angle/(2*np.pi)*360, reshape=False)

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
    sob_kernel = ndimage.rotate(sob_kernel, -angle/(2*np.pi)*360, reshape=False)

    return sob_kernel
    
##############################################################################################################
##############################################################################################################
