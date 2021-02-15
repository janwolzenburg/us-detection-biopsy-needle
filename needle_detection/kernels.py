import numpy as np
import astropy.convolution as ascon
from scipy import ndimage, signal
from skimage import draw
import needle_detection.parameters as p
import matplotlib.pyplot as plt
     
def build_gauss_kernel(sigma_x, sigma_y, angle):
    """
    Build the rotated anisotropic gaussian filter kernel

    Parameters
    ----------
    sigma_x : numpy.float64
        sigma in x-direction
    sigma_y: numpy.float64
        sigma in y-direction
    angle: int
         angle in degrees of the needle holder measuered with respect to 'vertical' transducer axis

    Returns
    -------
    kernel: numpy.ndarray
        roteted filter kernel 
    """

    angle = np.pi/2-np.deg2rad(angle)  
    # Calculate gaussian kernel
    kernel = ascon.Gaussian2DKernel(sigma_x, sigma_y, 0)
    # Extract size and kernel values
    x_size = kernel.shape[0]; y_size = kernel.shape[1]
    kernel = kernel.array
    # Rotate
    kernel = ndimage.rotate(kernel,np.rad2deg(-angle), reshape=False)

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


def build_sobel_kernel(n, angle):
    """
    Build the rotated sobel kernel

    Parameters
    ----------
    n: int
        size of sobel kernel. Must be a multiple of three
    angle: int
         angle in degrees of the needle holder measuered with respect to 'vertical' transducer axis

    Returns
    -------
    kernsob_kernel: numpy.ndarray
        sobel filter kernel
    """

    np.pi/2-np.deg2rad(angle)  
    n_3 = int(n/3)
    sob_kernel = np.zeros([n,n])
    sob_kernel[0:n_3,0:n_3] = 1; sob_kernel[0:n_3,n_3:2*n_3] = 2; sob_kernel[0:n_3,2*n_3:3*n_3] = 1
    sob_kernel[2*n_3:3*n_3,0:n_3] = -1; sob_kernel[2*n_3:3*n_3,n_3:2*n_3] = -2; sob_kernel[2*n_3:3*n_3,2*n_3:3*n_3] = -1 
    sob_kernel = ndimage.rotate(sob_kernel, np.rad2deg(-angle), reshape=False)

    return sob_kernel
    

def convolution(array_1, array_2):
    """
    Convolution of two arrays

    Parameters
    ----------
    array_1: numpy.ndarray
        x
    array_2: numpy.ndarray
         x

    Returns
    -------
    convolved_array: numpy.ndarray
        convolved kernel
    """

    convolved_array = ndimage.convolve(array_1, array_2)

    return convolved_array

def filter_kernel_parameters(frame, value=12):
    """
    Finding ilter kernel parameters for smoothing and edge improvement

    Parameters
    ----------
    frame: numpy.ndarray
        x
    value: numpy.ndarray
         x, default 12

    Returns
    -------
     sigma_x : numpy.float64
        sigma in x-direction
    sigma_y: numpy.float64
        sigma in y-direction
    sob_kernel_size: int
        size of sobel kernel. Must be a multiple of three
    """
    wdt_hgt_ref = np.sqrt((np.shape(frame)[0])**2+(np.shape(frame)[1])**2)
    sigma_x = wdt_hgt_ref/75                                            # Sigma x of gaussian kernel
    sigma_y = sigma_x/p.kernel_aspect_ratio                             # Sigma y
    sob_kernel_size = int(value)   

    return sigma_x, sigma_y, sob_kernel_size
