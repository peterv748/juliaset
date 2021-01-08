"""
Julia Set using the GPU
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import draw_juliaset
import complex_calculation_juliaset


@cuda.jit
def julia_numba_cuda_kernel(image_size, cpx_number_1, cpx_number_2, cp_constant, image, maxiter):
    """
    calculation and creation of 2D array with Julia set using GPU
    """

    stepsize = 2.0/image_size[1]

    start_x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    start_y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    grid_x = cuda.gridDim.x * cuda.blockDim.x;
    grid_y = cuda.gridDim.y * cuda.blockDim.y;
  
    
    for j in range(start_y, image_size[1], grid_y):
        for i in range(start_x, image_size[0], grid_x):
            cpx_number_1[0] = -2.0 + i*stepsize
            cpx_number_1[1] = -1.0 + j*stepsize
            cpx_number_2[0] = cpx_number_1[0]*cpx_number_1[0]
            cpx_number_2[1] = cpx_number_1[1]*cpx_number_1[1]
            image[j,i] = complex_calculation_juliaset.juliaset_calculate(cpx_number_1, cpx_number_2, \
                                                                        cp_constant, maxiter)
            
def main():
    """
    main function for calculating and drawing Julia set
    """

    block_dim = (32,32)
    grid_dim = (128,64)

    complex_constant = np.array([-0.837, -0.2321], dtype=np.float64)
    y_size = 2048
    x_size = 2* y_size
    max_iterations = np.int32(300)

    image_rectangle = np.array([-2, 0, 0, 1], \
                                dtype=np.float64)
    image_size = np.array([x_size,y_size], dtype=np.int32)
    image = np.zeros((y_size, x_size), dtype=np.int32)
    image_processed = np.array((y_size, x_size), dtype=np.int32)
    cpx_number_1 = np.array([0.0, 0.0], dtype=np.float64)
    cpx_number_2 = np.array([0.0, 0.0], dtype=np.float64)
    
    #start calculation
    d_image = cuda.to_device(image)
    start = time.time()
    julia_numba_cuda_kernel[grid_dim, block_dim](image_size, cpx_number_1, cpx_number_2, \
                            complex_constant, d_image, max_iterations)
    delta_time = time.time() - start
    image_processed = d_image.copy_to_host()
    
    draw_juliaset.plot_juliaset(image_rectangle,image_size, image_processed, delta_time, max_iterations)

if __name__ == "__main__":
    main()
