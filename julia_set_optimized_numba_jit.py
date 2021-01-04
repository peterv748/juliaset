import numpy as np
import time
from numba import jit
import draw_juliaset

@jit
def julia_calculate(z_real, z_imag, z_real2, z_imag2, c_imag, c_real, maxiter):
    n = 0
    temp = z_real2 + z_imag2
    while (temp <= 4) and (n <= maxiter):
          z_imag = 2* z_real*z_imag + c_imag
          z_real = z_real2 - z_imag2 + c_real
          z_real2 = z_real*z_real
          z_imag2 = z_imag*z_imag
          temp = z_real2 + z_imag2
          n = n + 1
    
    return n


@jit
def julia_numba(x_size, y_size, c_real, c_imag, image, maxiter):
    
    h = 2.0/y_size
    for y in range(y_size):
        for x in range(x_size):
            z_real = -2.0 + x*h
            z_imag = -1.0 + y*h          
            image[y,x] = 0
            z_real2 = z_real*z_real
            z_imag2 = z_imag*z_imag
            image[y,x] = julia_calculate(z_real, z_imag, z_real2, z_imag2, c_imag, c_real, maxiter)


def main():
    #initialization of constants
    c_real = -0.835
    c_imag = - 0.2321
    y_size = 2048
    x_size = 2* y_size
    image_box_xmin = -2
    image_box_xmax = 0
    image_box_ymin = 0
    image_box_ymax = 1
    image_array = np.zeros((y_size, x_size), dtype=np.uint32)
    max_iterations = np.int64(300)
    image_rectangle = np.array([image_box_xmin, image_box_xmax, image_box_ymin, image_box_ymax], dtype=np.float64)
    image_size = np.array([x_size,y_size], dtype=np.int64)

    start = time.time()
    julia_numba(x_size, y_size, c_real, c_imag, image_array, max_iterations)
    elapsed_time = time.time() - start
    
    draw_juliaset.plot_juliaset(image_rectangle, image_size, image_array, elapsed_time, max_iterations)

if __name__ == "__main__":
    main()
