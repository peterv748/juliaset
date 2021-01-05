"""
 draw julia set
"""
import time
import numpy as np
import draw_juliaset
import class_juliaset



def main():
    """
    main module
    """
    
    complex_constant = np.array([-0.835, -0.2321], dtype=np.float64)
    y_size = 2048
    x_size = 2* y_size
    image_array = np.zeros((y_size, x_size), dtype=np.uint32)
    max_iterations = np.int64(300)

    image_rectangle = np.array([-2, 0, 0, 1], \
                                dtype=np.float64)
    image_size = np.array([x_size,y_size], dtype=np.int64)
    juliaset = class_juliaset.JuliaSet(image_size, complex_constant, max_iterations)
    start = time.time()
    image_array = juliaset.julia_set_array()
    elapsed_time = time.time() - start

    draw_juliaset.plot_juliaset(image_rectangle, image_size, image_array, \
                                elapsed_time, max_iterations)

if __name__ == "__main__":
    main()
