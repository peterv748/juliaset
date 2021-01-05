"""
 draw julia set
"""

import numpy as np
import time
import draw_juliaset
import complex_calculation_juliaset
import class_juliaset



def main():
#initialization of constants
    c_real = -0.835
    c_imag = - 0.2321
    complex_constant = np.array([-0.835, -0.2321], dtype=np.float64)
    y_size = 2048
    x_size = 2* y_size
    image_box_xmin = -2
    image_box_xmax = 0
    image_box_ymin = 0
    image_box_ymax = 1
    image_array = np.zeros((y_size, x_size), dtype=np.uint32)
    max_iterations = np.int64(300)

    image_rectangle = np.array([image_box_xmin, image_box_xmax, image_box_ymin, image_box_ymax], \
                                dtype=np.float64)
    image_size = np.array([x_size,y_size], dtype=np.int64)
    juliaset = class_juliaset.JuliaSet(image_size, complex_constant, max_iterations)
    start = time.time()
    image_array = juliaset.julia_set_array()
    elapsed_time = time.time() - start

    draw_juliaset.plot_juliaset(image_rectangle, image_size, image_array, elapsed_time, max_iterations)

if __name__ == "__main__":
    main()

