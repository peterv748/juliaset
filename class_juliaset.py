"""
    Julia set
"""
import numpy as np
from numba.experimental import jitclass
from numba import int64, float64
import complex_calculation_juliaset

spec = [('size_x_axis', int64),('size_y_axis',int64), \
        ('c_real', float64),('c_imag', float64), \
        ('max_iterations',int64)]
@jitclass(spec)
class JuliaSet():
    """
    JUliaset class
    """

    def __init__(self, image_size, complex_constant, max_iter):
        """
        init of class variables
        """
        self.size_x_axis = image_size[0]
        self.size_y_axis = image_size[1]
        self.c_real = complex_constant[0]
        self.c_imag = complex_constant[1]
        self.max_iterations = max_iter

    def julia_set_array(self):
        """
        calculate 2D image matrix containing the image of the juliaset
        """
        image = np.zeros((self.size_y_axis, self.size_x_axis))
        h = 2.0/self.size_y_axis
        for y in range(self.size_y_axis):
            for x in range(self.size_x_axis):
                z_real = -2.0 + x*h
                z_imag = -1.0 + y*h
                z_real2 = z_real*z_real
                z_imag2 = z_imag*z_imag
                image[y,x] = complex_calculation_juliaset.juliaset_calculate(z_real,\
                         z_imag, z_real2, z_imag2, self.c_imag, self.c_real, self.max_iterations)
        return image
