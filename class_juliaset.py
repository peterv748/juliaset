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
        stepsize = 2.0/self.size_y_axis
        complex_constant = np.array([self.c_real, self.c_imag], dtype=np.float64)
        complex_number_1 = np.array([0.0, 0.0], dtype=np.float64)
        complex_number_2 = np.array([0.0, 0.0], dtype=np.float64)
        for j in range(self.size_y_axis):
            for i in range(self.size_x_axis):
                complex_number_1[0] = -2.0 + i*stepsize
                complex_number_1[1] = -1.0 + j*stepsize
                complex_number_2[0] = complex_number_1[0]*complex_number_1[0]
                complex_number_2[1] = complex_number_1[1]*complex_number_1[1]
                image[j,i] = complex_calculation_juliaset.juliaset_calculate(complex_number_1, \
                         complex_number_2, complex_constant, self.max_iterations)
        return image

    def julia_size (self):
        """
        to satisfy Py-lint
        """
        return self.size_x_axis, self.size_y_axis, self.c_real, self.c_imag
