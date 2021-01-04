"""
    mandelbrot class
"""
import numpy as np
from numba.experimental import jitclass
from numba import int64, float64
import complex_calculation_juliaset

spec = [('size_x_axis', int64),('size_y_axis',int64), \
        ('c_real', float64),('c_imag', float64), \
        ('max_iterations',int64)]
@jitclass(spec)

Class JuliaSet():
"""
    mandelbrot class
"""

def __init__(self, x_size, y_size, c_real, c_imag, max_iter):
    """
    init of class variables
    """
    self.size_x_axis = x_size
    self.size_y_axis = y_size
    self.c_real = c_real
    self.c_imag = c_imag
    self.max_iterations = max_iter

def julia_set(self):
    """
    calculate 2D image matrix
    """
    
    h = 2.0/self.size_y_axis
    for y in range(self.size_y_axis):
        for x in range(size_x_axis):
            z_real = -2.0 + x*h
            z_imag = -1.0 + y*h
            image[y,x] = 0
            z_real2 = z_real*z_real
            z_imag2 = z_imag*z_imag
            image[y,x] = complex_calculation_juliaset.juliaset_calculate(z_real,\
                         z_imag, z_real2, z_imag2, c_imag, c_real, maxiter)
    return image