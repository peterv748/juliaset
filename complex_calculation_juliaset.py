"""
 calculation of juliaset set formula
"""
from numba import jit_module
import numpy as np

def juliaset_calculate(complex_number_1, complex_number_2, complex_constant, maxiter):
    """
    calculation of julia set formula
    """
    count = 0
    temp = complex_number_2[0] + complex_number_2[1]
    while (temp <= 4) and (count <= maxiter):
        complex_number_1[1] = 2* complex_number_1[0]*complex_number_1[1] + complex_constant[1]
        complex_number_1[0] = complex_number_2[0] - complex_number_2[1] + complex_constant[0]
        complex_number_2[0] = complex_number_1[0]*complex_number_1[0]
        complex_number_2[1] = complex_number_1[1]*complex_number_1[1]
        temp = complex_number_2[0] + complex_number_2[1]
        count = count + 1
    return count

jit_module(nopython=True)

if __name__ == "__main__":
    REAL = 0.3
    IMAG = 0.5
    complex_constant = np.array([REAL, IMAG], dtype=np.float64)
    MAXIMUM_ITERATIONS = 200
    Z_REAL = -1.0
    Z_IMAG = -1.0
    z_real2 = Z_REAL*Z_REAL
    z_imag2 = Z_IMAG*Z_IMAG
    complex_number_1 = np.array([Z_REAL, Z_IMAG], dtype=np.float64)
    complex_number_2 = np.array([z_real2, z_imag2], dtype=np.float64)

    print(juliaset_calculate(complex_number_1, complex_number_2, complex_constant, MAXIMUM_ITERATIONS))
