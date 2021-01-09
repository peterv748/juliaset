"""
 calculation of juliaset set formula
"""
from numba import jit_module
import numpy as np

def juliaset_calculate(cmplx_number_1, cmplx_number_2, cmplx_constant, maxiter):
    """
    calculation of julia set formula
    """
    count = 0
    temp = cmplx_number_2[0] + cmplx_number_2[1]
    while (temp <= 4) and (count <= maxiter):
        cmplx_number_1[1] = 2* cmplx_number_1[0]*cmplx_number_1[1] + cmplx_constant[1]
        cmplx_number_1[0] = cmplx_number_2[0] - cmplx_number_2[1] + cmplx_constant[0]
        cmplx_number_2[0] = cmplx_number_1[0]*cmplx_number_1[0]
        cmplx_number_2[1] = cmplx_number_1[1]*cmplx_number_1[1]
        temp = cmplx_number_2[0] + cmplx_number_2[1]
        count = count + 1
    return int(count)

def juliaset_calculate_gpu(z_real_1, z_imag_1, z_real_2, z_imag_2, const_real, const_imag, maxiter):
    """
    calculation of julia set formula for gpus
    """
    count = 0
    temp = z_real_2 + z_imag_2
    temp_real_1 = z_real_1
    temp_imag_1 = z_imag_1
    temp_real_2 = z_real_2
    temp_imag_2 = z_imag_2
    
    while (temp <= 4) and (count <= maxiter):
        temp_imag_1 = 2* temp_real_1*temp_imag_1 + const_imag
        temp_real_1 = temp_real_2 - temp_imag_2 + const_real
        temp_real_2 = temp_real_1*temp_real_1
        temp_imag_2 = temp_imag_1*temp_imag_1
        temp = temp_real_2 + temp_imag_2
        count = count + 1
    return int(count)

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

    print(juliaset_calculate(complex_number_1, complex_number_2, complex_constant, \
                             MAXIMUM_ITERATIONS))
    