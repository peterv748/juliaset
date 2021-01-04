"""
 calculation of juliaset set formula
"""
from numba import jit_module

def juliaset_calculate(z_real, z_imag, z_real2, z_imag2, c_imag, c_real, maxiter):
    """
    calculation of julia set formula
    """
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

jit_module(nopython=True)

if __name__ == "__main__":
    REAL = 0.3
    IMAG = 0.5
    MAXIMUM_ITERATIONS = 200
    z_real = -1.0
    z_imag = -1.0
    z_real2 = z_real*z_real
    z_imag2 = z_imag*z_imag

    print(juliaset_calculate(z_real, z_imag, z_real2, z_imag2, IMAG, REAL, MAXIMUM_ITERATIONS))
