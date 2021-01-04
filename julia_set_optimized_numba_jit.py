import numpy as np
import time
import matplotlib.pyplot as plt
from numba import jit

@jit
def julia_calculate(zreal, zimag, zreal2, zimag2, cimag, creal, maxiter):
    n = 0
    temp = zreal2 + zimag2
    while (temp <= 4) and (n <= maxiter):
          zimag = 2* zreal*zimag + cimag
          zreal = zreal2 - zimag2 + creal
          zreal2 = zreal*zreal
          zimag2 = zimag*zimag
          temp = zreal2 + zimag2
          n = n + 1
    
    return n


@jit
def julia_numba(Y_size, X_size, creal, cimag, image, maxiter):
    
    h = 2.0/Y_size
    for y in range(Y_size):
        for x in range(X_size):
            zreal = -2.0 + x*h
            zimag = -1.0 + y*h          
            image[y,x] = 0
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            image[y,x] = julia_calculate(zreal, zimag, zreal2, zimag2, cimag, creal, maxiter)


#initialization of constants
creal = -0.835
cimag = - 0.2321
Y_size = 2048
X_size = 2* Y_size
Z = np.zeros((Y_size, X_size), dtype=np.uint32)
maxiter = 300

#start calculation
start = time.time()
julia_numba(Y_size, X_size, creal, cimag, Z, maxiter)
dt = time.time() - start



#plot image in window
plt.imshow(Z, cmap = plt.cm.prism)
plt.xlabel("Re(c), using numba jit processing time: %f s" % dt)
plt.ylabel("Im(c)")
plt.title("julia set, image size (y,x): 2048 x 4096 pixels")
plt.savefig("julia_set_optimize_numba_jit.png")
plt.show()
plt.close()

