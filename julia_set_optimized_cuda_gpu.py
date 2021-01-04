import numpy as np
import time
import matplotlib.pyplot as plt
from numba import cuda

@cuda.jit (device=True)
def julia_calculate(zreal, zimag, zreal2, zimag2, cimag, creal, maxiter):
    temp = zreal2 + zimag2
    n = 0
    while (temp <= 4) and (n <= maxiter):
          zimag = 2* zreal*zimag + cimag
          zreal = zreal2 - zimag2 + creal
          zreal2 = zreal*zreal
          zimag2 = zimag*zimag
          temp = zreal2 + zimag2
          n = n + 1
    
    return n

@cuda.jit
def julia_numba_cuda_kernel(Y_size, X_size, creal, cimag, image, maxiter):
    
    
    h = 2.0/Y_size
    startX, startY = cuda.grid(2)
    startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for y in range(startY, Y_size, gridY):     
        for x in range(startX, X_size, gridX):
            zreal = -2.0 + x*h
            zimag = -1.0 + y*h
            image[y,x] = 0
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            image[y,x] = julia_calculate(zreal, zimag, zreal2, zimag2, cimag, creal, maxiter)      

#initialization of constants
creal = -0.837
cimag = - 0.2321
Y_size = 2048
X_size = 2 * Y_size
Z= np.zeros((Y_size, X_size), dtype=np.uint32)
maxiter = 300
blockdim = (32,32)
griddim = (32,16)


#start calculation
d_image = cuda.to_device(Z)
start = time.time()
julia_numba_cuda_kernel[griddim, blockdim](Y_size, X_size, creal, cimag, d_image, maxiter)
dt = time.time() - start
A = d_image.copy_to_host()


#plot image in window
plt.imshow(A, cmap = plt.cm.prism)
plt.xlabel("Re(c), using numba cuda gpu processing time: %f s" % dt)
plt.ylabel("Im(c)")
plt.title("julia set, image size (y,x): 2048 x 4096 pixels")
plt.savefig("julia_set_optimize_cuda_gpu.png")
plt.show()
plt.close()
