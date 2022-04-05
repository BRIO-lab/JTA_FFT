import numpy as np
import pygpufit.gpufit as gf

x = [0,1,2,3,4,5]
y = [4,5,6,7,8,9]

arr = np.atleast_1d([x,y])
print(arr)

print(arr.shape)

print(gf.get_cuda_version())