from numba import cuda
import numba
import numpy as np   
import time


xrotmax = 30
xrotinc = 3
yrotmax = 30
yrotinc = 3
imsize = 1024  # Image Size
    
# x_val and y_val are 1-D arrays
@cuda.jit(device = True)
def create_NFD_from_contour(x_vals, y_vals):
    big_number = 10000          # To increase runtime
    for i in range(big_number):
        centroid = np.sum(x_vals) + np.sum(y_vals)  # Dummy variable (scalar)
        magnitude = np.sum(x_vals) - np.sum(y_vals) # Dummy (scalar)
        
    angle = xrotmax    # Dummy (scalar)
    NFD = x_vals            # Dummy (1-D array)
    m_k = 10                # Dummy (scalar)
    
    return centroid, magnitude, angle, NFD, m_k

@numba.jit()
def create_nfd_library():
    # define our rotation parameters
    xrot = np.linspace(int(-1*xrotmax),
                        int(xrotmax),
                        int((2*xrotmax/xrotinc))+1)

    yrot = np.linspace(int(-1*yrotmax),
                        int(yrotmax),
                        int((2*yrotmax/yrotinc))+1)

    rot_indices = np.ones((xrot.size, yrot.size, 2))    # SPECIFY ARGUMENTS AS TUPLE NOT LIST, numba specific

    # j will be indexes from 0 -> the amount of elements in xrot
    # xr will be the actual elements from -30 to 20 in increments of 3
    # same for y
    for j in range(xrot.size):
        for k in range(yrot.size):
            rot_indices[j,k,0] = xrot[j]
            rot_indices[j,k,1] = yrot[k]
            
            # Define xval and yval (length of 128)
            x_vals = np.random.rand(1, 128)*100
            y_vals = np.random.rand(1, 128)*100
            
            cent,mag,ang,nfd,mk = create_NFD_from_contour(x_vals, y_vals)

start_time = time.time()
create_nfd_library()
print("Time: ", time.time() - start_time)
