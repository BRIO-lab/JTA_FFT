from numba import cuda
from numba import vectorize
import numpy as np
import math
import time

start_time = time.time()
#@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y  # This is Python, so let's return a tuple

#@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)
    
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def polar_distance_array(rho1, theta1, rho2, theta2, n):
    x1_arr = np.zeros((n))
    y1_arr = np.zeros((n))
    x2_arr = np.zeros((n))
    y2_arr = np.zeros((n))
    answers = np.zeros((n))
    
    for i in range(n):
        x1_arr[i], y1_arr[i] = polar_to_cartesian(rho1[i], theta1[i])
        x2_arr[i], y2_arr[i] = polar_to_cartesian(rho2[i], theta2[i])
        answers[i] = ((x1_arr[i] - x2_arr[i])**2 + (y1_arr[i] - y2_arr[i])**2)**0.5
    
    return answers

n = 100000000
rho1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

answers = polar_distance_array(rho1, theta1, rho2, theta2, n)
print("Printing subset of values (its working): ", answers)
print("Time: ", time.time() - start_time)


        



