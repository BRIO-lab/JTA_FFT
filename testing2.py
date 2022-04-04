import multiprocessing
import numpy as np
import time

def sumall(value):
    return sum(range(1, value + 1))

if __name__ == '__main__':
    
    start_time = time.time()

    my_array = np.full((1, 10), 100)
    
    pool_obj = multiprocessing.Pool()
    answer = pool_obj.map(sumall,range(0,100))
    print(my_array)
    print(time.time() - start_time)
