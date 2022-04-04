import multiprocessing
import numpy as np
import time

def sumall(value):
    return sum(range(1, value + 1))

if __name__ == '__main__':
    test_runs = 10000000
    
    # test multiprocessing running in parallel
    start_time = time.time()
    my_array = []
    my_array = [100 for i in range(test_runs)]
    pool_obj = multiprocessing.Pool()
    answer = pool_obj.map(sumall,my_array)
    print("The time for parallel", time.time() - start_time)
    
    # Run sequentially with a for loop
    start_time2 = time.time()
    my_array2 = []
    my_array2 = [100 for i in range(test_runs)]
    for i in my_array2:
        answer2 = sumall(i)
    print("The time for for loop: ", time.time() - start_time2)

    