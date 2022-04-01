import multiprocessing

def sumall(value):
    return sum(range(1, value + 1))

pool_obj = multiprocessing.Pool()

answer = pool_obj.map(sumall,range(0,5))
print(answer)