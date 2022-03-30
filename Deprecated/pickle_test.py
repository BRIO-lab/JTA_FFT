import pickle
import numpy as np

dogs_dict = { 'Ozzy': np.array([[1,1],[2,2],[3,3]]), 'Filou': 8, 'Luna': 5, 'Skippy': 10, 'Barco': 12, 'Balou': 9, 'Laika': 16 }
outfile = open('dogs.fft', 'wb')
pickle.dump(dogs_dict, outfile)
outfile.close()

infile = open('testing_saving_lalalalallalallalalalalla.fft', 'rb')
new_dict = pickle.load(infile)
infile.close()

print(new_dict.sc)