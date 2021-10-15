import numpy as np
import os
import glob
from JTA_FFT import JTA_FFT
import cv2




FFT = JTA_FFT("lima_files/calibration.txt")

FFT.save("this_is_a_file_that_wont_BE_CREATED.fft")

# print(FFT.angle_library)

print(FFT.params['pd'])
# FFT.NFD_library

# new_FFT = JTA_FFT.load("path/to/fftdoc")


