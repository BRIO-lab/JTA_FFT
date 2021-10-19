import numpy as np
import os
import glob
from JTA_FFT import JTA_FFT
import cv2

FFT = JTA_FFT("lima_files/calibration.txt")

"""
    Testing right now: passing in a NN Model with an unprocessed image, 
     passing it through the NN Model, then running create contour, get_NFD, 
     and estimate_pose. Once this works, we can save it to a file to double 
     check that that function works as well.
"""

FFT.Segment("ALBUMENTATIONS_HPG_210826_fem_allData_2.pth", "HL-T4-0218.tif", True)
# FFT.create_contour()

FFT.save("output_data.fft")
# print(FFT.angle_library)
# print(FFT.params['pd'])
# FFT.NFD_library

# new_FFT = JTA_FFT.load("path/to/fftdoc")


