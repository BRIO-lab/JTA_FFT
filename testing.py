import numpy as np
import os
import glob
from JTA_FFT import JTA_FFT
import cv2
"""
    11/2 Added functions for Segmentation, saving data into a pickle, and loading from that pickle.

    To use, create an instance of JTA_FFT and pass in the file path with the desired calibration file.
    i.e. FFT = JTA_FFT("$filepath_here")
    From there, you can access any of the functions in JTA_FFT:
        Segment
        Make_Contour_Lib
        Create_NFD_Library
        create_contour
        get_NFD
        estimate_pose
        load_pickle
        save
"""
FFT = JTA_FFT("lima_files/calibration.txt")

FFT.Segment("ALBUMENTATIONS_HPG_210826_fem_allData_2.pth", "HL-T4-0218.tif")
FFT.create_contour(FFT.outputImg)
FFT.get_NFD(FFT.x_new, FFT.y_new)
FFT.save("Output/output_data.nfd")

# Create a new object to check if the created pickle from the previous FFT object can be loaded in and its values stored in the new object.

blank_NFD = JTA_FFT("lima_files/calibration.txt")
blank_NFD.load_pickle("Output/output_data.nfd")

print(blank_NFD.NFD_library)