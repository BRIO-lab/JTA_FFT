from JTA_FFT import *
import os
import numpy as np

def main():
    GMK_DIR = "C:\Datasets_FemCleaned/Lima/Lima Testing/Patient 77-13-PF/Session_1/Kneel_1"
    FEM_STL = GMK_DIR + "/KR_right_6_fem.stl"
    FEM_NAME = os.path.splitext(FEM_STL)[0]
    TIB_STL = GMK_DIR + "/KR_right_6_tib.stl"
    TIB_NAME = os.path.splitext(TIB_STL)[0]
    calibration = GMK_DIR + "/calibration.txt"
    
    NFD = JTA_FFT(calibration)
    NFD.Make_Contour_Lib(FEM_STL)
    NFD.Create_NFD_Library()
    NFD.save(FEM_NAME)

    NFD2 = JTA_FFT(calibration)
    NFD2.Make_Contour_Lib(TIB_STL)
    NFD2.Create_NFD_Library()
    NFD2.save(TIB_NAME)
    

if __name__ == '__main__':
    main()