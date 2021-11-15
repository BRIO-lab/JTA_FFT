from JTA_FFT import *
import os
import numpy as np

def main():
    ACTIYAS_DIR = "C:/Datasets_FemCleaned/Actiyas/Actiyas_Organized/T01R/Session_1/Cross_1"
    FEM_STL = ACTIYAS_DIR + "/FemurR3_fem.stl"
    FEM_NAME = os.path.splitext(FEM_STL)[0]
    TIB_STL = ACTIYAS_DIR + "/TibiaR2_tib.stl"
    TIB_NAME = os.path.splitext(TIB_STL)[0]
    calibration = ACTIYAS_DIR + "/1024/cal1024.txt"
    
    NFD = JTA_FFT(calibration)
    NFD.Make_Contour_Lib(calibration,FEM_STL,ACTIYAS_DIR)
    NFD.Create_NFD_Library(ACTIYAS_DIR,"fem")
    NFD.save(FEM_NAME)

    NFD2 = JTA_FFT(calibration)
    NFD2.Make_Contour_Lib(calibration, TIB_STL, ACTIYAS_DIR)
    NFD2.Create_NFD_Library(ACTIYAS_DIR, "tib")
    NFD2.save(TIB_NAME)
    

if __name__ == '__main__':
    main()