from JTA_FFT import *
import os
import glob
import numpy as np

def main():
    DATA_DIR = "C:/Datasets_FemCleaned/Lima/Lima_Organized_Updated/"
    NFD_DIR = "C:/Datasets_FemCleaned/Lima/NFD Library/lib/"

    for pat_id in os.listdir(DATA_DIR):
        
        PAT_DIR = DATA_DIR + "/" + pat_id
        for sess_id in os.listdir(PAT_DIR):
            SESS_DIR = PAT_DIR + "/" + sess_id

            for mvt_id in os.listdir(SESS_DIR):
                MVT_DIR = SESS_DIR + "/" + mvt_id

                cal = MVT_DIR + "/calibration.txt"

                STL_LIST = glob.glob1(MVT_DIR, "*.stl")
                


        for STL in STL_LIST:
            NFD = JTA_FFT(cal)
            NFD.Make_Contour_Lib(MVT_DIR +"/" +  STL)
            NFD.Create_NFD_Library()
            stl_name = os.path.splitext(STL)[0]
            NFD.save(NFD_DIR + stl_name)

if __name__ == '__main__':
    main()