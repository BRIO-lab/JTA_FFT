import numpy as np
import os
import glob
from JTA_FFT import *


HOME_DIR = "C:/Datasets_FemCleaned/Arizona/Arizona_Organized"


for patient_id in os.listdir(HOME_DIR):
    pat_dir = HOME_DIR + "/" + patient_id
    for session_id in os.listdir(pat_dir):
        session_dir = pat_dir +  "/" + session_id
        for movement_id in os.listdir(session_dir):
            movement_dir = session_dir + "/" + movement_id
            
            #calibration = glob.glob1(movement_dir, "*cal1024.txt")
            cal_data = np.loadtxt(movement_dir + "/cal1024.txt", skiprows=1)
            fem_model = glob.glob1(movement_dir, "*fem*.stl")
            tib_model = glob.glob1(movement_dir, "*tib*.stl")
            fem_model_path = movement_dir + "/" + fem_model[0]
            tib_model_path = movement_dir + "/" + tib_model[0]
            fem_lib_name = os.path.splitext(fem_model_path)[0] 
            tib_lib_name = os.path.splitext(tib_model_path)[0] 
            print(movement_dir)



            FFT_fem = JTA_FFT(movement_dir + "/cal1024.txt")
            FFT_fem.Make_Contour_Lib(cal_data,fem_model_path,movement_dir)
            FFT_fem.Create_NFD_Library(movement_dir,"fem")
            FFT_fem.save(fem_lib_name)

            FFT_tib = JTA_FFT(movement_dir + "/cal1024.txt")
            FFT_tib.Make_Contour_Lib(cal_data,tib_model_path, movement_dir)
            FFT_tib.Create_NFD_Library(movement_dir,"tib")
            FFT_tib.save(tib_lib_name)