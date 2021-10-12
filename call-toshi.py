import numpy as np
import os
import glob
from MakeLib import JTA_FFT


HOME_DIR = "C:/Datasets_FemCleaned/Toshi/Toshi_Organized"


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
            print(movement_dir)



            FFT_fem = JTA_FFT(cal_data,fem_model_path)
            FFT_fem.MakeLib()
            FFT_fem.NFD_Lib(movement_dir,"fem")

            FFT_tib = JTA_FFT(cal_data,tib_model_path)
            FFT_tib.MakeLib()
            FFT_tib.NFD_Lib(movement_dir,"tib")