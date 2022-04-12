import numpy as np
import os
from JTA_FFT import *
import cv2

def main():
    HOME_DIR = "C:/Datasets_FemCleaned/Actiyas/Actiyas_Organized/T01L"
    cal = HOME_DIR + "/cal1024.txt"
    FEM_NFD = JTA_FFT(cal)
    TIB_NFD = JTA_FFT(cal)
    FEM_NFD.load_pickle(HOME_DIR + "/FemurL2_fem.nfd")
    TIB_NFD.load_pickle(HOME_DIR + "/TibiaL3_tib.nfd")
    jts_fmt = '%16.9s,%16.9s,%16.9s,\t\t   %16.9s,%16.9s,%16.9s'
    jts_header = "JT_EULER_312\n          x_tran,          y_tran,          z_tran,                   z_rot,           x_rot,           y_rot"

    for session in ["Session_1"]:
        SESS_DIR = HOME_DIR + "/" + session

        for mvt in os.listdir(SESS_DIR):
            MVT_DIR = SESS_DIR + "/" + mvt
            '''
            Do the femur calculations
            '''
            FEM_IMG_DIR = MVT_DIR + "/1024/fem/"
            num = len(os.listdir(FEM_IMG_DIR))
            fem_jts_file = np.empty([num,6])
            tib_jts_file = np.empty([num,6])
            for idx, img_id in enumerate(os.listdir(FEM_IMG_DIR)):
                img = cv2.imread(FEM_IMG_DIR + "/" + img_id, cv2.IMREAD_GRAYSCALE)
                xin, yin = FEM_NFD.Create_Contour(img)
                fem_instance = FEM_NFD.get_NFD(xin,yin)
                xt,yt,zt,zr,xr,yr = FEM_NFD.estimate_pose(fem_instance)
                fem_jts_file[idx,:] = [xt,yt,zt,zr,xr,yr]
            
            

            '''
            Do the tibia calculations
            '''
            TIB_IMG_DIR = MVT_DIR + "/1024/tib/"
            for idx, img_id in enumerate(os.listdir(TIB_IMG_DIR)):
                img = cv2.imread(TIB_IMG_DIR + "/" + img_id, cv2.IMREAD_GRAYSCALE)
                xin, yin = TIB_NFD.Create_Contour(img)
                fem_instance = TIB_NFD.get_NFD(xin,yin)
                xt,yt,zt,zr,xr,yr = TIB_NFD.estimate_pose(fem_instance)
                tib_jts_file[idx,:] = [xt,yt,zt,zr,xr,yr] 

            np.savetxt(MVT_DIR + "/tib_fft.jts",tib_jts_file,delimiter=",", header=jts_header, fmt = jts_fmt,comments = '')
            np.savetxt(MVT_DIR + "/fem_fft.jts",fem_jts_file, delimiter = ",", header = jts_header, fmt = jts_fmt, comments = '')
            
if __name__ == "__main__":
    main()