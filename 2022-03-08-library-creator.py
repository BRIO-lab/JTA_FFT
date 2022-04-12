from JTA_FFT_try import JTA_FFT
import os
import glob
import numpy as np
import gc
def main():

    
    
    HOME_DIR = "C:/Datasets_FemCleaned/"
    study_list = ["GMK"]
    
    for study in study_list:
        study_dir = HOME_DIR + "/" + study
        NFD_DIR = study_dir + "/" + "NFD Library/lib/"
        
        if study != "Lima":
            DATA_DIR =  study_dir + "/" + study +  "_Organized"
        else:
            DATA_DIR =  study_dir + "/" + study +  "_Organized_Updated"
            
        
        for pat_id in [x for x in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR + "/" + x)]:
            PAT_DIR = DATA_DIR + "/" + pat_id
            for sess_id in [x for x in os.listdir(PAT_DIR) if os.path.isdir(PAT_DIR + "/" + x)]:
                SESS_DIR = PAT_DIR + "/" + sess_id
                for mvt_id in [x for x in os.listdir(SESS_DIR) if (os.path.isdir(SESS_DIR + "/" + x))]:
                    MVT_DIR = SESS_DIR + "/" + mvt_id
                    cal = MVT_DIR + "/cal1024.txt"
                    if not os.path.exists(cal):
                        cal = MVT_DIR + "/1024/cal1024.txt"
                        if not os.path.exists(cal):
                            cal = MVT_DIR + "/calibration.txt"
                            if not os.path.exists(cal):
                                raise Exception ("No calibration file found: ",cal)
                    STL_LIST = glob.glob1(MVT_DIR, "*.stl")
                    for STL in STL_LIST:
                        stl_name = os.path.splitext(STL)[0]
                        if os.path.exists(NFD_DIR + stl_name + ".nfd"):
                            print("Skipped: ", stl_name)
                            continue
                        else:
                            NFD = JTA_FFT(cal)
                            NFD.create_nfd_library(MVT_DIR +"/" +  STL)
                            stl_name = os.path.splitext(STL)[0]
                            NFD.save_nfd_library(NFD_DIR + stl_name)
                            del(NFD)
                            gc.collect()
                    del(STL_LIST)

if __name__ == '__main__':
    main()