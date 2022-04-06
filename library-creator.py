from JTA_FFT_try import JTA_FFT
import os
import glob
import numpy as np
import gc
import nvtx

@nvtx.annotate("Library-Creator", color = "blue")
def main():
    HOME_DIR = "/red/nvidia-ai/miller-lab/data"
    study_list = ["GMK"]
    
    for study in study_list:
        study_dir = HOME_DIR + "/" + study
        NFD_DIR = study_dir + "/lib/"
        if not os.path.exists(NFD_DIR):
            os.mkdir(NFD_DIR)
        if os.path.exists(study_dir + "/cal1024.txt"):
            cal = study_dir + "/cal1024.txt"
        else:
            cal = study_dir + "/calibration.txt"
        if not os.path.exists(cal):
            raise Exception("NO calibration file found")

        for stl in glob.glob1(study_dir, "*.stl"):
            stl_path = study_dir + "/" + stl
            stl_name = os.path.splitext(stl)[0]
            print(stl_path)
            if os.path.exists(NFD_DIR + stl_name + ".nfd"):
                print("Skipped: ", stl_name)
                continue
            else:
                NFD = JTA_FFT(cal)
                NFD.create_nfd_library(stl_path)
                NFD.print_library(0,0.1,200)
                NFD.save_nfd_library(NFD_DIR + stl_name)
                del(NFD)


if __name__ == '__main__':
    main()