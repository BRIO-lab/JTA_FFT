import numpy as np
import os
from JTA_FFT import *
import cv2
import glob


def main():
    study_list = ["Actiyas", "Arizona", "GMK", "Lima", "Toshi"]
    HOME_DIR = "C:/Datasets_FemCleaned"
    
    for study in study_list:
        study_dir = HOME_DIR +  "/" + study
        NFD_DIR = study_dir + "/" + "NFD Library/lib/"


        if study != "Lima":
            study_org_dir =  study_dir + "/" + study +  "_Organized"
        else:
            study_org_dir =  study_dir + "/" + study +  "_Organized_Updated"

        for pat_id in [x for x in os.listdir(study_org_dir) if os.path.isdir(study_org_dir + "/" + x)]:
            
            pat_dir = study_org_dir + "/" + pat_id

            for sess_id in [x for x in os.listdir(pat_dir) if os.path.isdir(pat_dir + "/" + x)]:
                sess_dir = pat_dir + "/" + sess_id

                for mvt_id in [x for x in os.listdir(sess_dir) if (os.path.isdir(sess_dir + "/" + x))]:
                    mvt_dir = sess_dir + "/" + mvt_id


                    fem_stl = glob.glob1(mvt_dir, "*fem*.stl")
                    tib_stl = glob.glob1(mvt_dir, "*tib*.stl")

                    if len(fem_stl) ==0 or len(tib_stl) == 0:
                        print("NO STL FOUND: ",mvt_dir )
                        continue

                    tib_name = os.path.splitext(tib_stl[0])[0]
                    tib_nfd_path = NFD_DIR + "/" + tib_name + ".nfd"
                      
                    fem_name = os.path.splitext(fem_stl[0])[0]
                    fem_nfd_path = NFD_DIR + "/" + fem_name + ".nfd"
                    

                   

                    
                    

                    if os.path.exists(fem_nfd_path):
                        print("Found NFD: ", fem_name)
                    else:
                        print("NOT FOUND: ", fem_name)

if __name__ == "__main__":
    main()