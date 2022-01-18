import enum
import numpy as np
import os
import JTA_FFT
import cv2
import glob
import gc


def main():
    # A list of each of the studies for which we want to perform NFD estimations
    study_list = ["Arizona"]

    # the main directory of the studies
    HOME_DIR = "C:/Datasets_FemCleaned"

    # Creating a format and a header for the JTS files that will be created
    jts_fmt = '%16.9s,%16.9s,%16.9s,\t\t   %16.9s,%16.9s,%16.9s'
    jts_header = "JT_EULER_312\n          x_tran,          y_tran,          z_tran,                   z_rot,           x_rot,           y_rot"

    # Looping through each of the studies
    for study in study_list:
        study_dir = HOME_DIR +  "/" + study

        # NFD library location within each of the studies
        NFD_DIR = study_dir + "/" + "NFD Library/lib/"

        # Lima folders have a bit of a different structure
        if study != "Lima":
            study_org_dir =  study_dir + "/" + study +  "_Organized"
        else:
            study_org_dir =  study_dir + "/" + study +  "_Organized_Updated"

        # Looping through patients within each study using list comprehension
        # We want to make sure that we are only grabbing directories 
        for pat_id in [x for x in os.listdir(study_org_dir) if os.path.isdir(study_org_dir + "/" + x)]:
            
            pat_dir = study_org_dir + "/" + pat_id

            # looping through each session using list comprehension
            for sess_id in [x for x in os.listdir(pat_dir) if os.path.isdir(pat_dir + "/" + x)]:
                sess_dir = pat_dir + "/" + sess_id

                for mvt_id in [x for x in os.listdir(sess_dir) if (os.path.isdir(sess_dir + "/" + x))]:
                    mvt_dir = sess_dir + "/" + mvt_id

                    # Finding the location of each of the stls in each folder
                    fem_stl = glob.glob1(mvt_dir, "*fem*.stl")
                    tib_stl = glob.glob1(mvt_dir, "*tib*.stl")

                    # skipping this movement if we are unable to find any stls
                    if len(fem_stl) ==0 or len(tib_stl) == 0:
                        print("NO STL FOUND: ",mvt_dir )
                        continue


                    # determining the name of the stls so we can find the matching NFD
                    tib_name = os.path.splitext(tib_stl[0])[0]
                    tib_nfd_path = NFD_DIR + "/" + tib_name + ".nfd"
                      
                    fem_name = os.path.splitext(fem_stl[0])[0]
                    fem_nfd_path = NFD_DIR + "/" + fem_name + ".nfd"


                    # location of the machine learning images that we created
                    fem_img_dir = mvt_dir + "/1024/fem/"
                    tib_img_dir = mvt_dir + "/1024/tib/"

                    if os.path.exists(fem_img_dir):
                        num_imgs = len(os.listdir(fem_img_dir))


                    # checm to make sure there are images in the NN output folders for each movement
                    if (not os.path.exists(fem_img_dir)) or (num_imgs == 0):
                        continue
                    
                    # Create blank arrays that will be the JTS files
                    fem_jts_file = np.empty([num_imgs,6])
                    tib_jts_file = np.empty([num_imgs,6])

                    # Checking to see which of the calibrations to use based on the file that we are working in

                    for root, dirs, files in os.walk(mvt_dir):
                        if files == "cal1024.txt":
                            print(root,"/",files)

                    
                    if os.path.exists(mvt_dir + "/cal1024.txt"):
                        calibration = mvt_dir + "/cal1024.txt"
                    else:
                        calibration = mvt_dir + "/calibration.txt"
                    
                    # Throwing most of the logic into a try-catch statement so we can avoid some of the errors if something goes wrong
                    try:
                        FEM_NFD = JTA_FFT(calibration)
                        TIB_NFD = JTA_FFT(calibration)

                        FEM_NFD.load_pickle(fem_nfd_path)
                        TIB_NFD.load_pickle(tib_nfd_path)

                        # running through the femoral images

                    except:
                        print("Something went wrong with loading an NFD: ", mvt_dir)
                        continue


                    # want to try running through images in case some error gets flagged
                    try:
                        for idx, fem_img in enumerate(os.listdir(fem_img_dir)):
                            img = cv2.imread(fem_img_dir + "/" + fem_img, cv2.IMREAD_GRAYSCALE)
                            xin, yin = FEM_NFD.Create_Contour(img)
                            fem_instance = FEM_NFD.get_NFD(xin,yin)
                            xt,yt,zt,zr,xr,yr = FEM_NFD.estimate_pose(fem_instance)
                            fem_jts_file[idx,:] = [xt,yt,zt,zr,xr,yr]
                    except:
                        print("Error with Calculating fem pose: ", mvt_dir)
                        continue


                    try:
                         for idx, tib_img in enumerate(os.listdir(tib_img_dir)):
                             img = cv2.imread(tib_img_dir + "/" + tib_img, cv2.IMREAD_GRAYSCALE)
                             xin, yin = TIB_NFD.Create_Contour(img)
                             tib_instance = TIB_NFD.get_NFD(xin,yin)
                             xt,yt,zt,zr,xr,yr = TIB_NFD.estimate_pose(tib_instance)
                             tib_jts_file[idx,:] = [xt,yt,zt,zr,xr,yr] 
                    except:
                        print("Error with calculating tib pose: ", mvt_dir)
                        continue
                    

                    NFD_MVT_DIR = mvt_dir + "/FFT/"

                    if not os.path.exists(NFD_MVT_DIR):
                        os.mkdir(NFD_MVT_DIR)


                    try:
                        np.savetxt(NFD_MVT_DIR + "/tib_fft.jts", tib_jts_file, delimiter=",", header=jts_header, fmt=jts_fmt,comments='')
                        np.savetxt(NFD_MVT_DIR + "/fem_fft.jts",fem_jts_file, delimiter=",",header=jts_header,fmt=jts_fmt,comments='')
                    except:
                        print("Error saving jts files: ", mvt_dir)
                        continue

                        
                        # Tibial images and calculations
 

                    
                    
                    try:
                        del(FEM_NFD)
                        del(TIB_NFD)
                        gc.collect()
                    except:
                        continue

                    gc.collect()


                    

if __name__ == "__main__":
    main()