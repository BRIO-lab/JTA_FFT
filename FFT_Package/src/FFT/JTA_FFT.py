## JTA_FFT.py
# Copyright (c) Scott Banks banks@ufl.edu

# Imports
# from typing import OrderedDict
from numpy.fft.helper import fftshift
from numpy.lib.nanfunctions import _nansum_dispatcher
# from torch._C import float32, uint8, unify_type_list
from PIL import Image
import vtk
import numpy as np
import math
from vtk.util import numpy_support
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import pickle
from skimage import io
import torch
from torch import nn as nn
from torch import optim as optim
from torchvision import datasets, transforms, models
#from pose_hrnet_modded_in_notebook import PoseHighResolutionNet
from collections import OrderedDict
#from JTA_FFT_dataset import *


class JTA_FFT():

    # Initialize the class
    # TODO: determine the objects that we want this class to inherit 
    # I think that the best object to inherit would be the NFD library
    def __init__(self, CalFile):
        # Some of the coding variables

        # Number of Fourier Coefficients
        self.nsamp = 256
        # old was 128

        # Library Increment Parameters
        self.xrotmax = 45
        self.xrotinc = 3
        self.yrotmax = 45
        self.yrotinc = 3

        # Image Size
        self.imsize = 1024

        # Load in calibration file and check for proper formatting
        cal_data = np.loadtxt(CalFile, skiprows=1)
        self.CalFile = CalFile
        with open(CalFile, "r") as cal:
            if (cal.readline().strip() != "JT_INTCALIB"):
                raise Exception("Error! The provided calibration file has an incorrect header.")
        
        # Extracting the four components from the calibration file
        try:
            for idx, val in enumerate(cal_data):
                float(val)
                if val <= 0 and idx != 1 and idx != 2:
                    raise Exception("Error! Principal distance or scale is <= 0!")
        except ValueError as error:
            print("Error! ", error, " is not a float!")
        
        # principal distance
        pd = cal_data[0]
        self.pd = pd

        # scale (mm/px)
        sc = cal_data[3]
        self.sc = sc

        # x and y offset
        xo = cal_data[1]
        yo = cal_data[2]
        self.xo = xo
        self.yo = yo

        self.params = {'nsamp':self.nsamp, 
                       'xrotmax': self.xrotmax,
                       'xrotinc': self.xrotinc,
                       'yrotmax': self.yrotmax,
                       'yrotinc': self.yrotinc,
                       'imsize': self.imsize,
                       'pd': self.pd,
                       'sc': self.sc,
                       'xo': self.xo,
                       'yo': self.yo}
    
    def Segment(self, Model, Image):
        # Takes in a NN Model and unprocessed image, and segments the image. 
        # This allows it to be compared to the shape library that is created further on.

        '''
        We planned to use this but do not use it - you can ignore what is happening here
        '''

        # Load the model
        model = PoseHighResolutionNet(num_key_points=1,num_image_channels=1)
        cpu_model_state_dict = OrderedDict()

        for old_name, w in torch.load(Model, map_location='cpu')['model_state_dict'].items():
            if old_name[:6] == "module":
                name = old_name[7:] # remove "module
                cpu_model_state_dict[name] = w
                
            else:
                name = old_name
                cpu_model_state_dict[name] = w
            

        # set the model mode
        model.load_state_dict(cpu_model_state_dict)
        model.eval()

        # load the image that needs to be segmented
        imgset = FFTDataset(Image, transform = None)
        imgloader = torch.utils.data.DataLoader(imgset, batch_size = 1, shuffle = False)

        for batch in imgloader:
            batchx = batch["image"]
            self.outputImg = model(batchx)

        # pass the loaded image through the model
        
        L = (self.outputImg > 0).type(torch.float32)
        L = L[0,0,:,:]
        im = torchvision.transforms.ToPILImage()((255*L).type(torch.uint8))
        im.save("test_image_output.png") 
        self.outputImg = np.array(im)

        return self.outputImg

    def Make_Contour_Lib(self,STLFile):

        # Takes in a path to an STL model, and generates a contour library based on it.
        # This saves the generated rotation indices to self, and returns the x and y arrays of contours. 

        '''
        This is the function that creates the libraries for the contours. Meaning, it will take the current model and project/sample the contour to make a shape library
        '''

        plt.clf()

        self.STLFile = STLFile

        isc = 1/self.sc      # inverse scale [px/mm]
        fx = self.pd/self.sc # scale pd into pixel units
        fy = fx              # same pd in x and y
        cx = self.imsize/2   # project to image center
        cy = cx              # same x and y image center
        w = self.imsize      # assume square image 
        h = self.imsize      # assume square image


    # STL Render Setup
        renderer    = vtk.vtkRenderer()
        renWin      = vtk.vtkRenderWindow()
        renWin.SetOffScreenRendering(1)

    # Set up basic camera parameters in VTK and clipping planes
        cam = renderer.GetActiveCamera()
        near = 0.1
        far = 1.5*fx
        cam.SetClippingRange(near, far)
    
    # Position is at origin, looking in -z direction, y is up
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0, 0, -1)
        cam.SetViewUp(0, 1, 0)
        cam.SetWindowCenter(0, 0)
    
    # Set vertical view angle as an indirect way of 
    # setting the y focal distance
        angle = (180 / np.pi) * 2.0 * np.arctan2(self.imsize/2, fy)
        cam.SetViewAngle(angle)

    # Set vertical view angle as an indirect way of
    # setting the x focal distance

        m = np.eye(4)
        aspect = fy/fx
        m[0,0] = 1.0/aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        cam.SetUserTransform(t)

    # Set up vtk rendering again to make it stick
    # TODO: figure out why this doesn't work 
        renderer    = vtk.vtkRenderer()
        renWin      = vtk.vtkRenderWindow()
        renWin.SetOffScreenRendering(1)

    
    # Set basic camera parameters in VTK
    # TODO: figure out why this needs to be placed in the code again
        cam = renderer.GetActiveCamera()
        cam.SetClippingRange(near, far)
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0, 0, -1)
        cam.SetViewUp(0, 1, 0)
        cam.SetWindowCenter(0,0)
        cam.SetViewAngle(angle)
        cam.SetUserTransform(t)
    
    # Load in 3D model using VTK
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(self.STLFile)

    # Initialize VTK Transform Filter and mapper
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(stl_reader.GetOutputPort())

        stl_mapper = vtk.vtkPolyDataMapper()
        stl_mapper.SetInputConnection(transformFilter.GetOutputPort())

    # Define Rotations for library
        xrot = np.linspace(int(-1*self.xrotmax),
                           int(self.xrotmax),
                           int((2*self.xrotmax/self.xrotinc))+1)
        
        yrot = np.linspace(int(-1*self.yrotmax),
                           int(self.yrotmax),
                           int((2*self.yrotmax/self.yrotinc))+1) 

    # Create output arrays for contours
        self.xout = np.zeros([int((2*self.xrotmax/self.xrotinc)+1),
                              int((2*self.yrotmax/self.yrotinc)+1),
                              self.nsamp])
        
        self.yout = np.zeros([int((2*self.xrotmax/self.xrotinc)+1),
                              int((2*self.yrotmax/self.yrotinc)+1),
                              self.nsamp])

    # Create array to store rotation indices
        rot_indices = np.empty([xrot.size,yrot.size,2])

    # Create for-loop to run through each of the rotation combinations
        for j, xr in enumerate(xrot):
            for k, yr in enumerate(yrot):
            # Save the current rotation index
                rot_indices[j,k,0] = xr
                rot_indices[j,k,1] = yr

            # Transform the STL model based on the current rotation
                transform = vtk.vtkTransform()
                transform.PostMultiply()
                transform.Scale(isc,isc,isc)
                transform.RotateY(yr)
                transform.RotateX(xr)
                transform.Translate(0, 0, -0.9*fx)
                transformFilter.SetTransform(transform)
                transformFilter.Update()

            # STL Mapper
                stl_actor = vtk.vtkActor()
                stl_actor.SetMapper(stl_mapper)

            # STL Render

                renderer.AddActor(stl_actor)
                renderer.SetBackground(0.0, 0.0, 0.0)
                renWin.AddRenderer(renderer)
                renWin.SetSize(w,h)
                renWin.Render()

            # Render the scene into a numpy array for openCV processing
                winToIm = vtk.vtkWindowToImageFilter()
                winToIm.SetInput(renWin)
                winToIm.Update()
                vtk_image = winToIm.GetOutput()
                width, height, channels = vtk_image.GetDimensions()
                vtk_array = vtk_image.GetPointData().GetScalars()
                components = vtk_array.GetNumberOfComponents()
                arr = cv2.flip(numpy_support.vtk_to_numpy(vtk_array).reshape(height,width,components),0)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(arr, 1, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3,3),np.uint8)
                binary = cv2.dilate(binary, kernel, iterations = 1)
                binary = cv2.erode(binary, kernel, iterations = 1)
            
            # Get the contours of the created projection blob
                contours, hierarchy = cv2.findContours(binary,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)

                # Loop through the contours to only grab the largest

                for contour in contours:
                    x,y = contour.T
                    
                # Convert from numpy arrays to normal arrays
                    x = x.tolist()[0]
                    y = y.tolist()[0]
                    
                    if len(x) > 200:

                    # Resample contour in nsamp equispaced increments using spline interpolation
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                        tck, u = splprep([x,y], u=None, s=1.0, per=1)
                    
                    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                        u_new = np.linspace(u.min(), u.max(), self.nsamp)
                    
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                        x_new, y_new = splev(u_new, tck, der=0)
                        
                        
                    # Convert it back to numpy format for opencv to be able to display it
                        plt.plot(x_new,self.imsize-y_new)

                    # break code if you are done
                        break
                        
                    else:
                        x_new = 0
                        y_new = 0

                self.xout[j,k,:] = x_new
                self.yout[j,k,:] = y_new
        

        # Save the rotation indices in the working directory
        # np.save(dir + "/rot_indices.npy", rot_indices)
        self.rot_indices = rot_indices
        return self.xout, self.yout


# Now we want to write a function that will create the 
# Normalized Fourier Descriptor Library

    def Create_NFD_Library(self):
        # TODO: fix the loading of different types for each of the sub-functions
        '''
        This function takes in the shape library created from the previous step and turns everything into a 
        '''

        x = self.xout
        y = self.yout

    # Get Dimensions of Input Contours
        r, c, nsamp = x.shape

    # Set up library normalizations
    # See Banks and Hodge IEEE 1996

        kmax = 5 # arbitrary number of normalizations
        k_norm = np.array([2, -1, -2, -3, -4]) # typically falls on one of these

    # Centroid of Projection (px)
    # In Banks 96, this is S(0)
    # Dims: [x-rot, y-rot] - one per projection, unaffected by norms
        centroid_library = np.zeros([r,c],dtype='c16')
    
    # Magnitude of Projection (px)
    # In Banks 96, this is S(1)
    # Dims: [xrot, yrot] - one per projection, unaffected by norms
        mag_library = np.zeros([r,c])

    # In-plane-rotation angles
    # Dims: [x-rot, y-rot, norms] - 1 per projection per norm
        angle_library = np.zeros([r,c,kmax])

    # Fourier Descriptor Library
    # S(k) in Banks 96
    # Dims: [x-rot, y-rot, norms, num-samples] - storing full library per norm per projection
        NFD_library = np.zeros([r,c,kmax,self.nsamp], dtype = 'c16')

    # TODO: find out what the index vectors are
        index_vect = np.linspace(1,self.nsamp, self.nsamp) - nsamp/2

        max_norms = 0
    
    # Loop through each of the dimensions of the rotation indices
        for i in range(r):
            for j in range(c):
                x_new = x[i,j,:]
                y_new = y[i,j,:]

                

            # Performing Fourier decomp on complex contour variable
            # y is subtracted from imsize to accunct for image pixel coords
                # FIXME: Need to fix the way that the FFT works, change norm
                # norm = "ortho" should do the trick
                #fcoord = np.fft.fft((x_new + (self.imsize-y_new)*1j),nsamp)
                fcoord = np.fft.fft(
                    a = (x + (self.imsize-y)*1j),
                    n = nsamp,
                    norm = "ortho"
                )
                print(fcoord.imag)
            # Shift so that the centroid of the projection is in the center
                # FIXME: Fix this to normalize properly
                fcoord = np.fft.fftshift(fcoord)

            # Save the centriod in separate variable and make 0 in NFD
            # This normalizes NFD position
                centroid_library[i,j] = (fcoord[int(nsamp/2)])
                fcoord[int(nsamp/2)] = 0

            # Sort fft coeffs by magnitude
                idx = np.argsort(abs(fcoord))
                idx = idx[::-1] # sort descending
            
            # Find number of normalizations
                num_norms = abs(idx[1] - nsamp/2 - 1)

            # Normalizations Check
                if num_norms == 0:
                    print('No Valid Normalizations')
                    break

            # Set Max Norms
                if num_norms > max_norms:
                    max_norms = num_norms
            
            # Set the magnitude of the projected image
            # S(1) term in Banks 96
            # Fcoord starts at negative frequencies, so need to shift to center
                mag_library[i,j] = abs(fcoord[int(nsamp/2 + 1)])

            # Normalize fcoords by magnitude
                fcoord = fcoord/mag_library[i,j]

            # Loop through each normalization
                for norm in range(int(num_norms)):
                    k = k_norm[norm]

                # Compute the Phase Angles of A(1) and A(k)
                    u = np.arctan2(
                        fcoord.imag[int(nsamp/2 + 1)],
                        fcoord.real[int(nsamp/2 + 1)]
                    )

                    v = np.arctan2(
                        fcoord.imag[int(idx[1])],
                        fcoord.real[int(idx[1])]
                    )

                # Create reference angle and add to library
                    angle = ((v - k*u)/(k-1))*180./np.pi
                    angle_library[i,j,norm] = angle 

                # Compute Complex Angle to standardize in-plane rotation
                # and contour starting point
                    ang = ((index_vect - k)*u + (1 - index_vect)*v)/(k-1)
                    coeff = np.cos(ang) + np.sin(ang)*1j

                # Finish Normalization
                    NFD_library[i,j,norm,:] = fcoord*coeff
        

    # Remove the empty portions of the angle and NFD lib for number of norms
        max_norms = int(max_norms)
        NFD_library = NFD_library[:,:,:max_norms,:]
        angle_library = angle_library[:,:,:max_norms]

    # Saving numpy files for each of the pertinent variables
    # TODO: store these all as a class object using pickle
    # Storing as pickle would allow them to be stored in a separate class and load properly
    # For now, saving as .npy files works

        #np.save(dir + "/NFD-lib_" + model_type + ".npy", NFD_library)
        #np.save(dir + "/MAG-lib_" + model_type + ".npy", mag_library)
        #np.save(dir + "/ANGLE-lib_" + model_type + ".npy", angle_library)
        #np.save(dir + "/CENTROID-lib_" + model_type + ".npy", centroid_library)

        self.centroid_library = centroid_library
        self.mag_library = mag_library
        self.angle_library = angle_library
        self.NFD_library = NFD_library

        return centroid_library, mag_library, angle_library, NFD_library
    
    def Create_Contour(self,image):

    # Apply the same dilation and erosion to smooth image
        kernel = np.ones([3,3], np.uint8)
        binary = cv2.dilate(image, kernel, iterations = 1)
        binary = cv2.erode(binary, kernel, iterations = 1)

    # Find the contours of the provided image
        contours, hierarchy = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
        
        smoothened = []
        done = 0
        for contour in contours:
            x,y = contour.T

        # Convert from Numpy Arrays to Normal Arrays
            x = x.tolist()[0]
            y = y.tolist()[0]
            

            if len(x) > 200:
                # Resample contour in nsamp equispaced increments using spline interpolation
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                tck, u = splprep([x,y], u=None, s=1.0, per=1)
                # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                u_new = np.linspace(u.min(), u.max(), self.nsamp)
                #u_new = np.linspace(u.min(), u.max(), 64)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                self.x_new, self.y_new = splev(u_new, tck, der=0)
                # Convert it back to numpy format for opencv to be able to display it
                plt.plot(self.x_new,self.imsize-self.y_new)
                plt.show()
                # commented out the line above to skip display window opening.
                done = 1
            if done:
                break

        return self.x_new, self.y_new

    def get_NFD(self,x,y):

        # This function will create an NFD representation of provided x,y
        nsamp = self.nsamp

        kmax = 5
        k_norm = np.array([2, -1, -2, -3, -4])

        # Create an NFD instance for this variable
        # Dims: [norms, num-samp]
        NFD_instance = np.zeros([kmax,nsamp], dtype='c16')

        # Create an angle instance
        angle_instance = np.zeros([kmax])

        index_vect = np.linspace(1,nsamp,nsamp) - nsamp/2

        max_norms = 0
    # FIXME: Need to fix the normalization for the fourier transformation 
        #fcoord = np.fft.fft((x + (self.imsize-y)*1j),nsamp)
        fcoord = np.fft.fft(
            a = (x + (self.imsize-y)*1j),
            n = nsamp,
            norm = "ortho")
        fcoord = np.fft.fftshift(fcoord)

        centroid_instance = (fcoord[int(nsamp/2)])
        fcoord[int(nsamp/2)] = 0 
        idx = np.argsort(abs(fcoord))
        idx = idx[::-1]
        num_norms = abs(idx[1]-nsamp/2)

        if num_norms == 0:
            print('No Valid Normalizations')
            return
        
        if num_norms > max_norms:
            max_norms = num_norms

        mag_instance = abs(fcoord[int(nsamp/2 + 1)])
        fcoord = fcoord / mag_instance

        for norm in range(int(num_norms)):
            k = k_norm[norm]
        # Compute the Phase Angles of A(1) and A(k)
            u = np.arctan2(
                fcoord.imag[int(nsamp/2 + 1)],
                fcoord.real[int(nsamp/2 + 1)]
            )

            v = np.arctan2(
                fcoord.imag[int(idx[1])],
                fcoord.real[int(idx[1])]
            )

        # Save the Angle Instance
            angle_instance[norm] = ((v - k*u)/(k-1))*(180./np.pi)
            ang = ((index_vect - k)*u + (1 - index_vect)*v)/(k-1)
            coeff = np.cos(ang) + np.sin(ang)*1j
            NFD_instance[norm,:] = fcoord*coeff
            

        max_norms = int(max_norms)
        NFD_instance = NFD_instance[0:max_norms,:] 
        angle_instance = angle_instance[0:max_norms]

        instance = {"centroid":centroid_instance, 
                    "mag":mag_instance, 
                    "angle":angle_instance,
                    "NFD":NFD_instance}

        return instance
    
    def estimate_pose(self, instance):
        
        xspan = self.NFD_library.shape[0]
        yspan = self.NFD_library.shape[1]
        
    # Create an empty variable to fill up the distance 
    # from the instance to each of the library variables

        dist = np.empty([xspan,yspan])

    # TODO: REMOVE THIS ONCE YOU ARE DONE WITH CURRENT TESTING
    # TODO: the following code adjusts for the incorrect normalization when creating the libs
    # TODO: this is only a placeholder

        centroid_library = self.centroid_library
        centroid_instance = instance["centroid"]


        mag_library = self.mag_library
        mag_instance = instance["mag"]

    # Loop through all the indices in the library and check the distance w instance
        for i in range(0,xspan):
            for j in range(0,yspan):

            # Compute the difference between the instance and the library
                diff = instance["NFD"][0,:] - self.NFD_library[i,j,0,:]

            # Take the L2 norm to get the distance
                dist[i,j] = np.linalg.norm(diff)

        
    # Find the location of the minimum distance
        idx,idy = np.where(dist == dist.min())

    # Find the x and y rotation estimates from the library indices
        x_rot_est = self.rot_indices[idx,idy,0]
        y_rot_est = self.rot_indices[idx,idy,1]

    # Now, find the z-rotation based on the library angle normalizations
        z_rot_est = instance["angle"][0] - self.angle_library[idx,idy,0]
        z_rot_rad = z_rot_est * np.pi / 180
        z_rot_rad = -1*z_rot_rad

    # Set the value of the z-translation of the library
    # For similar triangles to work, this is measured from the image-plane, not the camera
    # Calculate the z_est using similar triangles
        z_lib = 0.1*self.pd
        z_est = self.pd - (self.pd - z_lib)*(mag_library[idx,idy] / mag_instance)

    # Calculate X and Y translations using the value of the centroid of the non-normalized vector
    # You also need to adjust for the image center based on camera 
        x_offset = (0.5 * self.imsize) + (self.xo/self.sc)
        y_offset = (0.5 * self.imsize) + (self.xo/self.sc)

    # Calculate the zoom based on similar triangles
        zoom = (self.pd - z_lib)/(self.pd - z_est)

        x_lib = (centroid_library[idx,idy].real - x_offset)*zoom
        y_lib = (centroid_library[idx,idy].imag - y_offset)*zoom


    # Now, calculate the location of the estimated x and y translation based on centroid and roatation

        rot = np.array([[math.cos(z_rot_rad),-math.sin(z_rot_rad)],
                        [math.sin(z_rot_rad), math.cos(z_rot_rad)]])
        
        x_inst = centroid_instance.real - x_offset
        y_inst = centroid_instance.imag - y_offset

        t_inst = np.array([[x_inst],[y_inst]])
        t_lib  = np.array([[x_lib[0]],[y_lib[0]]])
        t_est_px = t_inst - np.matmul(rot,t_lib)
        t_est_len = t_est_px * self.sc

        x_est, y_est = t_est_len

    # x_est = x_inst - (math.cos(z_rot_rad)*x_lib - math.sin(z_rot_rad)*y_lib)
    #y_est = y_inst - (math.sin(z_rot_rad)*x_lib + math.cos(z_rot_rad)*y_lib)

    # Fix the units based on the location of the focal angle. Move z_est based on camera
        z_est_corr = z_est - self.pd
        if abs(z_est_corr) > abs(self.pd):
            z_est_corr[0] = - self.pd

    # Fix the rotations based on the projective geometry
        phi_x = np.arctan2(y_est, (self.pd - z_est)) * np.pi/180
        phi_y = np.arctan2(x_est, (self.pd - z_est)) * np.pi/180

        x_rot_corr = x_rot_est + np.cos(z_rot_rad)*phi_x - np.sin(z_rot_rad)*phi_y
        y_rot_corr = y_rot_est - np.sin(z_rot_rad)*phi_x - np.cos(z_rot_rad)*phi_y

        return x_est[0], y_est[0], z_est_corr[0], -1*z_rot_est[0], x_rot_corr[0], y_rot_corr[0]

    def load_pickle(self, pickle_path):
        """
            Loads a pickle from a passed-in file path.
            Once the pickle is loaded, saves its params to self variables in order to allow easier access of needed data.
        """
        try:
            FFTFile = open(pickle_path, 'rb')
            self.FFTPickle = pickle.load(FFTFile)
            self.NFD_library = self.FFTPickle['NFD_library']
            self.angle_library = self.FFTPickle['angle_library']
            self.mag_library = self.FFTPickle['mag_library']
            self.centroid_library = self.FFTPickle['centroid_library']
            self.rot_indices = self.FFTPickle['rot_indices']
            FFTFile.close()

        except FileNotFoundError:
            print("Error! The file you are trying to load either does not exist, or does not exist at this location: ", pickle_path)

    def save (self, filename):
        """ 
            Saves the necessary library variables into a dict for easier access after unpickling.
            If any data does not exist, pickling and saving is skipped, and the user is informed.
            Otherwise, the data is saved to a file 'filename'.nfd
        """  
        try:
            self.nfd_dict = {"NFD_library":self.NFD_library, "angle_library": self.angle_library,
                "mag_library": self.mag_library, "centroid_library": self.centroid_library,
                "rot_indices": self.rot_indices}
            filename = filename + '.nfd'
            output = open(filename, 'wb')
            pickle.dump(self.nfd_dict, output)
            pickle.dump(self.params, output)
            output.close()
        except AttributeError as error:
            print("Error!", error, "\nAll library objects must be instantiated before trying to save!")