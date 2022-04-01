## JTA_FFT.py
# Copyright (c) Scott Banks banks@ufl.edu, Andrew Jensen andrewjensen321@gmail.com

# Imports
# from typing import OrderedDict
from numpy.fft.helper import fftshift
from numpy.lib.nanfunctions import _nansum_dispatcher
import vtk
import cupy as np
#import cupy as cp
import math
from vtk.util import numpy_support
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import pickle
#from skimage import io
#from pose_hrnet_modded_in_notebook import PoseHighResolutionNet
import os
from rotation_utility import *
import time 
import nvtx
# TODO: look into vtk-m


class JTA_FFT():
    
    
    def __init__(self, CalFile):
        # Some of the coding variables
        self.max_num_norms = 5
        # Number of Fourier Coefficients
        self.nsamp = 128
        #linspace is a tool for creating numeric sequences (creates a sequence of evenly spaces numers in a numpy array)
        self.index_vect = cp.linspace(-self.nsamp/2 + 1, self.nsamp/2, self.nsamp)  #FIXME: np -> cp
        
        # Library Increment Parameters: How far will you rotate in the x and y directions. That grid of directions 
        # Make it symmetric, 10 degree increments
        # Doesn't really matter
        self.xrotmax = 30
        self.xrotinc = 3
        self.yrotmax = 30
        self.yrotinc = 3

        # Image Size
        self.imsize = 1024

        # Load in calibration file and check for proper formatting
        cal_data = cp.loadtxt(CalFile, skiprows=1)    #FIXME: np -> cp
        
        self.CalFile = CalFile

        # Make sure that the proper calibration file is being used
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
        self.pd = pd    # was 1200 mm in Dr.Banks code

        # scale (mm/px)
        sc = cal_data[3]
        self.sc = sc    # 0.373 (all of this should be taken from calibration file)

        # x and y offset
        xo = cal_data[1]
        yo = cal_data[2]
        self.xo = xo
        self.yo = yo

        # store parameters as a dictionary in case you ever want to reference them
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

        
        # Initialize some of the viewing and rending windows in the initialization

        isc = 1/self.sc      # inverse scale [px/mm]
        fx = self.pd/self.sc # scale pd into pixel units
        fy = fx              # same pd in x and y
        cx = self.imsize/2   # project to image center
        cy = cx              # same x and y image center
        w = self.imsize      # assume square image 
        h = self.imsize      # assume square image
        self.isc = isc
        self.fx = fx

    # This is being called like a billion times, this function is what causes the create_nfd_library to be so fat
    @nvtx.annotate("create_projection", color = "purple")
    def create_projection(self, STLFile,renWin, renderer, transformFilter, stl_mapper, xr,yr,zr, translation = None):
        # Takes in a path to an STL model, and generates a contour library based on it.
        # This saves the generated rotation indices to self, and returns the x and y arrays of contours. 
        n = lambda a: int(cp.where(self.index_vect == a)[0][0])     # Lambda is used to specify an expression, Check where index_vect == a   #FIXME: np -> cp

        A = lambda a: int(self.index_vect[a])
        '''
        This is the function that creates the libraries for the contours. Meaning, it will take the current model and project/sample the contour to make a shape library
        '''

        

    # Define Rotations for library      #FIXME: np -> cp
        xrot = cp.linspace(int(-1*self.xrotmax),
                           int(self.xrotmax),
                           int((2*self.xrotmax/self.xrotinc))+1)
        
        yrot = cp.linspace(int(-1*self.yrotmax),
                           int(self.yrotmax),
                           int((2*self.yrotmax/self.yrotinc))+1) 
    
    
    ## Create output arrays for contours
    #    xout = np.zeros([int((2*self.xrotmax/self.xrotinc)+1),
    #                          int((2*self.yrotmax/self.yrotinc)+1),
    #                          self.nsamp])
    #    
    #    yout = np.zeros([int((2*self.xrotmax/self.xrotinc)+1),
    #                          int((2*self.yrotmax/self.yrotinc)+1),
    #                          self.nsamp])

    # Create array to store rotation indices
        rot_indices = cp.empty([xrot.size,yrot.size,2]) #FIXME: np -> cp

    # Create for-loop to run through each of the rotation combinations
    # Transform the STL model based on the current rotation
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Scale(self.isc,self.isc,self.isc)
        transform.RotateY(yr)
        transform.RotateX(xr)
        transform.RotateZ(zr)
        if translation is not None:
            xt = translation[0] / self.sc
            yt = translation[1] / self.sc
            zt = translation[2] / self.sc
            transform.Translate(xt, yt, zt)
        else:
            transform.Translate(0, 0, -0.9*self.fx)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        stl_actor = vtk.vtkActor()
        stl_actor.SetMapper(stl_mapper)
        renderer.AddActor(stl_actor)
        renderer.SetBackground(0.0, 0.0, 0.0)
        renWin.AddRenderer(renderer)
        renWin.SetSize(self.imsize,self.imsize)
        renWin.Render()
            # nder the scene into a numpy array for openCV processing
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
        kernel = cp.ones((5,5),cp.uint8)    #FIXME: np -> cp
        binary = cv2.dilate(binary, kernel, iterations = 1)
        binary = cv2.erode(binary, kernel, iterations = 1)
    
            # t the contours of the created projection blob
        contours, hierarchy = cv2.findContours(binary,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        # Loop through the contours to only grab the larges
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
                u_new = cp.linspace(u.min(), u.max(), self.nsamp)   #FIXME: np -> cp
            
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                x_new, y_new = splev(u_new, tck, der=0)
                
                
            # Convert it back to numpy format for opencv to be able to display it
                #fig, ax = plt.subplots(figsize = (10,10))
                #plt.plot(x_new,self.imsize-y_new)
                #ax.set_aspect('equal')
                #plt.show()
            # break code if you are done
                break
                
            else:
                x_new = 0
                y_new = 0
        
        return x_new, y_new
    
     
    @nvtx.annotate("create_NFD_from_contour", color = "red")
    def create_NFD_from_contour(self, x_vals, y_vals):
        '''
        This function takes a series of [x,y] values and converts them into a single FFT representation
        '''
        n = lambda a: int(cp.where(self.index_vect == a)[0][0]) #FIXME: np -> cp
        A = lambda a: int(self.index_vect[a])

        # store the number of samples locally for easier typing
        nsamp = self.nsamp
        max_norms = 0

        # a list of the possible normalization coefficients that might be used
        possible_k_values = cp.array([2,-1,-2,-3,-4])   #FIXME: np -> cp
        
        # initialize the different values that we are going to fill in as we calculate and normalize
        NFD = cp.zeros([possible_k_values.shape[0], nsamp], dtype = 'c16') #FIXME: np -> cp
        angle = cp.zeros([possible_k_values.shape[0]])

        # take the FFT of the input contour
        # We subtract v_vals from imsize because we are correcting for the element locations of a pixel in an image
        # this creates a 1D complex array using x and y values
        #TODO: can this be run in parallel
        #TODO: see if this runs in normal python
        with nvtx.annotate("fft call", color="blue"): # running in normal python?
            fcoord = cp.fft.fft(        #FIXME: np -> cp
                (x_vals + (self.imsize - y_vals)*1j),
                nsamp
            )
        
        

        # We shift the fft
        # (-N/2) + 1 <= i <= (N/2)
        fcoord = self.shift(fcoord)
        
        # pull out the centroid, which is the A(0) value
        # set this to zero to normalize the fft
        centroid = fcoord[n(0)]
        fcoord[n(0)] = 0
        
        # Now we want to find the value of A(1), which is the magnitude of the contour
        magnitude = abs(fcoord[n(1)])

        # then we normalize each of the values by the magnitude A(i) / A(1)
        fcoord = fcoord / magnitude
        self.testing_fcoord = fcoord[:]
        # Now, we want to find the value of k, where A(k) is the second largest magnitude in the fft
        idx = cp.argsort(abs(fcoord))   #FIXME: np -> cp
        idx = idx[::-1] # reverse the order 

        k_index = idx[1] # second largest value

        k_freq = A(idx[1]) # which frequency value is this index associated with

        # now determine the number of normalizations
        m_k = abs(k_freq - 1)

        if m_k == 0:
            raise Exception("No valid normalizations!")
        
        for norm_num, k in enumerate(possible_k_values):

            # u = phase of A(1)
            u = np.arctan2(
                        fcoord.imag[n(1)],
                        fcoord.real[n(1)]
                    )

            # v = phase of A(k)

            v = np.arctan2(
                fcoord.imag[n(k)],
                fcoord.real[n(k)]
            )

            # Calculate the specific angle instance of the rotation

            angle[norm_num] = ((v - k*u)/(k-1))*(180./np.pi)

            # calculate the rotation and shift starting point normalization for the contour
            ang = [((x - k)*u + (1 - x)*v)/(k-1) for x in np.linspace(-self.nsamp/2 + 1, self.nsamp / 2, self.nsamp)]
            coeff = [np.exp(1j*x) for x in ang]

            NFD[norm_num,:] = fcoord*coeff
        

        return centroid, magnitude, angle, NFD, m_k

     
    @nvtx.annotate("shift", color = "blue") # i dont think this one is taking long at all, just going to toss it here in case
    def shift(self,nfd):
        '''
        This replaces the np.fft.fftshift due to issues with the shifting parameters
        '''
        shift_amount = [((dim // 2) - 1) for dim in nfd.shape ]
        return np.roll(nfd, shift_amount)
    
     
    @nvtx.annotate("ishift", color = "blue")
    def ishift(self,nfd):
        '''
        This replaces np.fft.ifftshift based on our needs (off-by-one)
        '''
        shift_amount = [(-dim //2 + 1 ) for dim in nfd.shape]
        return np.roll(nfd, shift_amount)
    
     
    @nvtx.annotate("create_nfd_library", color = "green")
    def create_nfd_library(self, STLFile):
        '''
        This function creates an NFD library based on the STL and rotation indices that have been specified
        '''

        # define our rotation parameters
        xrot = np.linspace(int(-1*self.xrotmax),
                           int(self.xrotmax),
                           int((2*self.xrotmax/self.xrotinc))+1)
        
        yrot = np.linspace(int(-1*self.yrotmax),
                           int(self.yrotmax),
                           int((2*self.yrotmax/self.yrotinc))+1)

        # create all the different values that we are going to fill up
        rot_indices = np.empty([xrot.size, yrot.size, 2])
        NFD_library = np.zeros([xrot.size, yrot.size, self.max_num_norms,self.nsamp], dtype = 'c16')
        angle_library = np.zeros([xrot.size, yrot.size, self.max_num_norms])
        magnitude_library = np.zeros([xrot.size, yrot.size])
        centroid_library = np.zeros([xrot.size, yrot.size], dtype='c16')
        renWin, renderer, transformFilter, stl_mapper = self.set_visualization_scene(STLFile)
        for j, xr in enumerate(xrot):
            for k, yr in enumerate(yrot):
                rot_indices[j,k,0] = xr
                rot_indices[j,k,1] = yr

                # Should these get wrapped in the profiler also?
                # Does the time from the initial profiler add the time from each of these calls as well? Is it nested?
                xval, yval = self.create_projection(STLFile, renWin, renderer, transformFilter, stl_mapper, xr, yr, 0)
                cent,mag,ang,nfd,mk = self.create_NFD_from_contour(xval,yval)
                centroid_library[j,k] = cent
                magnitude_library[j,k] = mag
                angle_library[j,k,:] = ang
                NFD_library[j,k,:,:] = nfd
        
        self.rot_indices = rot_indices
        self.centroid_library = centroid_library
        self.magnitude_library = magnitude_library
        self.angle_library = angle_library
        self.NFD_library = NFD_library

    # probably not going to be running this because the data is a bit too hefty to add to the hipergator system.
    @nvtx.annotate("estimate_pose", color = "orange")
    def estimate_pose(self,instance):
        '''
        Given an input NFD instance, this will determine the pose
        '''

        xspan = self.NFD_library.shape[0]
        yspan = self.NFD_library.shape[1]

        dist = np.empty([xspan, yspan])

        # We have to divide by the nsamp because of the noramlization method used in the FFT

        centroid_library = self.centroid_library / self.nsamp
        centroid_instance = instance["centroid"] / self.nsamp

        mag_library = self.magnitude_library / self.nsamp
        mag_instance = instance["magnitude"] / self.nsamp

        for i in range(0,xspan):
            for j in range(0,yspan):

                diff = sum([x.real**2 + x.imag**2 for x in (instance["NFD"][1] - self.NFD_library[i,j,1,:])])
                dist[i,j] = diff
        

        idx,idy = np.where(dist == dist.min())
        #print(idx,idy)
        x_rot_est = self.rot_indices[idx,idy,0]
        y_rot_est = self.rot_indices[idx,idy,1]
        z_rot_est = (self.angle_library[idx,idy,1] - instance["angle"][1])
        z_rot_rad = z_rot_est * np.pi / 180
        z_rot_rad = z_rot_rad

        if z_rot_est[0] > 360:
            z_rot_est[0] -= 360
        if z_rot_est[0] < -360:
            z_rot_est[0] += 360

        z_lib = 0.1*self.pd
        z_est = self.pd - (self.pd - z_lib)*(mag_library[idx,idy] / mag_instance)

        x_offset = (0.5 * self.imsize) #+ (self.xo/self.sc)
        y_offset = (0.5 * self.imsize) #+ (self.yo/self.sc)

    # Calculate the zoom based on similar triangles
        zoom = (self.pd - z_lib)/(self.pd - z_est)

        x_lib = (centroid_library[idx,idy].real - x_offset)/zoom
        y_lib = (centroid_library[idx,idy].imag - y_offset)/zoom


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
        x_est = x_est * (z_est_corr / -self.pd) + self.xo
        y_est = y_est * (z_est_corr / -self.pd) + self.yo
    # Fix the rotations based on the projective geometry
        phi_x = np.arctan2(y_est, (self.pd - z_est))
        phi_y = np.arctan2(x_est, (self.pd - z_est))

        #x_rot_corr = x_rot_est + (np.cos(z_rot_rad)*phi_x - np.sin(z_rot_rad)*phi_y) * 180/np.pi
        #y_rot_corr = y_rot_est - (np.sin(z_rot_rad)*phi_x - np.cos(z_rot_rad)*phi_y)*180/np.pi
        
        rot_corr = np.array([
            [-np.cos(z_rot_rad), np.sin(z_rot_rad)],
            [np.sin(z_rot_rad), np.cos(z_rot_rad)]
        ])
        rotation_correction = np.matmul(rot_corr, np.array([[phi_x],[phi_y]]))
        
        x_rot_corr = x_rot_est - rotation_correction[0] * 180/np.pi
        y_rot_corr = y_rot_est - rotation_correction[1] * 180/np.pi
        
        
        vector1 = np.array([
            (x_lib[0])*self.sc,
            (y_lib[0])*self.sc,
            -0.9*self.pd
        ])
        
        
        vector2 = np.array([
            x_est[0],
            y_est[0],
            z_est_corr[0]
        ])
        
        
        rot_corr = two_vector_rotation_matrix(vector1, vector2)
        
        # Create rotation matrix based on the rotations at 0,0,0
        
        rot_at_origin = create_rotation_matrix_312(z_rot_est[0], x_rot_est[0], y_rot_est[0])
        
        # now, we apply the rotation from the value at the origin
        new_rot = np.matmul(rot_at_origin,rot_corr)
        
        # extract the rotations
        
        zr, xr, yr = getRotations("312",new_rot)

        return x_est[0], y_est[0], z_est_corr[0], zr, xr, yr

     
    @nvtx.annotate("save_nfd_library", color = "red")
    def save_nfd_library(self,filename):
        """ 
            Saves the necessary library variables into a dict for easier access after unpickling.
            If any data does not exist, pickling and saving is skipped, and the user is informed.
            Otherwise, the data is saved to a file 'filename'.nfd
        """  
        try:
            self.nfd_dict = {"NFD_library":self.NFD_library, "angle_library": self.angle_library,
                "mag_library": self.magnitude_library, "centroid_library": self.centroid_library,
                "rot_indices": self.rot_indices}
            filename = filename + '.nfd'
            output = open(filename, 'wb')
            pickle.dump(self.nfd_dict, output)
            pickle.dump(self.params, output)
            output.close()
        except AttributeError as error:
            print("Error!", error, "\nAll library objects must be instantiated before trying to save!")

     
    def load_nfd_library(self, pickle_path):
        """
            Loads a pickle from a passed-in file path.
            Once the pickle is loaded, saves its params to self variables in order to allow easier access of needed data.
        """
        try:
            FFTFile = open(pickle_path, 'rb')
            self.FFTPickle = pickle.load(FFTFile)
            self.NFD_library = self.FFTPickle['NFD_library']
            self.angle_library = self.FFTPickle['angle_library']
            self.magnitude_library = self.FFTPickle['mag_library']
            self.centroid_library = self.FFTPickle['centroid_library']
            self.rot_indices = self.FFTPickle['rot_indices']
            FFTFile.close()

        except FileNotFoundError:
            print("Error! The file you are trying to load either does not exist, or does not exist at this location: ", pickle_path)

    
    @nvtx.annotate("extract_contour_from_image", color = "green")
    def extract_contour_from_image(self, image):
        '''
        Extract the contour from a loaded image
        '''

        if not os.path.exists(image):
            raise Exception("Image does not exist at path: ", image)

        try:
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        except:
            print("Could not load image")
        
        # apply a contour detector on the image
        kernel = np.ones([5,5], np.uint8)
        binary = cv2.dilate(img, kernel, iterations = 1)
        binary = cv2.erode(binary, kernel, iterations = 1)

        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        done = 0
        for contour in contours:
            x,y = contour.T

            x = x.tolist()[0]
            y = y.tolist()[0]

            if len(x) >200:

                # Resample contour in nsamp equispaced increments using spline interpolation
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                tck, u = splprep([x,y], u=None, s=1.0, per=1)
                # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                u_new = np.linspace(u.min(), u.max(), self.nsamp)
                #u_new = np.linspace(u.min(), u.max(), 64)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                x_new, y_new = splev(u_new, tck, der=0)
                # Convert it back to numpy format for opencv to be able to display it
                #plt.plot(x_new,self.imsize-y_new)
                #plt.show()
                # commented out the line above to skip display window opening.
                done = 1
            if done:
                break 

        return x_new, y_new    

    
    @nvtx.annotate("set_visualization_scene", color = "blue")
    def set_visualization_scene(self, STLFile):
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

        return renWin, renderer, transformFilter, stl_mapper
    
    
    @nvtx.annotate("create_single_projection", color = "purple")
    def create_single_projection(self,STLFile, xr,yr,zr, translate = None):

        renWin, renderer, transformFilter, stl_mapper = self.set_visualization_scene(STLFile)

        return self.create_projection(STLFile, renWin, renderer, transformFilter, stl_mapper, xr, yr, zr, translate)
    
     
    @nvtx.annotate("create_single_instance", color = "orange")
    def create_single_instance(self, STLFile, xr, yr, zr, translate = None):

        x, y = self.create_single_projection(STLFile, xr, yr, zr, translate)

        centroid, magnitude, angle, NFD, m_k = self.create_NFD_from_contour(x,y)

        instance = {
            "centroid" : centroid,
            "magnitude" : magnitude,
            "angle" : angle,
            "NFD" : NFD,
            "m" : m_k
        }
        return instance

     # TODO: make some of the different colors for better visualization
    @nvtx.annotate("pose_from_segmentation", color = "purple")
    def pose_from_segmentation(self,image):
        '''
        This will take in the path to an image and return the pose at that specific value
        '''
        
        x, y = self.extract_contour_from_image(image)
        centroid, magnitude, angle, NFD, m_k = self.create_NFD_from_contour(x,y)
        
        img_inst = {
            "centroid" : centroid,
            "magnitude" : magnitude,
            "angle" : angle,
            "NFD" : NFD
        }
        
        return self.estimate_pose_better(img_inst)
    
    ## TODO: add a function that will plot the library for anyone that wants a nice visualization tool
    
     
    def estimate_pose_better(self,inst):
        '''
        Rewriting the estimate_pose library in a way that is more intuitive
        '''
        xspan = self.NFD_library.shape[0]
        yspan = self.NFD_library.shape[1]

        dist = np.empty([xspan, yspan, 5])
        
        # First, we want to grab the magnitude of the current library and 
        # normalize for all the different values
        
        # Find the location in the NFD library that is closest to the input
        
        for i in range(0,xspan):
            for j in range(0, yspan):
                
                for norm in range(0,5):
                    diff = sum([x.real**2 + x.imag**2 for x in (inst["NFD"][norm] - self.NFD_library[i,j,norm,:])])
                    dist[i,j,norm] = diff
        
        # Now, we know the rotation indices for the value at the given location
        # take the smallest distance
        idx,idy, norm_idx = np.where(dist == dist.min())
        #print(self.rot_indices[idx,idy])
        #print(self.rot_indices[idx,idy].shape)
        # This gives us values for all the different library indices 
        
        # Pull out the magnitude of the instance and the matching library value
        
        mag_inst = inst["magnitude"]
        mag_lib = self.magnitude_library[idx,idy]
        
        # determine the z translation
        # define z_lib translation
        z_lib = 0.1 * self.pd
        z_est = self.pd - (self.pd - z_lib)*(mag_lib/mag_inst) # this inherits units from self.pd
        
        # FINAL VALUE THAT GETS RETURNED
        z_final = z_est[0] - self.pd
        if abs(z_final) > abs(self.pd):
            z_final = -self.pd # this makes sure that the value is not beyond the image plane
            
        # Now we determine the rotations based on the caluculated values from the library and instance
        
        x_rot_est = self.rot_indices[idx,idy,0][0]
        y_rot_est = self.rot_indices[idx,idy,1][0]
        #print(x_rot_est, idx, "\n", y_rot_est, idy)
        z_rot_est = (self.angle_library[idx,idy,norm_idx] - inst["angle"][norm_idx])[0]
        z_rot_rad = z_rot_est * np.pi / 180
        
        # Now we determine the x,y translations 
        
        # start with the two centroid values
        
        cent_inst = inst["centroid"]
        cent_lib = self.centroid_library[idx,idy]
        
        x_in = (cent_inst.real / self.nsamp) - 512 # correcting for FFT normalization and setting center of image as origin
        y_in = (cent_inst.imag / self.nsamp) - 512
        #print("Centroid_in ",x_in, y_in)
        x_lib = (cent_lib.real / self.nsamp) - 512
        y_lib = (cent_lib.imag / self.nsamp) - 512
        #print("Centroid_lib: ", x_lib, y_lib)
        cz = np.cos(z_rot_rad)
        sz = np.sin(z_rot_rad)
        centroid_zoom = (self.pd - z_lib)/(self.pd - z_est)
        #print("zoom factoprs: ",centroid_zoom, (inst["magnitude"]/self.magnitude_library[idx,idy]))
        
        x_est_px = x_in - (cz*x_lib - sz*y_lib)*(centroid_zoom)
        y_est_px = y_in - (sz*x_lib + cz*y_lib)*(centroid_zoom)
        
        # convert to mm
        x_est_mm = x_est_px * self.sc
        y_est_mm = y_est_px * self.sc
        #print(x_est_mm, y_est_mm)
        # adjust for projection
        x_final = x_est_mm * ((self.pd - z_est)/self.pd)# - self.xo # might need to change caliblration offset to a plus/minus
        y_final = y_est_mm * ((self.pd - z_est)/self.pd)# - self.yo
        #print(x_final, y_final, ((self.pd - z_est)/self.pd))
        # now to adjust for rotation estimates based on the perspective shift
        # going to correct by making an axis-angle representation
        
        v1 = np.array([
            0,
            0,
            self.pd - z_est[0]
        ])
        
        v2 = np.array([
            x_final[0],
            y_final[0],
            self.pd - z_est[0]    
        ])

        rot_adjusting = two_vector_rotation_matrix(v1,v2)
        #rot_adjusting = np.linalg.inv(rot_adjusting_temp)
        rot_at_center = create_rotation_matrix_312(z_rot_est,x_rot_est,y_rot_est)
        
        # perform a rotation based on the global axes defined in rot_adjusting
        
        final_rot = np.matmul(rot_adjusting, rot_at_center) # might need to tinker with the negative values in here to get it to work better
        
        z_rot, x_rot_final, y_rot_final = getRotations("312",final_rot)
        
        phi_x = np.arctan2(y_final, abs(z_final))
        phi_y = np.arctan2(x_final, abs(z_final))
        
        #x_rot_final = (x_rot_est + np.rad2deg((-cz*phi_x + sz*phi_y)))[0]
        #y_rot_final = (y_rot_est + np.rad2deg((sz*phi_x + cz*phi_y)))[0]
        if z_rot_est < -180:
            z_rot_final = 360 + z_rot_est
        elif z_rot_est > 180:
            z_rot_final = z_rot_est - 360
        else:
            z_rot_final = z_rot_est
        
        # need to make sure that each of the values get stored correctly
        try:
            x_final = x_final[0]
        except:
            pass
        
        try:
            y_final = y_final[0]
        except:
            pass
        
        try:
            z_final = z_final[0]
        except:
            pass
        
        return x_final, y_final, z_final, z_rot_final, x_rot_final, y_rot_final
    
     
    def print_library(self, norm_coeff, clock_arm_length, scale):
        """
        This will print the fourier library for visualization purposes
        """
        fig,ax = plt.subplots(figsize = (13,13))
        
        idx,jdx, _ = self.rot_indices.shape
        norm = norm_coeff
        
        for i in range(0,idx):
            for j in range(0,jdx):
            
                inv = self.ishift(self.NFD_library[i,j][norm])
                inv = np.fft.ifft(inv)
                angle = self.angle_library[i,j][norm] * np.pi / 180
                x_cent = self.rot_indices[i,j][0]
                y_cent = self.rot_indices[i,j][1]
                clock_x = [x_cent,x_cent + np.cos(angle)*clock_arm_length ]
                clock_y = [y_cent, y_cent + np.sin(angle)*clock_arm_length]

                x = inv.real*scale + x_cent
                y = inv.imag*scale + y_cent

                ax.plot(x,y, linewidth = 4)
                ax.plot(clock_x, clock_y, linewidth = 2, color = 'black')
                ax.plot(x_cent, y_cent, marker = "x", color = "black", markersize = 10)
                ax.set_xlabel("X-rotation", size = 35)
                plt.xticks(fontsize = 25)
                ax.set_ylabel("Y-rotation", size = 35)
                plt.yticks(fontsize = 25)
    
     
    def print_instance(self, instance, norm_coeff, clock_arm_length, scale):
        fig, ax = plt.subplots(figsize = (13,13))
        norm = norm_coeff
        inv = self.ishift(instance["NFD"][norm])
        inv = np.fft.ifft(inv)
        angle = instance["angle"][norm] * np.pi / 180
        x_cent = 0
        y_cent = 0
        clock_x = [x_cent,x_cent + np.cos(angle)*clock_arm_length ]
        clock_y = [y_cent, y_cent + np.sin(angle)*clock_arm_length]
        x = inv.real*scale + x_cent
        y = inv.imag*scale + y_cent
        ax.plot(x,y, linewidth = 4)
        ax.plot(clock_x, clock_y, linewidth = 2, color = 'black')
        ax.plot(x_cent, y_cent, marker = "x", color = "black", markersize = 10)
        ax.set_xlabel("X-rotation", size = 35)
        plt.xticks(fontsize = 25)
        ax.set_ylabel("Y-rotation", size = 35)
        plt.yticks(fontsize = 25)