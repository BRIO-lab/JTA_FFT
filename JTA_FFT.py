# JTA_FFT.py
# Copyright (c) Scott Banks banks@ufl.edu

# Imports
from numpy.fft.helper import fftshift
from numpy.lib.nanfunctions import _nansum_dispatcher
import vtk
import numpy as np
import math
from vtk.util import numpy_support
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


class JTA_FFT():

    # Initialize the class
    # TODO: determine the objects that we want this class to inherit 
    # I think that the best object to inherit would be the NFD library
    def __init__(self, CalFile):
        # TODO: replace this with better initializations

        # Some of the coding variables

        # Number of Fourier Coefficients
        nsamp = 128
        self.nsamp = nsamp

        # Library Increment Parameters
        self.xrotmax = 30
        self.xrotinc = 3
        self.yrotmax = 30
        self.yrotinc = 3

        # Image Size
        self.imsize = 1024

        # Load in calibration file
        # TODO: perform a check to make sure that the 
        #       calibration file is correctly formatted
        cal_data = np.loadtxt(CalFile, skiprows=1)

        # Extracting the four components from the calibration file

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
    
    def Make_Contour_Lib(self,CalFile,STLFile,dir):

        # TODO: add function description 

        plt.clf()

        self.CalFile = CalFile
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
        cam.setFocalPoint(0, 0, -1)
        cam.setViewUp(0, 1, 0)
        cam.setWindowCenter(0, 0)
    
    # Set vertical view angle as an indirect way of 
    # setting the y focal distance
        angle = (180 / np.pi) * 2.0 * np.arctan2(self.imsize/2, fy)
        cam.setViewAngle(angle)

    # Set vertical view angle as an indirect way of
    # setting the x focal distance

        m = np.eye(4)
        aspect = fy/fx
        m[0,0] = 1.0/aspect
        t = vtk.vtkTransform()
        t.setMatrix(m.flatten())
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

                renderer.addActor(stl_actor)
                renderer.SetBackground(0.0, 0.0, 0.0)
                renWin.AddRenderer(renderer)
                renWin.SetSize(w,h)
                renWin.Render()

            # Render the scene into a numpy array for openCV processing
                winToIm = vtk.vtkWindowToImageFilter()
                winToIm.SetInput(renWin)
                winToIm.Update()
                vtk_image = winToIm.GetOutput()
                width, height = vtk_image.GetDimensions()
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
                smoothened = []
                done = 0

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
                        u_new = np.linspace(u.min(), u.max(), nsamp)
                    
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                        x_new, y_new = splev(u_new, tck, der=0)
                        
                    # Convert it back to numpy format for opencv to be able to display it
                        plt.plot(x_new,self.imsize-y_new)

                    # break code if you are done

                        done = 1

                    if done:
                        break

                self.xout[j,k,:] = x_new
                self.yout[j,k,:] = y_new
        

        # Save the rotation indices in the working directory
        np.save(dir + "/rot_indices.npy", rot_indices)

        return self.xout, self.yout


# Now we want to write a function that will create the 
# Normalized Fourier Descriptor Library

    def Create_NFD_Library(self, dir, model_type):
        # TODO: fix the loading of different types for each of the sub-functions

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
        centroid = np.zeros([r,c],dtype='c16')
    
    # Magnitude of Projection (px)
    # In Banks 96, this is S(1)
        mag = np.zeros([r,c])

    

