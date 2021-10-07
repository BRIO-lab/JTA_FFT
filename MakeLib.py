#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:52:31 2021

@author: SAB
"""
import vtk
import numpy as np
import math
from vtk.util import numpy_support
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

class JTA_FFT ():
    def __init__ (self, CalFile, STLFile):#, lib_config):
        self.CalFile = CalFile
        self.STLFile = STLFile

        #self.lib_config = lib_config

    def MakeLib (self):
        plt.clf()

    # truncate the contents of the text file that stores outputted data
        txtFile = open(r"output_data.txt","w")
        txtFile.truncate(0)
        txtFile.write("row, column, x rotation, y rotation\n")
        txtFile.close()

    # Setup NFD library config contours represented by 128 samples
        nsamp = 128         # Normalized contours represented by 128 samples
        xrotmax = 30        # X rotation max in degrees - assumes library will be symmetric +/-
        xrotinc = 10        # x rotation increment in degrees
        yrotmax = 30        # y rotation max in degrees - assumes library will be symmetric +/-
        yrotinc = 10        # y rotation increment in degrees
        
    # Assume image size is 1024x1024 pixels
        self.imsize = 1024    
        
    # Set up projection geometry based on calibration file
    # This program SHOULD read in projection geometry from a JointTrack 
    # calibration file, but for now the parameters are just hard-coded 
    # for demo purposes.
    # pd, sc, xo, yo = ReadCalFile(CalFile)

        pd = 1200   # nominal prin dist in mm
        sc = 0.373  # nominal pixel dimension in mm
        xo = 0.0    # x offset for principal point
        yo = 0.0    # y offiset for principal point
        
        isc = 1/sc     # inverse scale
        fx = pd/sc     # scale prin dist into pixel units
        fy = fx        # same prin dist in x,y
        cx = self.imsize/2  # project to image center
        cy = cx        # assume square image
        w = self.imsize     # width = imsize
        h = self.imsize     # height = imsize
            
    # STL render setup
        renderer = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.SetOffScreenRendering(1)

    # Set basic camera parameters in VTK and clipping planes
        cam = renderer.GetActiveCamera()
        near = 0.1
        far = 1.5*pd/sc
        cam.SetClippingRange(near, far)
            
    # Position is at origin, looking in -z direction with y up
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0, 0, -1)
        cam.SetViewUp(0, 1, 0)
        cam.SetWindowCenter(0,0)
            
    # Set vertical view angle as a indirect way of setting the y focal distance
        angle = (180 / np.pi) * 2.0 * np.arctan2(self.imsize/2.0, fy)
        cam.SetViewAngle(angle)
            
    # Set the image aspect ratio as an indirect way of setting the x focal distance
        m = np.eye(4)
        aspect = fy/fx
        m[0,0] = 1.0/aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        cam.SetUserTransform(t)

    # STL render setup - again, seems like it needs kicked twice to update?
        renderer = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.SetOffScreenRendering(1)

    # Set basic camera parameters in VTK
        cam = renderer.GetActiveCamera()
        cam.SetClippingRange(near, far)
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0, 0, -1)
        cam.SetViewUp(0, 1, 0)
        cam.SetWindowCenter(0,0)
        cam.SetViewAngle(angle)
        cam.SetUserTransform(t)
            
    # Load in STL file using VTK
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(self.STLFile)

    # Initialize VTK transform filter and mapper
        transformFilter=vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(stl_reader.GetOutputPort())

        stl_mapper = vtk.vtkPolyDataMapper()
        stl_mapper.SetInputConnection(transformFilter.GetOutputPort())

    # define rotation ranges for library
        xrot = np.linspace(int(-1.0*xrotmax),int(xrotmax),int((2*xrotmax/xrotinc)+1))
        yrot = np.linspace(int(-1.0*yrotmax),int(yrotmax),int((2*yrotmax/yrotinc)+1))

        self.xout = np.zeros((int((2*xrotmax/xrotinc)+1),int((2*yrotmax/yrotinc)+1),nsamp))
        self.yout = np.zeros((int((2*xrotmax/xrotinc)+1),int((2*yrotmax/yrotinc)+1),nsamp))
        
        j=0
        for xr in xrot:
            k=0
            for yr in yrot:
                print(j,k,xr,yr)

            # save the row, column, x rotation, and y rotation into the output text file
                txtFile = open(r"output_data.txt","a")
                txtFile.write("%d,%d,%f,%f\n"%(j,k,xr,yr))
                txtFile.close()

            # STL transform model
                transform = vtk.vtkTransform()
                transform.PostMultiply()
                transform.Scale(isc,isc,isc)    # Scale into pixel units
                transform.RotateY(yr)        
                transform.RotateX(xr)
                transform.Translate(0,0,-0.9*fx)
                transformFilter.SetTransform(transform)
                transformFilter.Update()
                
            # STL mapper
                stl_actor = vtk.vtkActor()
                stl_actor.SetMapper(stl_mapper)

            # STL render
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
                width, height, _ = vtk_image.GetDimensions()
                vtk_array = vtk_image.GetPointData().GetScalars()
                components = vtk_array.GetNumberOfComponents()
                arr = cv2.flip(numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components), 0)
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(arr, 1, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3,3), np.uint8)
                binary = cv2.dilate(binary, kernel, iterations=1)   # use dilate/erode to get rid of small spurious gaps
                binary = cv2.erode(binary, kernel, iterations=1)
            # Get contours of blob
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                smoothened = []
                done = 0
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
                        # plt.show()
                        # commented out the line above to skip display window opening.
                        done = 1
                    if done:
                        break

                self.xout[j,k,:] = x_new
                self.yout[j,k,:] = y_new
                k +=1
            j +=1
        return self.xout, self.yout

    def NFD_Lib(self):
        x = self.xout
        y = self.yout
    # Get dimensions of input contours
        r, c, nsamp = np.shape(x)

    # write header for next set of data
        txtFile = open(r"output_data.txt","a")
        txtFile.write("dc,mag,lib_angle,surface\n")
        txtFile.close()

    # Set up library variables

    #    surface = np.zeros(num_k,nsamp,rxinc,ryinc)  ; Library variables
    #    mag = fltarr(rxinc,ryinc)
    #    dc = complexarr(rxinc,ryinc)
    #    lib_angle = fltarr(num_k,rxinc,ryinc)

        kmax = 5    # arbitrary number of normalizations - probably ends up being 2 or 3
        k_norm = np.array([2,-1,-2,-3,-4])

        dc      = np.zeros((r,c),dtype='c16')
        mag     = np.zeros((r,c))
        surface = np.zeros((r,c,kmax,nsamp),dtype='c16')
        lib_angle = np.zeros((r,c,kmax))
        index_vect = np.linspace(1,nsamp,nsamp)-nsamp/2

    #    px = np.linspace(0, 2 * np.pi, nsamp)
        max_norms = 0
        for i in range(r):
            for j in range(c):
                x_new = x[i,j,:]
                y_new = y[i,j,:]
                fcoord = np.fft.fft((x_new+(self.imsize-y_new)*1j),nsamp)
                fcoord = np.fft.fftshift(fcoord)    # shift so DC is in center
    #            dc[i,j] = abs(fcoord[int(nsamp/2)])
                dc[i,j] = (fcoord[int(nsamp/2)])
                fcoord[int(nsamp/2)] = 0            # normalize x,y position
                idx = np.argsort(abs(fcoord))       # sort fft coeffs by magnitude
                idx = idx[::-1]                     # sort descending
                num_norms = abs(idx[1]-nsamp/2-1)   # number of normalizations
                print('Number of normalizations:',num_norms, 'Index',idx[1])

                if num_norms == 0:
                    print('No valid normalizations')
                    dc[i,j]         = 0.0
                    mag[i,j]        = 0.0
                    surface[i,j,:,:] = 0.0
                    lib_angle[i,j,:] = 0.0
                    break
                
                if num_norms > max_norms:
                    max_norms = num_norms
                    
                mag[i,j] = abs(fcoord[int(nsamp/2+1)])  # A(1) term to normalize size
                fcoord = fcoord/mag[i,j]                # Normalize for magnitude A(1)
                for norm in range(int(num_norms)):
                    k = k_norm[norm]
                    # Compute phase angles of A(1) and A(k)
                    u = np.arctan2(fcoord.imag[int(nsamp/2+1)],fcoord.real[int(nsamp/2+1)])
                    v = np.arctan2(fcoord.imag[int(idx[1])],fcoord.real[int(idx[1])])
    #                print('u',u,'v',v)
                    # Save reference angle for library matching - in degrees
                    lib_angle[i,j,norm] = ((v-k*u)/(k-1))*180./(math.pi)
                    # Compute complex angle to standardize in-plane rotation and contour starting point
                    angle = ((index_vect - k)*u + (1-index_vect)*v)/(k-1)
                    coeff = np.cos(angle)+np.sin(angle)*1j
    #                print('Angle:',angle,'coeff',coeff)
                    # Finish normalization
                    surface[i,j,norm,:] = fcoord*coeff

        max_norms = int(max_norms)
        surface = surface[:,:,0:max_norms,:]
        lib_angle = lib_angle[:,:,0:max_norms]

    # writing the returned data to the output text file
        txtFile = open(r"output_data.txt","a")
    # newlines added after commas for readability, remove to allow for easier data processing if needed
        txtFile.write(str(dc) + ",\n" + str(mag) + ",\n" + str(lib_angle) + ",\n" + str(surface))
        txtFile.close()

        return dc, mag, lib_angle, surface
