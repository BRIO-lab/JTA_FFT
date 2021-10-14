#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:52:31 2021

@author: SAB
"""
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
    def __init__ (self,CalFile):#, lib_config):
        abcdefghij = 1
        nsamp = 128
        self.nsamp = nsamp

        self.imsize = 1024

        cal_data = np.loadtxt(CalFile, skiprows=1)

        pd = cal_data[0]
        sc = cal_data[3]
        xo = cal_data[1]
        yo = cal_data[2]
        self.pd = pd
        self.xo = xo
        self.yo = yo
        self.sc = sc

        #self.lib_config = lib_config

    def MakeLib (self, CalFile, STLFile,dir):
        plt.clf()
        self.CalFile = CalFile
        self.STLFile = STLFile
        
        

    # truncate the contents of the text file that stores outputted data
        # txtFile = open(r"output_data.txt","w")
        # txtFile.truncate(0)
        # txtFile.write("row, column, x rotation, y rotation\n")
        # txtFile.close()
# 
    # Setup NFD library config contours represented by 128 samples
        nsamp = 128         # Normalized contours represented by 128 samples
        xrotmax = 30        # X rotation max in degrees - assumes library will be symmetric +/-
        xrotinc = 3        # x rotation increment in degrees
        yrotmax = 30        # y rotation max in degrees - assumes library will be symmetric +/-
        yrotinc = 3        # y rotation increment in degrees
        # self.nsamp = nsamp
    # Assume image size is 1024x1024 pixels
        self.imsize = 1024    
        
    # Set up projection geometry based on calibration file
    # This program SHOULD read in projection geometry from a JointTrack 
    # calibration file, but for now the parameters are just hard-coded 
    # for demo purposes.
    # pd, sc, xo, yo = ReadCalFile(CalFile)

        # pd = 1200   # nominal prin dist in mm
        # sc = 0.373  # nominal pixel dimension in mm
        # xo = 0.0    # x offset for principal point
        # yo = 0.0    # y offiset for principal point
        
        pd = self.CalFile[0]
        sc = self.CalFile[3]
        xo = self.CalFile[1]
        yo = self.CalFile[2]
        self.pd = pd
        self.xo = xo
        self.yo = yo
        self.sc = sc

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
        
        rot_indices = np.empty([xrot.size, yrot.size,2])
        for j ,xr in enumerate(xrot):
            
            for k, yr in enumerate(yrot):
                #print(j,k,xr,yr)
                rot_indices[j,k,0] = xr
                rot_indices[j,k,1] = yr
            # save the row, column, x rotation, and y rotation into the output text file
                # txtFile = open(r"output_data.txt","a")
                # txtFile.write("%d,%d,%f,%f\n"%(j,k,xr,yr))
                # txtFile.close()
# 
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
                #k +=1
            #j +=1
        # txtFile = open(r"output_data.txt","a")
        # txtFile.write("xout, yout\n"+str(self.xout)+",\n"+str(self.yout)+"\n")
        # txtFile.close()
        np.save(dir + "/rot_indices.npy", rot_indices)
        return self.xout, self.yout

    def NFD_Lib(self, dir, model_type):
        x = self.xout
        y = self.yout
    # Get dimensions of input contours
        r, c, nsamp = np.shape(x)

    # write header for next set of data
        # txtFile = open(r"output_data.txt","a")
        # txtFile.write("dc,mag,lib_angle,surface\n")
        # txtFile.close()
# 
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

                ###################################
                ## FIX THIS TO NORMALIZE PROPERLY##
                ###################################
                ## norm = "ortho" ## put this into the np.fft.fft call ##
                fcoord = np.fft.fft((x_new+(self.imsize-y_new)*1j),nsamp)

                ###################################
                ## FIX THIS TO NORMALIZE PROPERLY##
                ###################################
                fcoord = np.fft.fftshift(fcoord)    # shift so DC is in center
    #            dc[i,j] = abs(fcoord[int(nsamp/2)])
                dc[i,j] = (fcoord[int(nsamp/2)])
                fcoord[int(nsamp/2)] = 0            # normalize x,y position
                idx = np.argsort(abs(fcoord))       # sort fft coeffs by magnitude
                idx = idx[::-1]                     # sort descending
                num_norms = abs(idx[1]-nsamp/2-1)   # number of normalizations
                #print('Number of normalizations:',num_norms, 'Index',idx[1])

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
        #txtFile = open(r"output_data.txt","a")
    # newlines added after commas for readability, remove to allow for easier data processing if needed
        # txtFile.write(str(dc) + ",\n" + str(mag) + ",\n" + str(lib_angle) + ",\n" + str(surface))
        # txtFile.close()
# 
        np.save(dir + "/surface_" + model_type + ".npy",surface)
        np.save(dir + "/dc_" + model_type + ".npy",dc)
        np.save(dir + "/mag_" + model_type + ".npy",mag)
        np.save(dir + "/lib-angle_" + model_type + ".npy",lib_angle)

        return dc, mag, lib_angle, surface

    def create_contour(self,image):
        # print(self.nsamp)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(image, kernel, iterations=1)   # use dilate/erode to get rid of small spurious gaps
        binary = cv2.erode(binary, kernel, iterations=1)
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
                u_new = np.linspace(u.min(), u.max(), self.nsamp)
                #u_new = np.linspace(u.min(), u.max(), 64)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                x_new, y_new = splev(u_new, tck, der=0)
                # Convert it back to numpy format for opencv to be able to display it
                # plt.plot(x_new,self.imsize-y_new)
                # plt.show()
                # commented out the line above to skip display window opening.
                done = 1
            if done:
                break

        return x_new, y_new

    def get_NFD(self,x,y):
        nsamp = self.nsamp
        
        kmax = 5
        k_norm = np.array([2,-1,-2,-3,-4])

        # dc      = np.zeros((r,c),dtype='c16')
        # mag     = np.zeros((r,c))
        surface = np.zeros((kmax,nsamp),dtype='c16')
        lib_angle = np.zeros((kmax))
        index_vect = np.linspace(1,nsamp,nsamp)-nsamp/2

        max_norms = 0

        fcoord = np.fft.fft((x + (self.imsize-y)*1j),nsamp)
        fcoord = np.fft.fftshift(fcoord)

        dc = (fcoord[int(nsamp/2)])
        fcoord[int(nsamp/2)] = 0 
        idx = np.argsort(abs(fcoord))       # sort fft coeffs by magnitude
        idx = idx[::-1]                     # sort descending
        num_norms = abs(idx[1]-nsamp/2-1)

        if num_norms == 0:
            print('No valid normalizations')
            dc         = 0.0
            mag        = 0.0
            surface[:,:] = 0.0
            lib_angle[:] = 0.0
            return
        
        if num_norms > max_norms:
            max_norms = num_norms

        mag = abs(fcoord[int(nsamp/2+1)])
        fcoord = fcoord/mag

        for norm in range(int(num_norms)):
            k = k_norm[norm]
            # Compute phase angles of A(1) and A(k)
            u = np.arctan2(fcoord.imag[int(nsamp/2+1)],fcoord.real[int(nsamp/2+1)])
            v = np.arctan2(fcoord.imag[int(idx[1])],fcoord.real[int(idx[1])])
            print('u',u,'v',v)
            # Save reference angle for library matching - in degrees
            lib_angle[norm] = ((v-k*u)/(k-1))*180./(math.pi)
            # Compute complex angle to standardize in-plane rotation and contour starting point
            angle = ((index_vect - k)*u + (1-index_vect)*v)/(k-1)
            coeff = np.cos(angle)+np.sin(angle)*1j
            print('Angle:',angle,'coeff',coeff)
            # Finish normalization
            surface[norm,:] = fcoord*coeff

        max_norms = int(max_norms)
        surface = surface[0:max_norms,:]
        lib_angle = lib_angle[0:max_norms]

        return dc, mag, lib_angle, surface

    def estimate_pose(self,rot_indices, known_dc,known_mag,known_lib_angle,known_surface, uk_dc,uk_mag,uk_libangle,uk_surface):
        xspan = known_surface.shape[0]
        yspan = known_surface.shape[1]
        dist = np.empty([xspan,yspan])

        known_dc = known_dc / self.nsamp
        uk_dc = uk_dc / self.nsamp

        known_mag = known_mag / self.nsamp
        uk_mag = uk_mag / self.nsamp

        
        for i in range(0,xspan):
            for j in range(0,yspan):
                diff = uk_surface[0,:] - known_surface[i,j,0,:]
                #print(diff)
                #print("TEST TEST" , np.linalg.norm(diff))

                dist[i,j] = np.linalg.norm(diff)

               # dist[i,j] = np.transpose(np.conjugate(diff))
        
        idx,idy = np.where(dist == dist.min())

        x_rot_est = rot_indices[idx,idy,0]
        y_rot_est = rot_indices[idx,idy,1]

        z_rot_est = uk_libangle[0] - known_lib_angle[idx,idy,0]
        #z_rot_est = -uk_libangle[0] + known_lib_angle[idx,idy,0]
        zrot_est_rad = z_rot_est[0]*np.pi/180

        lib_z = 0.1 * self.pd

        #z_trans_est = self.pd - (known_mag[idx,idy]/uk_mag)*(self.pd - (lib_z))
        z_trans_est = self.pd -(self.pd - lib_z) * (known_mag[idx,idy] / uk_mag)

        #using similar triangles, we can estimate the z translation - look at banks code

        # X and Y translations

        x_trans = 0.5 * self.imsize + (self.xo*self.imsize)/self.sc
        y_trans = 0.5 * self.imsize + (self.yo*self.imsize)/self.sc

        zoom = (self.pd - lib_z)/(self.pd - z_trans_est)

        x_dc = (known_dc[idx,idy].real - x_trans) * zoom
        y_dc = (known_dc[idx,idy].imag - y_trans) * zoom

        rot = np.array([[math.cos(zrot_est_rad),-math.sin(zrot_est_rad)],[math.sin(zrot_est_rad), math.cos(zrot_est_rad)]])
        t_input = np.array([[uk_dc.real - x_trans],[uk_dc.imag - y_trans]])
        t_lib = np.array([[x_dc[0]],[y_dc[0]]])
        t_est_px = t_input - np.matmul(rot,t_lib)*zoom
        t_est_mm = t_est_px * self.sc
        # t_est_mm = t_est_px / 128

        x_est, y_est = t_est_mm

        
        #x_trans_est = np.array([[uk_dc.real - x_trans],[uk_dc.imag - y_trans]]) - np.matmul()
        #x_trans_est = (uk_dc.real - x_trans) - (math.cos(zrot_est_rad)*x_dc + math.sin(zrot_est_rad)*y_dc)*self.sc/self.imsize
        #y_trans_est = (uk_dc.imag - y_trans) - (math.sin(zrot_est_rad)*x_dc - math.cos(zrot_est_rad)*y_dc)*self.sc/self.imsize

       # x_trans_est = x_trans_est*(self.pd-lib_z)/(self.pd - z_trans_est)
       # y_trans_est = y_trans_est*(self.pd-lib_z)/(self.pd - z_trans_est)

        #x_est = ((uk_dc.real - x_trans) - (math.cos(zrot_est_rad)*x_dc - math.sin(zrot_est_rad)*y_dc)*zoom)*(self.sc/self.imsize)
        #y_est = ((uk_dc.imag - y_trans) - (math.sin(zrot_est_rad)*x_dc + math.cos(zrot_est_rad)*y_dc)*zoom)*(self.sc/self.imsize)

        z_trans_corr = z_trans_est - self.pd


        # compensate for x and y rotations

        phi_x = math.atan2(y_est, (self.pd - z_trans_est))*np.pi/180
        phi_y = math.atan2(x_est, (self.pd - z_trans_est))*np.pi/180

        x_rot_corr = x_rot_est + math.cos(zrot_est_rad)*phi_x - math.sin(zrot_est_rad)*phi_y
        y_rot_corr = y_rot_est - math.sin(zrot_est_rad)*phi_x - math.cos(zrot_est_rad)*phi_y
         


        return x_est[0],y_est[0],z_trans_corr[0],-1*z_rot_est[0], x_rot_corr[0], y_rot_corr[0]
        
        # you want the index of the smallest value (i and j)

        

