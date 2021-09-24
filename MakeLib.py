#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:52:31 2021

@author: SAB
"""
import vtk
import numpy as np
from vtk.util import numpy_support
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def MakeLib (CalFile,STLFile):

    plt.clf()

# Setup NFD library config contours represented by 128 samples
    nsamp = 128         # Normalized contours represented by 128 samples
    xrotmax = 30        # X rotation max in degrees - assumes library will be symmetric +/-
    xrotinc = 10        # x rotation increment in degrees
    yrotmax = 30        # y rotation max in degrees - assumes library will be symmetric +/-
    yrotinc = 10        # y rotation increment in degrees
    
# Assume image size is 1024x1024 pixels
    imsize = 1024    
    
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
    cx = imsize/2  # project to image center
    cy = cx        # assume square image
    w = imsize     # width = imsize
    h = imsize     # height = imsize
        
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
    angle = (180 / np.pi) * 2.0 * np.arctan2(imsize/2.0, fy)
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
    stl_reader.SetFileName(STLFile)

# Initialize VTK transform filter and mapper
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(stl_reader.GetOutputPort())

    stl_mapper = vtk.vtkPolyDataMapper()
    stl_mapper.SetInputConnection(transformFilter.GetOutputPort())

# define rotation ranges for library
    xrot = np.linspace(int(-1.0*xrotmax),int(xrotmax),int((2*xrotmax/xrotinc)+1))
    yrot = np.linspace(int(-1.0*yrotmax),int(yrotmax),int((2*yrotmax/yrotinc)+1))

    xout = np.zeros((int((2*xrotmax/xrotinc)+1),int((2*yrotmax/yrotinc)+1),nsamp))
    yout = np.zeros((int((2*xrotmax/xrotinc)+1),int((2*yrotmax/yrotinc)+1),nsamp))
    
    j=0
    for xr in xrot:
        k=0
        for yr in yrot:
            print(j,k,xr,yr)
 
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
                    plt.plot(x_new,imsize-y_new)
                    plt.show()
                    done = 1
                if done:
                    break

            xout[j,k,:] = x_new
            yout[j,k,:] = y_new
            k +=1
        j +=1
    return xout, yout
        
#cv2.destroyAllWindows()