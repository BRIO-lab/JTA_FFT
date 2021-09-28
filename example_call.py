from MakeLib import MakeLib
from NFD_Lib import NFD_Lib

xout, yout = MakeLib('','3D Models/tib/ASCII_KR_left_8_tib.stl')
dc, mag, lib_angle, surface = NFD_Lib(xout,yout,1024)

# to run, open a powershell in vsCode or through windows, and type 'python examplecall.py' after navigating to G:/JTA_FFT (or wherever that folder is located)