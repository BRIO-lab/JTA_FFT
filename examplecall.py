from MakeLib import MakeLib
from NFD_Lib import NFD_Lib

xout, yout = MakeLib('','3D Models/tib/ASCII_KR_left_8_tib.stl')

dc, mag, lib_angle, surface = NFD_Lib(xout,yout,1024)



#hi
