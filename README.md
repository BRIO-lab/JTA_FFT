# Joint Track Auto Fast Fourier Transform
MakeLib.py - this takes in an object model and projects it at various angles and generates resampled (x,y) contours.
NFD_Lib.py - this takes in the array of (x,y) contours generated by MakeLib and creates the variables needed for the Normalized-Fourier Descriptor-based matching algorithm.

## Pre-run steps:
***I would recommend running all commands and code in powershell, as it is what has worked for multiple people at this point. Install dependencies as needed, the commands used are below.*** 
*(these may not need to be installed depending on your machine)*

*Note: if the installation fails due to insufficient permissions, add "--user" after install, and before the name of the package (e.g. pip install --user vtk)*

**`pip install --upgrade pip` (this is to make sure your pip version is up to date, ***definitely*** do this)**

`pip install vtk`

`pip install opencv-python`

`pip install scipy`

`pip install matplotlib`

## Steps to run:

To run `MakeLib.py` and `NFD_Lib.py`, create a *new .py* file in JTA_FFT - this will call the two other files. You can name it whatever you like, I used `example_call.py`

Place the following code in the example call file you created:

`from MakeLib import MakeLib`

`from NFD_Lib import NFD_Lib`

`xout, yout = MakeLib('','3D Models/tib/ASCII_KR_left_8_tib.stl')` <-This line generates a figure of the .STL file you are projecting

`dc, mag, lib_angle, surface = NFD_Lib(xout,yout,1024)` <- This line generates the NFD library variables

Once your call file is created, simply go into your powershell, navigate to the location of the call file, and run the command `python [$CALL_FILE_NAME].py`