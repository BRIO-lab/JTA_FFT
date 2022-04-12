README for the code of the article Optimization of
Image B-spline Interpolation for GPU Architectures

* Author    : DAVY Axel <axel.davy@ens-cachan.fr>
* Copyright : (C) 2019 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : BSD 3-Clause

------------
DESCRIPTION

This code implements image B-spline interpolation using OpenCL and Python.

The python code supports several common usages of B-spline interpolation.
See 'python splinecl.py -h' for the list of available commands and their
description.

The code uses the BSD 3-Clause License, and thus the OpenCL code can easily
be extracted and used in other projects (provided the respect of the
license's conditions).

----
REQUIREMENTS

The Python library requirements are described in requirements.txt
Optional Python libraries: tifffile (to read/write tiffs)
Alternative libraries to imageio: cv2 or piio

The code requires an OpenCL driver (and an available OpenCL device) to run.
To run on CPU, a driver such as Pocl can be used. For GPUs, see the
documentation of your GPU vendor on how to setup OpenCL.

For systems with several OpenCL devices, the device to use can be specified
with the env var PYOPENCL_CTX.
For example 'export PYOPENCL_CTX=0' to use the first device.

-----
USAGE

List all available options:
$ python splinecl.py -h

Further options can be found for each respective application. For example:
$ python splinecl.py zoom -h

The following command computes the shift of (`shiftx`, `shifty`) of `input.png` using
the boundary condition defined by `bc`, the result is saved in `output.png`:
$ python splinecl.py shift input.png output.png shiftx shifty spline bc 0.01

Example of calls:

Shift Lena by 0.5 on the x axis with B-spline order 3 and half-symmetric boundary condition
$ python splinecl.py shift Lenna.png Lenna_shift05.png 0.5 0 3 1 0.01

Zoom Lena by 2.5 with B-spline order 3
$ python splinecl.py zoom Lenna.png Lenna_zoom25.png 2.5 2.5 3 1 0.01

-----
FILES

This project contains the following source files:

    main function:       splinecl.py
    utilities functions: utils.py
    OpenCL functions:    spline_kernel.cl
                         spline_prefiltering.cl
                         spline_utils.cl
                         transpose.cl
