"""
Copyright 2019 Axel Davy
Copyright 2019 CMLA ENS PARIS SACLAY
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import absolute_import, print_function
import errno
import fnmatch
import math
import numpy as np
import os
import os.path
import pyopencl as cl

tifffile_loaded = False
imageio_loaded = False
cv2_loaded = False
piio_loaded = False

try:
    import piio
    piio_loaded = True
except:
    pass
try:
    import tifffile
    tifffile_loaded = True
except:
    pass
try:
    import imageio
    imageio_loaded = True
except:
    pass
try:
    import cv
    cv2_loaded = True
except:
    pass

#tifffile_loaded = False
#imageio_loaded = False
#cv2_loaded = False
#piio_loaded = False

if not(tifffile_loaded or imageio_loaded or cv2_loaded or piio_loaded):
    print('No supported module to read images found')
    print('Supported are: tifffile, imageio, cv2 and piio')

# Use highest available (leads to better opts)
def std_string(ctx):
    """
    Select highest OPENCL C version.
    """
    device = ctx.devices[0]
    version = device.get_info(cl.device_info.OPENCL_C_VERSION)
    if ("1.2" in version):
        return ' -cl-std=CL1.2'
    elif ("2.0" in version):
        return ' -cl-std=CL2.0'
    elif ("2.1" in version):
        return ' -cl-std=CL2.1'
    else:
        return ' -cl-std=CL2.2'

def profile(events, f_res, name, is_kernel):
    """
    Add an event to the profile. Events are organized by types.
    """
    if events is None:
        return
    evt = f_res
    if name in events:
        events[name].append(evt)
    else:
        events[name] = [evt]
    if 'evt_all' in events:
        events['evt_all'].append(evt)
    else:
        events['evt_all'] = [evt]
    if is_kernel:
        if 'evt_all_kernel' in events:
            events['evt_all_kernel'].append(evt)
        else:
            events['evt_all_kernel'] = [evt]
    else:
        if 'evt_all_other' in events:
            events['evt_all_other'].append(evt)
        else:
            events['evt_all_other'] = [evt]

def print_profile_info(events):
    """
    Print a profile history of the code (with computation time).
    """
    if events is None:
        return
    print ('Profiling Information:')
    tot_time = ((events['evt_all'][-1].profile.end - events['evt_all'][0].profile.start) * 1e-6)
    print ('Time between first and last OpenCL event: %f ms' % tot_time)
    sum_time = sum([(evt.profile.end - evt.profile.start) for evt in events['evt_all']]) * 1e-6
    print ('Time spent in OpenCL events: %f ms' % sum_time)
    print ('Thus a CPU overhead of: %f ms\n' % (tot_time-sum_time))

    #tot_time_k = ((events['evt_all_kernel'][-1].profile.end - events['evt_all_kernel'][0].profile.start) * 1e-6)
    #print ('Time between first and last kernel: %f ms' % tot_time_k)
    sum_time_k = sum([(evt.profile.end - evt.profile.start) for evt in events['evt_all_kernel']]) * 1e-6
    print ('Time spent executing kernels: %f ms' % sum_time_k)
    print ('Time spent in transfers and synchronizations: %f ms\n' % (sum_time-sum_time_k))

    print ('Total cumulated time spent per kernel/OpenCL operation:')
    for key in sorted(events.keys()):
        if key[:4] == "evt_":
            continue
        print ('%s: %f ms' % (key, sum([(evt.profile.end - evt.profile.start) for evt in events[key]]) * 1e-6))
    print ('Note: enqueue_copy corresponds to transfers CPU<->OpenCL (CPU or GPU)')

def DIVUP(a, b):
    """
    Integer division with rounding to the upper value.
    """
    return int(math.ceil( float(a) / float(b)))

def load_file(f):
    """
    Loads file f.
    If grayscale, the shape with be of length 3 (channel first).
    If color, the channels are first.
    """
    global tifffile_loaded, imageio_loaded, cv2_loaded, piio_loaded
    # tifffile keeps the type
    # of the data untouched
    if (tifffile_loaded and (f[-4:] == 'tiff' or f[-3:] == 'tif')):
        try:
            img = tifffile.imread(f)
            if len(img.shape) == 3:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis, :, :]
            return img
        except:
            pass
    # Reads a lot of formats
    if (imageio_loaded):
        try:
            img = imageio.imread(f)
            if len(img.shape) == 3:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis, :, :]
            return img
        except:
            pass
    # No way to know if grayscale or not
    if (cv2_loaded):
        try:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            return img
        except:
            pass
    # Reads a lot of formats,
    # but may change the data type
    # to float32
    if (piio_loaded):
        try:
            img = piio.read(f)
            img = np.transpose(img, (2, 0, 1))
            return img
        except:
            pass
    if not (os.path.exists(f)):
        print ("File %s doesn't exist" % f)
        assert(False)
    print ("No library found to read %s" % f)
    assert(False)

def save_file(f, data):
    """
    Saves file f.
    If grayscale, the shape will be of length 3 (channel first).
    If color, the channels are first.
    """
    global tifffile_loaded, imageio_loaded, cv2_loaded, piio_loaded
    assert(len(data.shape) == 3)
    # tifffile keeps the type
    # of the data untouched
    if ((f[-4:] == 'jpeg' or f[-3:] == 'png' or f[-3:] == 'jpg') and data.dtype == np.float32):
        #if np.max(data) <= 1.9:
        #    data*=255
        data = np.floor(data + 0.5)
        data[data<0] = 0
        data[data>255] = 255
        data = np.asarray(data, dtype=np.uint8)

    if (tifffile_loaded and (f[-4:] == 'tiff' or f[-3:] == 'tif')):
        try:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[0, :, :]
            tifffile.imsave(f, data)
            return
        except:
            pass
    # Reads a lot of formats
    if (imageio_loaded):
        try:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data[0, :, :]
            imageio.imwrite(f, data)
            return
        except:
            pass
    if (cv2_loaded and data.shape[0] == 3):
        try:
            data = np.transpose(data, (1, 2, 0))
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f, data)
            return
        except:
            pass
    # Reads a lot of formats,
    # but may change the data type
    if (piio_loaded):
        try:
            data = np.transpose(data, (1, 2, 0))
            data = np.ascontiguousarray(data) #old piio bug
            piio.write(f, data)
            return
        except:
            pass
    print ("Couldn't write file %s" % f)
    assert(False)


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    if path:
        try:
            os.makedirs(path)
        except OSError as exc: # requires Python > 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: assert(False)

