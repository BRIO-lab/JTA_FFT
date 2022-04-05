#!/usr/bin/env python3
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

import numpy as np
import pyopencl as cl
import argparse
import math
from math import floor, log
from utils import *

poles_for_order = {
    0: [],
    1: [],
    2: [-1.715728752538099e-1],
    3: [-2.679491924311227e-1],
    4: [-3.6134122590021989e-1,-1.3725429297339109e-2],
    5: [-4.305753470999738e-1,-4.309628820326465e-2],
    6: [-4.8829458930303893e-1,-8.1679271076238694e-2,-1.4141518083257976e-3],
    7: [-5.352804307964382e-1, -1.225546151923267e-1,-9.148694809608277e-3],
    8: [-5.7468690924876376e-1,-1.6303526929728299e-1,-2.3632294694844336e-2,-1.5382131064168442e-4],
    9: [-6.079973891686259e-1,-2.017505201931532e-1,-4.322260854048175e-2,-2.121306903180818e-3],
    10: [-6.365506639694650e-1,-2.381827983775487e-1,-6.572703322831758e-2,-7.528194675547741e-3,-1.698276282327549e-5],
    11: [-6.612660689007345e-1,-2.721803492947859e-1,-8.975959979371331e-2,-1.666962736623466e-2,-5.105575344465021e-4]
    }

DEVICE_ONLY = 0 # Flag for OpenCL buffers only accessed by the device

def compute_mu(poles):
    """
    Computes the mu_i using Eq. (21).
    """
    mu = np.zeros([len(poles)], dtype=np.float64)
    acc = 0.
    log_pole = log(-poles[0]);
    # mu[0] = 0
    for i in range(1, len(poles)):
        acc += 1./log_pole
        log_pole = log(-poles[i])
        mu[i] = log_pole * acc / (1. + log_pole * acc)
    return mu

def compute_truncation(order, eps):
    """
    Computes the N^(i,eps') using Eq. (22). Note that eps' here
    corresponds to the eps' from Alg. 5 for 2d interpolations.
    """
    poles = poles_for_order[order]
    if len(poles) == 0:
        return [0]
    trunc = np.zeros([len(poles)], dtype=np.int)
    mu = compute_mu(poles)
    rhon = 1.
    for i in range(len(poles)):
        rhon *= (1. + poles[i]) / (1. - poles[i])
    rhon *= rhon

    log_eps_cte = log(eps * rhon * rhon * 0.5)
    prod_mu = 1.

    for i in range(len(poles)-1, -1, -1):
        alpha = poles[i]
        trunc[i] = 1 + \
            int(floor((log_eps_cte + log((1-alpha)*(1-mu[i])*prod_mu))/log(-alpha)))
        prod_mu *= mu[i]

    return trunc

class splinecl:
    def __init__(self, order=3, boundary=1, epsilon=1e-6,
                 ctx=None, queue=None,
                 use_double_prefilter_compute=False,
                 use_double_prefilter_storage=False,
                 use_double_kernel=False,
                 output_double=False, enable_profile=False):
        """
        Initializes OpenCL and check the validity of parameters.
        """
        # Create an OpenCL context
        if ctx is None:
            ctx = cl.create_some_context(interactive=False)
            if not(queue is None):
                queue = None

        if queue is None:
            if enable_profile:
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                events = {}
            else:
                queue = cl.CommandQueue(ctx)
                events = None

        if enable_profile:
            print ('Using device ' + ctx.devices[0].get_info(cl.device_info.NAME))

        order = int(order)
        if order < 0 or order > 11:
            raise ValueError("Invalid Spline order")

        boundary = int(boundary)
        if boundary < 1 or boundary > 3:
            raise ValueError("Invalid boundary condition")

        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError("Invalid epsilon")

        # Here compute trunc for the passes of the order
        trunc_for_order = compute_truncation(order, epsilon)

        # Subdivision of the prefiltering work in subregions.
        # The code assumes min(height-1, width-1, lines_per_item) >= trunc
        lines_per_item = [max(256, trunc) for trunc in trunc_for_order]

        # Compile and define the OpenCL transpose function from transpose.cl
        script_dir = os.path.dirname(os.path.realpath(__file__))
        f = open(os.path.join(script_dir, 'transpose.cl'), 'r')
        fstr = "".join(f.readlines())
        build_options = "-DTILE_DIM=32 -DBLOCK_ROWS=8 -Dcn=1"
        build_options += " -DT=double" if use_double_prefilter_storage else " -DT=float"
        program = cl.Program(ctx, fstr).build(options=build_options)
        transpose = program.transpose
        transpose.set_scalar_arg_dtypes([None, np.int32, np.int32, np.int32, np.int32, None, np.int32, np.int32])

        # Compile and define the OpenCL functions for shifting, zooming and applying an homography from spline_kernel.cl
        f = open(os.path.join(script_dir, 'spline_utils.cl'), 'r')
        fstr = "".join(f.readlines())
        f = open(os.path.join(script_dir, 'spline_kernel.cl'), 'r')
        fstr += "".join(f.readlines())
        build_options = "-DORDER=%d -DBOUNDARY=%d" % (order, boundary)
        if use_double_prefilter_compute:
            build_options += " -DDOUBLE_PRECISION_COMPUTE_PREFILTER"
        if use_double_prefilter_storage:
            build_options += " -DDOUBLE_PRECISION_STORAGE_PREFILTER"
        if use_double_kernel:
            build_options += " -DDOUBLE_PRECISION_KERNEL"
        if output_double:
            build_options += " -DINTERP_OUT_DOUBLE"
        if (order % 2 == 0):
            build_options += " -DRESCALE=%f" % (2.**(2*order)) + ("" if use_double_kernel else "f")
        program = cl.Program(ctx, fstr).build(options=build_options)
        spline_interp_shift = program.spline_interp_shift
        spline_interp_shift.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32, np.int32])
        spline_interp_constant_shift = program.spline_interp_constant_shift
        spline_interp_constant_shift.set_scalar_arg_dtypes([None, None, np.float32, np.float32, np.int32, np.int32, np.int32, np.int32, np.int32])
        spline_interp_zoom = program.spline_interp_zoom
        spline_interp_zoom.set_scalar_arg_dtypes([None, None, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32])
        spline_interp_homography = program.spline_interp_homography
        spline_interp_homography.set_scalar_arg_dtypes([None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32])

        # Precompile all kernels
        prefilters_causal = []
        prefilters_anticausal = []

        num_prefilter_steps = order//2

        # Compile and define the OpenCL functions for prefilterings from spline_prefiltering.cl
        f = open(os.path.join(script_dir, 'spline_utils.cl'), 'r')
        fstr = "".join(f.readlines())
        f = open(os.path.join(script_dir, 'spline_prefiltering.cl'), 'r')
        fstr += "".join(f.readlines())

        for step in range(num_prefilter_steps):
            build_options = "-DORDER=%d -DTRUNC=%d" % (order, trunc_for_order[step])
            build_options += " -DBOUNDARY=%d -DLINES_PER_ITEM=%d" % (boundary, lines_per_item[step])
            build_options += " -DRUN=%d" % step
            if use_double_prefilter_compute:
                build_options += " -DDOUBLE_PRECISION_COMPUTE_PREFILTER"
            if use_double_prefilter_storage:
                build_options += " -DDOUBLE_PRECISION_STORAGE_PREFILTER"
            if use_double_kernel:
                build_options += " -DDOUBLE_PRECISION_KERNEL"
            if output_double:
                build_options += " -DINTERP_OUT_DOUBLE"
            program = cl.Program(ctx, fstr).build(options=build_options)
            prefilter_column_for_pole_causal = program.prefilter_column_for_pole_causal
            prefilter_column_for_pole_causal.set_scalar_arg_dtypes([None, None, np.int32, np.int32, np.int32, np.int32, np.int32])
            prefilter_column_for_pole_anticausal = program.prefilter_column_for_pole_anticausal
            prefilter_column_for_pole_anticausal.set_scalar_arg_dtypes([None, None, np.int32, np.int32, np.int32, np.int32, np.int32])
            prefilters_causal.append(prefilter_column_for_pole_causal)
            prefilters_anticausal.append(prefilter_column_for_pole_anticausal)

        self.ctx = ctx
        self.queue = queue
        self.events = events
        self.src_shape = [0, 0, 0]
        self.dst_shape = [0, 0, 0]
        self.min_size = max(trunc_for_order)
        self.size_data = 8 if use_double_prefilter_storage else 4
        self.size_output = 8 if output_double else 4
        self.transpose = transpose
        self.spline_interp_shift = spline_interp_shift
        self.spline_interp_constant_shift = spline_interp_constant_shift
        self.spline_interp_zoom = spline_interp_zoom
        self.spline_interp_homography = spline_interp_homography
        self.prefilters_causal = prefilters_causal
        self.prefilters_anticausal = prefilters_anticausal
        self.lines_per_item = lines_per_item
        self.num_prefilter_steps = num_prefilter_steps
        self.input_dtype = np.float64 if use_double_prefilter_storage else np.float32
        self.output_dtype = np.float64 if output_double else np.float32

    def __allocate_buffers(self, src_shape, dst_shape):
        """
        Allocates, if necessary, all buffers required for the OpenCL code.
        It also checks that input and output sizes are consistent.
        """
        if len(self.src_shape) == len(src_shape) and \
           len(self.dst_shape) == len(dst_shape) and \
           np.all(self.src_shape == src_shape) and np.all(self.dst_shape == dst_shape):
               return # Already allocated

        if len(src_shape) == 2:
            channels = 1
            [src_height, src_width] = src_shape
        else:
            if len(src_shape) != 3:
                raise ValueError("Invalid input shape")
            [channels, src_height, src_width] = src_shape
        if len(dst_shape) == 2:
            if channels != 1:
                raise ValueError("Invalid output shape")
            [dst_height, dst_width] = dst_shape
        else:
            if channels != dst_shape[0]:
                raise ValueError("Invalid output shape")
            [_, dst_height, dst_width] = dst_shape

        # The code assumes min(height-1, width-1, lines_per_item) >= trunc
        if self.min_size >= min(src_height, src_width) - 1:
            # If one must handle this case, clamp trunc_for_order to that min.
            # or increase epsilon for very small images
            raise ValueError("Image too small, computing would be innacurate")

        # Create OpenCL buffers
        # Ensures C order (and contiguous array, but the OpenCL code can support
        # different stride).
        # Deinterlace channels.
        num_data_src = np.prod(np.asarray(src_shape))
        num_data_dst = np.prod(np.asarray(dst_shape))
        mf = cl.mem_flags

        # Buffer containing the input image
        self.img_cl = cl.Buffer(self.ctx, mf.READ_ONLY, num_data_src * self.size_data)

        # tmp_prefilter_cl: float buffer of the size of the image
        # where the prefilter result is stored
        # tmp_prefilter_2_cl: second buffer (because transforms are not in place)
        self.tmp_prefilter_cl = cl.Buffer(self.ctx, DEVICE_ONLY, num_data_src * self.size_data)
        self.tmp_prefilter_2_cl = cl.Buffer(self.ctx, DEVICE_ONLY, num_data_src * self.size_data)
        # tmp_transpose_cl: float buffer of the size of the image
        # used for the transposition
        self.tmp_transpose_cl = cl.Buffer(self.ctx, DEVICE_ONLY, num_data_src * self.size_data)

        # output buffer
        self.out_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, num_data_dst * self.size_output)

        self.channels = channels
        self.src_width = src_width
        self.src_height = src_height
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.row_bytes = src_width * self.size_data
        self.col_bytes = src_height * self.size_data

        self.src_shape = src_shape
        self.dst_shape = dst_shape

    def __upload_image(self, img):
        """
        Copy the input image into the input buffer.
        It also adds the event to the profile.
        """
        assert(img.shape == self.src_shape)
        profile(self.events, cl.enqueue_copy(self.queue, self.img_cl, img, device_offset=0), "enqueue_copy", False)

    def __retrieve_result(self):
        """
        Extract the output image from the output buffer.
        It also adds the event to the profile.
        """
        out = np.empty(self.dst_shape, dtype=self.output_dtype, order='C')
        profile(self.events, cl.enqueue_copy(self.queue, out, self.out_cl), "enqueue_copy", False)
        return out

    def __prefilter_pass(self):
        """
        Alg. 2 for both dimensions.
        Note that the truncation indexes N^(i,eps) have been precomputed during the initialization.
        """
        src_width = self.src_width
        src_height = self.src_height
        channels = self.channels
        global_size_prefilters = [[DIVUP(src_width, 64) * 64, DIVUP(src_height, lines_per_item_step)] for lines_per_item_step in self.lines_per_item]
        # For best memory access pattern, use a line work group
        local_size_prefilters = [64, 1]

        # Alg. 2 for the first dimension
        for step in range(self.num_prefilter_steps):
            if step == 0:
                input_buffer_cl = self.img_cl
                middle_buffer_cl = self.tmp_prefilter_cl
                output_buffer_cl = self.tmp_prefilter_2_cl
            else:
                input_buffer_cl = self.tmp_prefilter_2_cl
                middle_buffer_cl = self.tmp_prefilter_cl
                output_buffer_cl = self.tmp_prefilter_2_cl
            for c in range(channels):
                profile(self.events, self.prefilters_causal[step](self.queue,
                                    global_size_prefilters[step],
                                    local_size_prefilters,
                                    middle_buffer_cl,
                                    input_buffer_cl,
                                    c * src_width * src_height,
                                    c * src_width * src_height,
                                    src_width,
                                    src_height,
                                    src_width),
                    "prefilter_causal", True)
                profile(self.events, self.prefilters_anticausal[step](self.queue,
                                    global_size_prefilters[step],
                                    local_size_prefilters,
                                    output_buffer_cl,
                                    middle_buffer_cl,
                                    c * src_width * src_height,
                                    c * src_width * src_height,
                                    src_width,
                                    src_height,
                                    src_width),
                    "prefilter_anticausal", True)

        # Transpose the result so to apply the prefiltering on the second dimension
        if self.num_prefilter_steps >= 1:
            global_size_tranpose = [DIVUP(src_width, 32)* 32, DIVUP(src_height, 32) * 8]
            local_size_transpose = [32, 8]
            for c in range(channels):
                profile(self.events, self.transpose(self.queue,
                      global_size_tranpose,
                      local_size_transpose,
                      self.tmp_prefilter_2_cl,
                      self.row_bytes,
                      c * self.row_bytes * src_height,
                      src_height,
                      src_width,
                      self.tmp_transpose_cl,
                      self.col_bytes,
                      c * src_width * self.col_bytes),
                    "transpose", True)

        global_size_prefilters = [[DIVUP(src_height, 64) * 64, DIVUP(src_width, lines_per_item_step)] for lines_per_item_step in self.lines_per_item]

        # Alg. 2 for the second dimension
        for step in range(self.num_prefilter_steps):
            if step == 0:
                input_buffer_cl = self.tmp_transpose_cl
                middle_buffer_cl = self.tmp_prefilter_cl
                output_buffer_cl = self.tmp_prefilter_2_cl
            else:
                input_buffer_cl = self.tmp_prefilter_2_cl
                middle_buffer_cl = self.tmp_prefilter_cl
                output_buffer_cl = self.tmp_prefilter_2_cl
            for c in range(channels):
                profile(self.events, self.prefilters_causal[step](self.queue,
                                    global_size_prefilters[step],
                                    local_size_prefilters,
                                    middle_buffer_cl,
                                    input_buffer_cl,
                                    c * src_width * src_height,
                                    c * src_width * src_height,
                                    src_height,
                                    src_width,
                                    src_height),
                    "prefilter_causal", True)
                profile(self.events, self.prefilters_anticausal[step](self.queue,
                                    global_size_prefilters[step],
                                    local_size_prefilters,
                                    output_buffer_cl,
                                    middle_buffer_cl,
                                    c * src_width * src_height,
                                    c * src_width * src_height,
                                    src_height,
                                    src_width,
                                    src_height),
                    "prefilter_anticausal", True)

        # Transpose the result back to the original layout
        if self.num_prefilter_steps >= 1:
            global_size_tranpose = [DIVUP(src_height, 32)* 32, DIVUP(src_width, 32) * 8]

            for c in range(channels):
                profile(self.events, self.transpose(self.queue,
                      global_size_tranpose,
                      local_size_transpose,
                      self.tmp_prefilter_2_cl,
                      self.col_bytes,
                      c * src_width * self.col_bytes,
                      src_width,
                      src_height,
                      self.tmp_transpose_cl,
                      self.row_bytes,
                      c * self.row_bytes * src_height),
                    "transpose", True)

    def shift(self, img, dx, dy):
        """
        Shift the input image by (dx,dy)
        """
        self.__allocate_buffers(img.shape, img.shape)
        self.__upload_image(np.ascontiguousarray(img, dtype=self.input_dtype))
        self.__prefilter_pass()

        dst_width = src_width = self.src_width
        dst_height = src_height = self.src_height
        channels = self.channels

        global_size_interp = [DIVUP(dst_width, 64) * 64, dst_height]
        local_size_interp = [64, 1]

        input_kernel = self.img_cl if self.num_prefilter_steps == 0 else self.tmp_transpose_cl

        for c in range(channels):
            profile(self.events, self.spline_interp_constant_shift(self.queue,
                      global_size_interp,
                      local_size_interp,
                      self.out_cl,
                      input_kernel,
                      dx,
                      dy,
                      c * dst_height * dst_width,
                      c * dst_height * dst_width,
                      dst_width, dst_height, dst_width),
               "spline_interp", True)

        out = self.__retrieve_result()
        print_profile_info(self.events)

        return out

    def zoom(self, img, dst_shape):
        """
        Zoom the input image so to match the wanted output size
        """
        self.__allocate_buffers(img.shape, dst_shape)
        self.__upload_image(np.ascontiguousarray(img, dtype=self.input_dtype))
        self.__prefilter_pass()

        src_width = self.src_width
        src_height = self.src_height
        dst_width = self.dst_width
        dst_height = self.dst_height
        channels = self.channels

        global_size_interp = [DIVUP(dst_width, 64) * 64, dst_height]
        local_size_interp = [64, 1]

        input_kernel = self.img_cl if self.num_prefilter_steps == 0 else self.tmp_transpose_cl

        for c in range(channels):
            profile(self.events, self.spline_interp_zoom(self.queue,
                      global_size_interp,
                      local_size_interp,
                      self.out_cl,
                      input_kernel,
                      c * dst_height * dst_width,
                      c * src_height * src_width,
                      dst_width, dst_height, dst_width,
                      src_width, src_height, src_width),
               "spline_interp", True)

        out = self.__retrieve_result()
        print_profile_info(self.events)

        return out

    def homography(self, img, dst_shape, p):
        """
        Apply the homography p to the input image and resize so to fit the wanted output size
        """
        self.__allocate_buffers(img.shape, dst_shape)
        self.__upload_image(np.ascontiguousarray(img, dtype=self.input_dtype))
        self.__prefilter_pass()

        [channels, src_height, src_width] = self.src_shape
        [channels, dst_height, dst_width] = self.dst_shape

        global_size_interp = [dst_width, dst_height]

        input_kernel = self.img_cl if self.num_prefilter_steps == 0 else self.tmp_transpose_cl

        for c in range(channels):
            profile(self.events, self.spline_interp_homography(self.queue,
                      global_size_interp,
                      None,
                      self.out_cl,
                      input_kernel,
                      p[0], p[1], p[2], p[3],
                      p[4], p[5], p[6], p[7],
                      c * dst_height * dst_width,
                      c * src_height * src_width,
                      dst_width, dst_height, dst_width,
                      src_width, src_height, src_width),
               "spline_interp", True)

        out = self.__retrieve_result()
        print_profile_info(self.events)

        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', help='shift, zoom or homography',
                                       metavar='{shift, zoom, homography}')

    parser_t = subparsers.add_parser('shift', help='Shift an image')
    parser_t.add_argument('infile', help=('Input file'))
    parser_t.add_argument('outfile', help=('Output file'))
    parser_t.add_argument('dx', default=0., help=('Shift on the x axis'))
    parser_t.add_argument('dy', default=0., help=('Shift on the y axis'))
    parser_t.add_argument('order', default=3, help=('Interpolation order'))
    parser_t.add_argument('boundary', default=1, help=('Boundary condition (1:hsym, 2:wsym, 3:per)'))
    parser_t.add_argument('epsilon', default=1e-6, help=('Epsilon (precision control)'))
    parser_t.add_argument('--use-double-prefilter-compute', action='store_true',
                          help=('Use double precision inside the prefilter for computation'))
    parser_t.add_argument('--use-double-prefilter-storage', action='store_true',
                          help=('Use double precision for prefilter result storage'))
    parser_t.add_argument('--use-double-kernel', action='store_true',
                          help=('Use double precision for kernel computation'))
    parser_t.add_argument('--output-double', action='store_true',
                          help=('Result is stored in double instead of float'))
    parser_t.add_argument('--profile', action='store_true',
                          help=('Print profile info'))

    parser_z = subparsers.add_parser('zoom', help='Zoom an image')
    parser_z.add_argument('infile', help=('Input file'))
    parser_z.add_argument('outfile', help=('Output file'))
    parser_z.add_argument('zx', default=1., help=('Zoom on the x axis'))
    parser_z.add_argument('zy', default=1., help=('Zoom on the y axis'))
    parser_z.add_argument('order', default=3, help=('Interpolation order'))
    parser_z.add_argument('boundary', default=1, help=('Boundary condition (1:hsym, 2:wsym, 3:per)'))
    parser_z.add_argument('epsilon', default=1e-6, help=('Epsilon (precision control)'))
    parser_z.add_argument('--use-double-prefilter-compute', action='store_true',
                          help=('Use double precision inside the prefilter for computation'))
    parser_z.add_argument('--use-double-prefilter-storage', action='store_true',
                          help=('Use double precision for prefilter result storage'))
    parser_z.add_argument('--use-double-kernel', action='store_true',
                          help=('Use double precision for kernel computation'))
    parser_z.add_argument('--output-double', action='store_true',
                          help=('Result is stored in double instead of float'))
    parser_z.add_argument('--profile', action='store_true',
                          help=('Print profile info'))

    parser_h = subparsers.add_parser('homography', help='Apply homography')
    parser_h.add_argument('infile', help=('Input file'))
    parser_h.add_argument('outfile', help=('Output file'))
    parser_h.add_argument('order', default=3, help=('Interpolation order'))
    parser_h.add_argument('boundary', default=1, help=('Boundary condition (1:hsym, 2:wsym, 3:per)'))
    parser_h.add_argument('epsilon', default=1e-6, help=('Epsilon (precision control)'))
    parser_h.add_argument('--homography', default=[1., 0., 0., 0., 1., 0., 0., 0.], type=float, nargs=8, help=('Homography parameters'))
    parser_h.add_argument('--zx', default=1., help=('Zoom on the x axis'))
    parser_h.add_argument('--zy', default=1., help=('Zoom on the y axis'))
    parser_h.add_argument('--use-double-prefilter-compute', action='store_true',
                          help=('Use double precision inside the prefilter for computation'))
    parser_h.add_argument('--use-double-prefilter-storage', action='store_true',
                          help=('Use double precision for prefilter result storage'))
    parser_h.add_argument('--use-double-kernel', action='store_true',
                          help=('Use double precision for kernel computation'))
    parser_h.add_argument('--output-double', action='store_true',
                          help=('Result is stored in double instead of float'))
    parser_h.add_argument('--profile', action='store_true',
                          help=('Print profile info'))

    args = parser.parse_args()

    if args.cmd not in ["shift", "zoom", "homography"]:
        print('Please use -h or --help to get information on available commands')
        exit()

    op = splinecl(order=args.order, boundary=args.boundary, epsilon=args.epsilon, use_double_prefilter_compute=args.use_double_prefilter_compute, use_double_prefilter_storage=args.use_double_prefilter_storage, use_double_kernel=args.use_double_kernel, output_double=args.output_double, enable_profile=args.profile)

    img = load_file(args.infile)
    if args.cmd == "shift":
        dx = float(args.dx)
        dy = float(args.dy)
        out = op.shift(img, dx, dy)
    elif args.cmd == "zoom":
        zx = float(args.zx)
        zy = float(args.zy)
        [channels, src_height, src_width] = img.shape
        dst_height = int(zy * src_height)
        dst_width = int(zx * src_width)
        out = op.zoom(img, [channels, dst_height, dst_width])
    elif args.cmd == "homography":
        zx = float(args.zx)
        zy = float(args.zy)
        [channels, src_height, src_width] = img.shape
        dst_height = int(zy * src_height)
        dst_width = int(zx * src_width)
        p = args.homography
        if len(p) != 8:
            print('Invalid homography. 8 parameters are expected (the homography matrix in row-major order, minus last elements assumed to be 1)')
        out = op.homography(img, [channels, dst_height, dst_width], p)

    save_file(args.outfile, out)
