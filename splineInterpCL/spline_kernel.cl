/*
 * Copyright 2018 Axel Davy
 * Copyright 2018 CMLA ENS PARIS SACLAY
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if ORDER == 0
#define KERNEL_WIDTH 2
#else
#define KERNEL_WIDTH (ORDER+1)
#endif

/* % operator is not the modulo, but the remainder operator.
 * It gives a negative if the input is negative. */
#define MODULO(a, b) {\
    a = a % (b); \
    if (a < 0) \
        a += (b); \
    }

/* Computes the value of the image src at position (xcenter, ycenter). Write the result
 * in dst at position dst_offset. */
inline void spline_interp(__global uchar * restrict dst,
                          __global const STORAGE_PREFILTER_TYPE * restrict src,
                          int dst_offset,
                          float xcenter,
                          float ycenter,
                          int src_offset,
                          int src_width, int src_height, int src_items_row)
{

    KERNEL_TYPE KX[KERNEL_WIDTH];
    KERNEL_TYPE KY[KERNEL_WIDTH];
    int i,j;

    int xmin = ceil(xcenter-RADIUS);
    int ymin = ceil(ycenter-RADIUS);

    KERNEL_TYPE result = (KERNEL_TYPE)0.;

    /* Tabulate values of the spline both directions */
    #pragma unroll
    for (i = 0; i < KERNEL_WIDTH; i++) {
        KX[i] = BSpline(convert_KT(xcenter-(xmin+i)));
        KY[i] = BSpline(convert_KT(ycenter-(ymin+i)));
    }

    /* If boundary checks are not required, we can use a faster path */
    if (xmin < 0 || xmin + KERNEL_WIDTH > src_width ||
        ymin < 0 || ymin + KERNEL_WIDTH > src_height) {
        #pragma unroll
        for (i = 0; i < KERNEL_WIDTH; i++) {
            KERNEL_TYPE result_col = (KERNEL_TYPE)0.;
            int posy = ymin+i;

            /* If the interpolation kernel position falls outside of the image,
             * compute the equivalent position inside the image using the
             * boundary condition. */
            if (posy < 0 || posy >= src_height) {
#if BOUNDARY == 1 /* Half-symmetric */
                /* The infinitely extended image is (2 * src_height) periodic */
                MODULO(posy, 2 * src_height);
                /* We are now in [0, 2 * src_height - 1] */
                if (posy >= src_height)
                    /* Mirror */
                    posy = 2 * src_height - 1 - posy;
#elif BOUNDARY == 2 /* Whole-symmetric */
                MODULO(posy, 2 * src_height - 2);
                /* We are now in [0, (2 * src_height - 2) - 1] */
                if (posy >= src_height)
                    /* Mirror */
                    posy = 2 * src_height - 2 - posy;
#elif BOUNDARY == 3 /* Periodic */
                MODULO(posy, src_height);
#endif
            }
            #pragma unroll
            for (j = 0; j < KERNEL_WIDTH; j++) {
                int posx = xmin+j;

                /* If the interpolation kernel position falls outside of the image,
                 * compute the equivalent position inside the image using the boundary
                 * condition. */
                if (posx < 0 || posx >= src_width) {
#if BOUNDARY == 1 /* Half-symmetric */
                    /* The infinitely extended image is (2 * src_width) periodic */
                    MODULO(posx, 2 * src_width);
                    /* We are now in [0, 2 * src_width - 1] */
                    if (posx >= src_width)
                        /* Mirror */
                        posx = 2 * src_width - 1 - posx;
#elif BOUNDARY == 2 /* Whole-symmetric */
                    /* The infinitely extended image is (2 * src_width - 2) periodic */
                    MODULO(posx, 2 * src_width - 2);
                    /* We are now in [0, (2 * src_width - 2) - 1] */
                    if (posx >= src_width)
                        /* Mirror */
                        posx = 2 * src_width - 2 - posx;
#elif BOUNDARY == 3 /* Periodic */
                    MODULO(posx, src_width);
#endif
                }
                result_col += convert_KT(src[src_offset + posx + posy*src_items_row]) * KX[j];
            }
            result += result_col * KY[i];
        }
    } else { /* Fast path - no border handling */
        #pragma unroll
        for (i = 0; i < KERNEL_WIDTH; i++) {
            KERNEL_TYPE result_col = (KERNEL_TYPE)0.;
            int posy = ymin+i;
            #pragma unroll
            for (j = 0; j < KERNEL_WIDTH; j++) {
                int posx = xmin+j;
                result_col += convert_KT(src[src_offset + posx + posy*src_items_row]) * KX[j];
            }
            result += result_col * KY[i];
        }
    }

#ifdef RESCALE
    result *= RESCALE;
#endif

#ifdef INTERP_OUT_DOUBLE
    ((__global double * restrict)dst)[dst_offset] = convert_double(result);
#else
    ((__global float * restrict)dst)[dst_offset] = convert_float(result);
#endif
}

/* Kernel used to compute a shifted version of an image
 * according to a non uniform shift (for example an optical flow). */
__kernel void spline_interp_shift(__global uchar * restrict dst,
                                  __global const STORAGE_PREFILTER_TYPE * restrict src,
                                  __global const float * restrict shiftx,
                                  __global const float * restrict shifty,
                                  int dst_offset, int src_offset,
                                  int width, int height, int items_row)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int pos = x+y*items_row;

    if (x >= width || y >= height)
        return;

    float xshift = shiftx[pos];
    float yshift = shifty[pos];

    float xcenter = (float)x-xshift;
    float ycenter = (float)y-yshift;

    spline_interp(dst, src, dst_offset+pos, xcenter, ycenter,
                  src_offset, width, height, items_row);
}

/* Kernel used to compute a shifted version of an image with a constant shift. */
__kernel void spline_interp_constant_shift(__global uchar * restrict dst,
                                           __global const STORAGE_PREFILTER_TYPE * restrict src,
                                           float dx,
                                           float dy,
                                           int dst_offset, int src_offset,
                                           int width, int height, int items_row)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int pos = x+y*items_row;

    if (x >= width || y >= height)
        return;

    float xcenter = (float)x-dx;
    float ycenter = (float)y-dy;

    spline_interp(dst, src, dst_offset+pos, xcenter, ycenter,
                  src_offset, width, height, items_row);
}

/* Kernel used to compute a zoomed version on image to match the output size. */
__kernel void spline_interp_zoom(__global uchar * restrict dst,
                                 __global const STORAGE_PREFILTER_TYPE * restrict src,
                                 int dst_offset, int src_offset,
                                 int dst_width, int dst_height, int dst_items_row,
                                 int src_width, int src_height, int src_items_row)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int pos = x+y*dst_items_row;

    if (x >= dst_width || y >= dst_height)
        return;

    float scale_x = (float)dst_width/(float)src_width;
    float scale_y = (float)dst_height/(float)src_height;

    float xcenter = ((float)x-0.5f*(scale_x-1.f))/scale_x;
    float ycenter = ((float)y-0.5f*(scale_y-1.f))/scale_y;

    spline_interp(dst, src, dst_offset+pos, xcenter, ycenter,
                  src_offset, src_width, src_height, src_items_row);
}

/* Kernel used to compute a warped version on image using an homography.
 * It is done so as to match the output size. */
__kernel void spline_interp_homography(__global uchar * restrict dst,
                                       __global const STORAGE_PREFILTER_TYPE * restrict src,
                                       float p0, float p1, float p2,
                                       float p3, float p4, float p5,
                                       float p6, float p7,
                                       int dst_offset, int src_offset,
                                       int dst_width, int dst_height, int dst_items_row,
                                       int src_width, int src_height, int src_items_row)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int pos = x+y*dst_items_row;

    if (x >= dst_width || y >= dst_height)
        return;

    float scale_x = (float)dst_width/(float)src_width;
    float scale_y = (float)dst_height/(float)src_height;

    float xcenter = ((float)x-0.5f*(scale_x-1.f))/scale_x;
    float ycenter = ((float)y-0.5f*(scale_y-1.f))/scale_y;

    float d = p6*xcenter + p7*ycenter + 1.f;
    float xcenter_ = (p0*xcenter + p1*ycenter + p2)/d;
    ycenter = (p3*xcenter + p4*ycenter + p5)/d;
    xcenter = xcenter_;

    spline_interp(dst, src, dst_offset+pos, xcenter, ycenter,
                  src_offset, src_width, src_height, src_items_row);
}
