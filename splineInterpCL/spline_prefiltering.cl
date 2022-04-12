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


/* Compute the causal pass for a given pole. */
__kernel void prefilter_column_for_pole_causal(__global uchar * restrict dst, __global const uchar * restrict src, int dst_offset, int src_offset, int width, int height, int items_row)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    COMPUTE_PREFILTER_TYPE alpha = Poles[RUN];

    if (x >= width)
        return;

    int position = x + y*LINES_PER_ITEM*items_row; /* LINES_PER_ITEM >= TRUNC */

    COMPUTE_PREFILTER_TYPE powAlpha = 1.f;
    COMPUTE_PREFILTER_TYPE last, sum;
    int i;
    int src_position = (src_offset + position) * sizeof(STORAGE_PREFILTER_TYPE);
    int dst_position = (dst_offset + position) * sizeof(STORAGE_PREFILTER_TYPE);
    int stride = sizeof(STORAGE_PREFILTER_TYPE)*items_row;

    sum = read_src_p(src + src_position);

    /* Since LINES_PER_ITEM > TRUNC, boundary effects are only for y=0 */
    if (y == 0) {
        /* Since the image is finite, use symmetry
         * or periodicity to initialize the filter to
         * the required precision.
         * See Eq. (19) where TRUNC correspond to N to compute Eq. (14) */
        #pragma unroll
        for(i=0; i<TRUNC; i++) {
            int read_offset;
            powAlpha *= alpha;
#if BOUNDARY == 1 /* Half-symmetric */
            read_offset = i*stride;
#elif BOUNDARY == 2 /* Whole-symmetric */
            read_offset = i*stride + stride;
#else /* BOUNDARY == 3 Periodic */
            read_offset = (height-1)*stride-i*stride;
#endif
            sum += read_src_p(src + src_position + read_offset) * powAlpha;
        }
    } else {
        /* Initialize the filter to the required precision.
         * See Eq. (19) where TRUNC correspond to N to compute Eq. (14) */
        #pragma unroll
        for(i=0; i<TRUNC; i++) {
            int read_offset;
            powAlpha *= alpha;
            read_offset = -i*stride - stride;
            sum += read_src_p(src + src_position + read_offset) * powAlpha;
        }
    }

    write_dst_p(dst + dst_position, sum);
    last = sum;

    /* No border conditions. Work items are on a line,
     * thus all items are on the same branch. */
    if (y*LINES_PER_ITEM + LINES_PER_ITEM <= height) {
        #pragma unroll 20
        for(i=1; i<LINES_PER_ITEM; i++) {
            src_position += stride;
            dst_position += stride;
            /* See Eq. (15) */
            last = read_src_p(src + src_position) + alpha * last;
            write_dst_p(dst + dst_position, last);
        }
    } else {
        i=1;
        /* Process is done by unrolling blocks of 20 lines.
         * Process the blocks that fit in the image before
         * taking extra care for the last one. */
        while (y*LINES_PER_ITEM + (i+20) <= height) {
            #pragma unroll
            for(int count=0; count<20; count++,i++) {
                src_position += stride;
                dst_position += stride;
                /* See Eq. (15) */
                last = read_src_p(src + src_position) + alpha * last;
                write_dst_p(dst + dst_position, last);
            }
        }
        for(; i<LINES_PER_ITEM; i++) {
            src_position += stride;
            dst_position += stride;
            /* Check if we're not outside of the image already */
            if (y*LINES_PER_ITEM+i >= height)
                return;
            /* See Eq. (15) */
            last = read_src_p(src + src_position) + alpha * last;
            write_dst_p(dst + dst_position, last);
        }
    }
}

/* Compute the anticausal pass for a given pole */
__kernel void prefilter_column_for_pole_anticausal(__global uchar * restrict dst, __global const uchar * restrict src, int dst_offset, int src_offset, int width, int height, int items_row)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    COMPUTE_PREFILTER_TYPE alpha = Poles[RUN];

    if (x >= width)
        return;

    /* Anticausal starts at the bottom of the image */
    int top_area_y = (height-(y+1)*LINES_PER_ITEM);
    int top_area_position = x + (height-(y+1)*LINES_PER_ITEM)*items_row;

    int src_position = (src_offset + top_area_position + (LINES_PER_ITEM-1)*items_row) * sizeof(STORAGE_PREFILTER_TYPE);
    int dst_position = (dst_offset + top_area_position + (LINES_PER_ITEM-1)*items_row) * sizeof(STORAGE_PREFILTER_TYPE);
    int stride = sizeof(STORAGE_PREFILTER_TYPE)*items_row;

    COMPUTE_PREFILTER_TYPE powAlpha = 1.f;
    COMPUTE_PREFILTER_TYPE last, sum;
    int i;

    /* Since LINES_PER_ITEM > TRUNC, boundary effects are only for y=0 */
    if (y == 0) {
        /* Since the image is finite, use symmetry
         * or periodicity to initialize the filter to
         * the required precision.
         * See Eq. (20) where TRUNC correspond to N to compute Eq. (16)
         * See Table 2 for more detail on the extensions. */
        last = read_src(src, src_offset + x + (height-2)*items_row);
#if BOUNDARY == 1 /* Half-symmetric */
        last = read_src(src, src_offset + x + (height-1)*items_row) * alpha/(alpha - 1);
#elif BOUNDARY == 2 /* Whole-symmetric */
        last = (read_src(src, src_offset + x + (height-1)*items_row) + alpha*last) * (alpha/(alpha*alpha - 1));
#else /* BOUNDARY == 3 Periodic */
        sum = read_src(src, src_offset + x + (height-1)*items_row);
        #pragma unroll
        for(i=0; i<TRUNC; i++) {
            powAlpha *= alpha;
            sum += read_src(src, src_offset + x + i*items_row)*powAlpha;
        }
        last = -alpha * sum;
#endif
    } else {
        /* Initialize the filter to the required precision.
         * See Eq. (20) where TRUNC correspond to N to compute Eq. (16) */
        sum = read_src(src, src_offset + top_area_position + (LINES_PER_ITEM-1) *items_row);
        #pragma unroll
        for(i=0; i<TRUNC; i++) {
            powAlpha *= alpha;
            sum += read_src(src, src_offset + top_area_position + (LINES_PER_ITEM+i) *items_row) * powAlpha;
        }
        last = -alpha * sum;
    }

    write_dst_p(dst + dst_position, last);

    /* No border conditions. Work items are on a line,
     * thus all items are on the same branch. */
    if (top_area_position >= 0) {
        #pragma unroll 20
        for(i=1; i<LINES_PER_ITEM; i++) {
            src_position -= stride;
            dst_position -= stride;
            /* See Eq. (18) */
            last = alpha*(last - read_src_p(src + src_position));
            write_dst_p(dst + dst_position, last);
        }
    } else {
        i = LINES_PER_ITEM-2;
        /* Process is done by unrolling blocks of 20 lines.
         * Process the blocks that fit in the image before
         * taking extra care for the last one. */
        while (top_area_y + (i-20) >= 0) {
            #pragma unroll
            for(int count=0; count<20; count++,i--) {
                src_position -= stride;
                dst_position -= stride;
                /* See Eq. (18) */
                last = alpha * (last - read_src_p(src + src_position));
                write_dst_p(dst + dst_position, last);
            }
        }
        for(; i>=0; i--) {
            /* Check if we're not outside of the image already */
            if (top_area_y + i < 0)
                return;
            src_position -= stride;
            dst_position -= stride;
            /* See Eq. (18) */
            last = alpha * (last - read_src_p(src + src_position));
            write_dst_p(dst + dst_position, last);
        }
    }
}
