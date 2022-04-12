/*
 * Copyright 2018 Thibaud Briand
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

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if defined(cl_khr_fp64) && defined(DOUBLE_PRECISION_KERNEL)

/// Constant B-spline function (KernelRadius = 0.5)
double BSpline0(double x) {
    x = fabs(x);
    if(x < 0.5)
        return 1.;
    if(x==0.5)
        return 0.5;
    return 0.;
}

/// Linear B-spline function (KernelRadius = 1)
double BSpline1(double x) {
    x = fabs(x);
    if(x < 1.)
        return 1.-x;
    return 0.;
}

/// Quadratic B-spline function (KernelRadius = 1.5)
double BSpline2(double x) {
    x = fabs(x);
    if(x < 0.5)
        return 1.5 - 2.*x*x;
    if(x < 1.5) {
        x = 1.5 - x;
        return x*x;
    }
    return 0.;
}

/// Cubic B-spline function (KernelRadius = 2)

double BSpline3(double x) {
    x = fabs(x);
    if(x < 1.)
        return (4. + (-6. + 3.*x)*x*x);
    if(x < 2.) {
        x = 2. - x;
        return x*x*x;
    }
    return 0.;
}

/// Quartic B-spline function (KernelRadius = 2.5)
double BSpline4(double x) {
    x = fabs(x);
    if(x <= 0.5) {
        x *= x;
        return (14.375 + (-15. + 6.*x)*x);
    }
    if(x < 1.5) {
        x = 1.5 - x;
        return (1. + (4. + (6. + (4. - 4.*x)*x)*x)*x);
    }
    if(x < 2.5) {
        x = 2.5 - x;
        x *= x;
        return x*x;
    }
    return 0.;
}

/// Quintic B-spline function (KernelRadius = 3)
double BSpline5(double x) {
    x = fabs(x);
    if(x <= 1.) {
        double x2 = x*x;
        return (((-10.*x + 30.)*x2 - 60.)*x2 + 66.);
    }
    if(x < 2.) {
        x = 2. - x;
        return (1. + (5. + (10. + (10. + (5. - 5.*x)*x)*x)*x)*x);
    }
    if(x < 3.) {
        x = 3. - x;
        double x2 = x*x;
        return x2*x2*x;
    }
    return 0.;
}

/// Sextic B-spline function (KernelRadius = 3.5)
double BSpline6(double x) {
    x = fabs(x);
    if(x <= 0.5) {
        x *= x;
        return (367.9375 + (-288.75 + (105. - 20.*x)*x)*x);
    }
    if(x < 1.5) {
        x = 1.5 - x;
        return (57. + (150. + (135. + (20. + (-45. + (-30. + 15.*x)*x)*x)*x)*x)*x);
    }
    if(x < 2.5) {
        x = 2.5 - x;
        return (1. + (6. + (15. + ( 20. + (15. + (6. - 6.*x)*x)*x)*x)*x)*x);
    }
    if(x < 3.5) {
        x = 3.5 - x;
        x = x*x;
        return x*x*x;
    }
    return 0.;
}

/// Septic B-spline function (KernelRadius = 4)
double BSpline7(double x) {
    x = fabs(x);
    if(x <= 1.) {
        double x2 = x*x;
        return ((((35.*x - 140.)*x2 + 560.)*x2 - 1680.)*x2 + 2416.);
    }
    if(x < 2.) {
        x = 2. - x;
        return (120. + (392. + (504. + (280. + (-84. + (-42. +
            21.*x)*x)*x*x)*x)*x)*x);
    }
    if(x < 3.) {
        x = 3. - x;
        return (((((((-7.*x + 7.)*x + 21.)*x + 35.)*x + 35.)*x
            + 21.)*x + 7.)*x + 1.);
    }
    if(x < 4.) {
        x = 4. - x;
        double x2 = x*x;
        return x2*x2*x2*x;
    }
    return 0.;
}

/// Octic B-spline function (KernelRadius = 4.5)
double BSpline8(double x) {
    x = fabs(x);
    if(x < 0.5) {
        x *= x;
        return (18261.7734375 + (-11379.375 + (3386.25 + (-630. + 70.*x)*x)*x)*x);
    }
    if(x <= 1.5) {
        x = 1.5 - x;
        return (4293. + (8568. + (5292. + (-504. + (-1890. + (-504. + (252. + (168.
                - 56.*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 2.5) {
        x = 2.5 - x;
        return (247. + (952. + (1540. + (1288. + (49.0 + (-56. +(-140. + (-56.
                + 28.*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 3.5) {
        x = 3.5 - x;
        return (1.+ (8.+ (28.+ (56.+ (70.+ (56.+ (28.+ (8. - 8.*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 4.5) {
        x = 4.5 - x;
        x = x*x; x = x*x; // x^4
        return x*x;
    }
    return 0.;
}

/// Nonic B-spline function (KernelRadius = 5)
double BSpline9(double x) {
    x = fabs(x);
    if(x <= 1.) {
        double x2 = x*x;
        return (((((-63.*x + 315.)*x2 - 2100.)*x2 + 11970.)*x2
            - 44100.)*x2 + 78095.)*2;
    }
    if(x <= 2.) {
        x = 2. - x;
        return (14608. + (36414. + (34272. + (11256. + (-4032. + (-4284. + (-672.
                + (504. + (252. - 84.*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 3.) {
        x = 3. - x;
        return (502. + (2214. + (4248. + (4536. + (2772. + (756. + (-168. + (-216.
                + (-72. + 36.*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 4.) {
        x = 4. - x;
        return (1. + (9. + (36. + (84. + (126. + (126. + (84. + (36. + (9.
                - 9.*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 5.) {
        x = 5. - x;
        double x3 = x*x*x;
        return x3*x3*x3;
    }
    return 0.;
}

/// 10th-Degree B-spline function (KernelRadius = 5.5)
double BSpline10(double x) {
    x = fabs(x);
    if(x < 0.5) {
        x *= x;
        return (1491301.23828125 + (-769825.546875 + (191585.625 + (-30607.5
                + (3465. - 252.*x)*x)*x)*x)*x);
    }
    if(x <= 1.5) {
        x = 1.5 - x;
        return (455192.+ (736260.+ (327600.+ (-95760. + (-119280. + (-13608.
                + (16800.+ (5040.+ (-1260.+(-840.+210.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 2.5) {
        x = 2.5 - x;
        return (47840. + (141060. + (171000. + (100080. + (16800. + (-13608. + (-8400.
                + (-720. + (900. + (360. - 120.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 3.5) {
        x = 3.5 - x;
        return (1013. + (5010. + (11025. + (14040. + (11130. + (5292. + (1050.
                + (-360. + (-315. + (-90. + 45.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 4.5) {
        x = 4.5 - x;
        return (1. + (10. + (45. + (120. + (210. + (252. + (210. + (120. + (45. + (10.
                -10.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 5.5) {
        x = 5.5 - x;
        x = x*x; // x2;
        double x4 = x*x;
        return x4*x4*x;
    }
    return 0;
}

/// 11th-Degree B-spline function (KernelRadius = 6)
double BSpline11(double x) {
    x = fabs(x);
    if(x <= 1) {
        double x2 = x*x;
        return (15724248. + (-7475160. + (1718640. + (-255024. + (27720.
            + (-2772. + 462.*x)*x2)*x2)*x2)*x2)*x2);
    }
    if(x <= 2.) {
        x = 2. - x;
        return (2203488. + (4480872. + (3273600. + (574200. + (-538560.
            + (-299376. + (39600. + (7920. + (-2640. + (-1320.
            + 330.*x)*x)*x)*x)*x*x)*x)*x)*x)*x)*x);
    }
    if(x <= 3.) {
        x = 3. - x;
        return (152637. + (515097. + (748275. + (586575. + (236610. + (12474.
            + (-34650. + (-14850. + (-495. + (1485.
            + (495.-165.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 4.) {
        x = 4. - x;
        return (2036. + (11132. + (27500. + (40260. + (38280. + (24024. + (9240.
            + (1320. + (-660. + (-440. + (-110.
            + 55.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 5.) {
        x = 5. - x;
        return (1. + (11. + (55. + (165. + (330. + (462. + (462. + (330. + (165.
            + (55. + (11. - 11.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 6.) {
        x = 6. - x;
        double x2 = x*x;
        double x4 = x2*x2;
        return x4*x4*x2*x;
    }
    return 0.;
}

#define KERNEL_TYPE double
#define convert_KT convert_double
#else

/// Constant B-spline function (KernelRadius = 0.5)
float BSpline0(float x) {
    x = fabs(x);
    if(x < 0.5f)
        return 1.f;
    if(x==0.5f)
        return 0.5f;
    return 0.f;
}

/// Linear B-spline function (KernelRadius = 1)
float BSpline1(float x) {
    x = fabs(x);
    if(x < 1.f)
        return 1.f-x;
    return 0.f;
}

/// Quadratic B-spline function (KernelRadius = 1.5)
float BSpline2(float x) {
    x = fabs(x);
    if(x < 0.5f)
        return 1.5f - 2.f*x*x;
    if(x < 1.5f) {
        x = 1.5f - x;
        return x*x;
    }
    return 0.f;
}

/// Cubic B-spline function (KernelRadius = 2)

float BSpline3(float x) {
    x = fabs(x);
    if(x < 1.f)
        return (4.f + (-6.f + 3.f*x)*x*x);
    if(x < 2.f) {
        x = 2.f - x;
        return x*x*x;
    }
    return 0.f;
}

/// Quartic B-spline function (KernelRadius = 2.5)
float BSpline4(float x) {
    x = fabs(x);
    if(x <= 0.5f) {
        x *= x;
        return (14.375f + (-15.f + 6.f*x)*x);
    }
    if(x < 1.5f) {
        x = 1.5f - x;
        return (1.f + (4.f + (6.f + (4.f - 4.f*x)*x)*x)*x);
    }
    if(x < 2.5f) {
        x = 2.5f - x;
        x *= x;
        return x*x;
    }
    return 0.f;
}

/// Quintic B-spline function (KernelRadius = 3)
float BSpline5(float x) {
    x = fabs(x);
    if(x <= 1.f) {
        float x2 = x*x;
        return (((-10.f*x + 30.f)*x2 - 60.f)*x2 + 66.f);
    }
    if(x < 2.f) {
        x = 2.f - x;
        return (1.f + (5.f + (10.f + (10.f + (5.f - 5.f*x)*x)*x)*x)*x);
    }
    if(x < 3.f) {
        x = 3.f - x;
        float x2 = x*x;
        return x2*x2*x;
    }
    return 0.f;
}

/// Sextic B-spline function (KernelRadius = 3.5)
float BSpline6(float x) {
    x = fabs(x);
    if(x <= 0.5f) {
        x *= x;
        return (367.9375f + (-288.75f + (105.f - 20.f*x)*x)*x);
    }
    if(x < 1.5f) {
        x = 1.5f - x;
        return (57.f + (150.f + (135.f + (20.f + (-45.f + (-30.f + 15.f*x)*x)*x)*x)*x)*x);
    }
    if(x < 2.5f) {
        x = 2.5f - x;
        return (1.f + (6.f + (15.f + ( 20.f + (15.f + (6.f - 6.f*x)*x)*x)*x)*x)*x);
    }
    if(x < 3.5f) {
        x = 3.5f - x;
        x = x*x;
        return x*x*x;
    }
    return 0.f;
}

/// Septic B-spline function (KernelRadius = 4)
float BSpline7(float x) {
    x = fabs(x);
    if(x <= 1.f) {
        float x2 = x*x;
        return ((((35.f*x - 140.f)*x2 + 560.f)*x2 - 1680.f)*x2 + 2416.f);
    }
    if(x < 2.f) {
        x = 2.f - x;
        return (120.f + (392.f + (504.f + (280.f + (-84.f + (-42.f +
            21.f*x)*x)*x*x)*x)*x)*x);
    }
    if(x < 3.f) {
        x = 3.f - x;
        return (((((((-7.f*x + 7.f)*x + 21.f)*x + 35.f)*x + 35.f)*x
            + 21.f)*x + 7.f)*x + 1.f);
    }
    if(x < 4.f) {
        x = 4.f - x;
        float x2 = x*x;
        return x2*x2*x2*x;
    }
    return 0.f;
}

/// Octic B-spline function (KernelRadius = 4.5)
float BSpline8(float x) {
    x = fabs(x);
    if(x < 0.5f) {
        x *= x;
        return (18261.7734375f + (-11379.375f + (3386.25f + (-630.f + 70.f*x)*x)*x)*x);
    }
    if(x <= 1.5f) {
        x = 1.5f - x;
        return (4293.f + (8568.f + (5292.f + (-504.f + (-1890.f + (-504.f + (252.f + (168.f
                - 56.f*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 2.5f) {
        x = 2.5f - x;
        return (247.f + (952.f + (1540.f + (1288.f + (490.f + (-56.f +(-140.f + (-56.f
                + 28.f*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 3.5f) {
        x = 3.5f - x;
        return (1.f+ (8.f+ (28.f+ (56.f+ (70.f+ (56.f+ (28.f+ (8.f - 8.f*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 4.5f) {
        x = 4.5f - x;
        x = x*x; x = x*x; // x^4
        return x*x;
    }
    return 0.f;
}

/// Nonic B-spline function (KernelRadius = 5)
float BSpline9(float x) {
    x = fabs(x);
    if(x <= 1.f) {
        float x2 = x*x;
        return (((((-63.f*x + 315.f)*x2 - 2100.f)*x2 + 11970.f)*x2
            - 44100.f)*x2 + 78095.f)*2;
    }
    if(x <= 2.f) {
        x = 2.f - x;
        return (14608.f + (36414.f + (34272.f + (11256.f + (-4032.f + (-4284.f + (-672.f
                + (504.f + (252.f - 84.f*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 3.f) {
        x = 3.f - x;
        return (502.f + (2214.f + (4248.f + (4536.f + (2772.f + (756.f + (-168.f + (-216.f
                + (-72.f + 36.f*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 4.f) {
        x = 4.f - x;
        return (1.f + (9.f + (36.f + (84.f + (126.f + (126.f + (84.f + (36.f + (9.f
                - 9.f*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 5.f) {
        x = 5.f - x;
        float x3 = x*x*x;
        return x3*x3*x3;
    }
    return 0.f;
}

/// 10th-Degree B-spline function (KernelRadius = 5.5)
float BSpline10(float x) {
    x = fabs(x);
    if(x < 0.5f) {
        x *= x;
        return (1491301.23828125f + (-769825.546875f + (191585.625f + (-30607.5f
                + (3465.f - 252.f*x)*x)*x)*x)*x);
    }
    if(x <= 1.5f) {
        x = 1.5f - x;
        return (455192.f+ (736260.f+ (327600.f+ (-95760.f + (-119280.f + (-13608.f
                + (16800.f+ (5040.f+ (-1260.f+(-840.f+210.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 2.5f) {
        x = 2.5f - x;
        return (47840.f + (141060.f + (171000.f + (100080.f + (16800.f + (-13608.f + (-8400.f
                + (-720.f + (900.f + (360.f - 120.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 3.5f) {
        x = 3.5f - x;
        return (1013.f + (5010.f + (11025.f + (14040.f + (11130.f + (5292.f + (1050.f
                + (-360.f + (-315.f + (-90.f + 45.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 4.5f) {
        x = 4.5f - x;
        return (1.f + (10.f + (45.f + (120.f + (210.f + (252.f + (210.f + (120.f + (45.f + (10.f
                -10.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 5.5f) {
        x = 5.5f - x;
        x = x*x; // x2;
        float x4 = x*x;
        return x4*x4*x;
    }
    return 0.f;
}

/// 11th-Degree B-spline function (KernelRadius = 6)
float BSpline11(float x) {
    x = fabs(x);
    if(x <= 1) {
        float x2 = x*x;
        return (15724248.f + (-7475160.f + (1718640.f + (-255024.f + (27720.f
            + (-2772.f + 462.f*x)*x2)*x2)*x2)*x2)*x2);
    }
    if(x <= 2.f) {
        x = 2.f - x;
        return (2203488.f + (4480872.f + (3273600.f + (574200.f + (-538560.f
            + (-299376.f + (39600.f + (7920.f + (-2640.f + (-1320.f
            + 330.f*x)*x)*x)*x)*x*x)*x)*x)*x)*x)*x);
    }
    if(x <= 3.f) {
        x = 3.f - x;
        return (152637.f + (515097.f + (748275.f + (586575.f + (236610.f + (12474.f
            + (-34650.f + (-14850.f + (-495.f + (1485.f
            + (495.f-165.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x <= 4.f) {
        x = 4.f - x;
        return (2036.f + (11132.f + (27500.f + (40260.f + (38280.f + (24024.f + (9240.f
            + (1320.f + (-660.f + (-440.f + (-110.f
            + 55.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 5.f) {
        x = 5.f - x;
        return (1.f + (11.f + (55.f + (165.f + (330.f + (462.f + (462.f + (330.f + (165.f
            + (55.f + (11.f - 11.f*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x);
    }
    if(x < 6.f) {
        x = 6.f - x;
        float x2 = x*x;
        float x4 = x2*x2;
        return x4*x4*x2*x;
    }
    return 0.f;
}

#define KERNEL_TYPE float
#define convert_KT convert_float

#endif

#if defined(cl_khr_fp64) && defined(DOUBLE_PRECISION_COMPUTE_PREFILTER)

// These two have no poles
__constant double BSpline0Poles[1] =
    {0.};

__constant double BSpline1Poles[1] =
    {0.};

__constant double BSpline2Poles[1] =
    {-1.715728752538099e-1}; // -3+sqrt(8)

__constant double BSpline3Poles[1] =
    {-2.679491924311227e-1}; // -2+sqrt(3)

__constant double BSpline4Poles[2] =
    {-3.6134122590021989e-1,-1.3725429297339109e-2};

__constant double BSpline5Poles[2] =
    {-4.305753470999738e-1,  // sqrt(105)/2+sqrt(135-13*sqrt(105))/sqrt(2)-13/2.
     -4.309628820326465e-2}; // sqrt(13*sqrt(105)+135)/sqrt(2)-sqrt(105)/2-13/2.

__constant double BSpline6Poles[3] =
    {-4.8829458930303893e-1,-8.1679271076238694e-2,-1.4141518083257976e-3};

__constant double BSpline7Poles[3] =
    {-5.352804307964382e-1, -1.225546151923267e-1,-9.148694809608277e-3};

__constant double BSpline8Poles[4] =
    {-5.7468690924876376e-1,-1.6303526929728299e-1,
     -2.3632294694844336e-2,-1.5382131064168442e-4};

__constant double BSpline9Poles[4] =
    {-6.079973891686259e-1,-2.017505201931532e-1,
     -4.322260854048175e-2,-2.121306903180818e-3};

__constant double BSpline10Poles[5] =
    {-6.365506639694650e-1,-2.381827983775487e-1,-6.572703322831758e-2,
     -7.528194675547741e-3,-1.698276282327549e-5};

__constant double BSpline11Poles[5] =
    {-6.612660689007345e-1,-2.721803492947859e-1,-8.975959979371331e-2,
     -1.666962736623466e-2,-5.105575344465021e-4};

#define COMPUTE_PREFILTER_TYPE double
#define convert_CPT convert_double
#else

// These two have no poles
__constant float BSpline0Poles[1] =
    {0.f};

__constant float BSpline1Poles[1] =
    {0.f};

__constant float BSpline2Poles[1] =
    {-1.715728752538099e-1f}; // -3+sqrt(8)

__constant float BSpline3Poles[1] =
    {-2.679491924311227e-1f}; // -2+sqrt(3)

__constant float BSpline4Poles[2] =
    {-3.6134122590021989e-1f,-1.3725429297339109e-2f};

__constant float BSpline5Poles[2] =
    {-4.305753470999738e-1f,  // sqrt(105)/2+sqrt(135-13*sqrt(105))/sqrt(2)-13/2.
     -4.309628820326465e-2f}; // sqrt(13*sqrt(105)+135)/sqrt(2)-sqrt(105)/2-13/2.

__constant float BSpline6Poles[3] =
    {-4.8829458930303893e-1f,-8.1679271076238694e-2f,-1.4141518083257976e-3f};

__constant float BSpline7Poles[3] =
    {-5.352804307964382e-1f, -1.225546151923267e-1f,-9.148694809608277e-3f};

__constant float BSpline8Poles[4] =
    {-5.7468690924876376e-1f,-1.6303526929728299e-1f,
     -2.3632294694844336e-2f,-1.5382131064168442e-4f};

__constant float BSpline9Poles[4] =
    {-6.079973891686259e-1f,-2.017505201931532e-1f,
     -4.322260854048175e-2f,-2.121306903180818e-3f};

__constant float BSpline10Poles[5] =
    {-6.365506639694650e-1f,-2.381827983775487e-1f,-6.572703322831758e-2f,
     -7.528194675547741e-3f,-1.698276282327549e-5f};

__constant float BSpline11Poles[5] =
    {-6.612660689007345e-1f,-2.721803492947859e-1f,-8.975959979371331e-2f,
     -1.666962736623466e-2f,-5.105575344465021e-4f};

#define COMPUTE_PREFILTER_TYPE float
#define convert_CPT convert_float
#endif

#if defined(cl_khr_fp64) && defined(DOUBLE_PRECISION_STORAGE_PREFILTER)
#define STORAGE_PREFILTER_TYPE double
#define convert_SPT convert_double
#else
#define STORAGE_PREFILTER_TYPE float
#define convert_SPT convert_float
#endif

#define read_src_p(src) convert_CPT(*((__global const STORAGE_PREFILTER_TYPE * restrict)(src)))
#define read_src(src, i) read_src_p((src) + (i) * sizeof(STORAGE_PREFILTER_TYPE))
#define write_dst_p(dst, data) *((__global STORAGE_PREFILTER_TYPE * restrict)(dst)) = convert_SPT(data)
#define write_dst(dst, i, data) write_dst_p((dst) + (i) * sizeof(STORAGE_PREFILTER_TYPE), data)

#if ORDER == 0
#define RADIUS 0.5
#define Poles BSpline0Poles
#define BSpline BSpline0

#elif ORDER == 1
#define RADIUS 1
#define Poles BSpline1Poles
#define BSpline BSpline1

#elif ORDER == 2
#define RADIUS 1.5
#define Poles BSpline2Poles
#define BSpline BSpline2

#elif ORDER == 3
#define RADIUS 2
#define Poles BSpline3Poles
#define BSpline BSpline3

#elif ORDER == 4
#define RADIUS 2.5
#define Poles BSpline4Poles
#define BSpline BSpline4

#elif ORDER == 5
#define RADIUS 3
#define Poles BSpline5Poles
#define BSpline BSpline5

#elif ORDER == 6
#define RADIUS 3.5
#define Poles BSpline6Poles
#define BSpline BSpline6

#elif ORDER == 7
#define RADIUS 4
#define Poles BSpline7Poles
#define BSpline BSpline7

#elif ORDER == 8
#define RADIUS 4.5
#define Poles BSpline8Poles
#define BSpline BSpline8

#elif ORDER == 9
#define RADIUS 5
#define Poles BSpline9Poles
#define BSpline BSpline9

#elif ORDER == 10
#define RADIUS 5.5
#define Poles BSpline10Poles
#define BSpline BSpline10

#elif ORDER == 11
#define RADIUS 6
#define Poles BSpline11Poles
#define BSpline BSpline11

#else
#error ORDER Unsupported
#endif
