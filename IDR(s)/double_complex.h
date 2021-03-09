/*!
\file double_complex.h
\brief Contains commonly used macros, functions and overloaded operators(all static inline) for cuDoubleComplex(datatype for complex numbers provided by CUDA)  
*/


 //Reference - Magma Sparse Library
# pragma once   

#include <math.h>

#include<cuComplex.h>

typedef cuDoubleComplex DoubleComplex;

#define MAKE(r,i)     make_cuDoubleComplex(r, i)    ///< @return complex number r + i*sqrt(-1).
#define REAL(a)       (a).x                         ///< @return real component of a.
#define IMAG(a)       (a).y                         ///< @return imaginary component of a.
#define ADD(a, b)     cuCadd(a, b)                  ///< @return (a + b).
#define SUB(a, b)     cuCsub(a, b)                  ///< @return (a - b).
#define MUL(a, b)     cuCmul(a, b)                  ///< @return (a * b).
#define DIV(a, b)     cuCdiv(a, b)                  ///< @return (a / b).
#define ABS(a)        cuCabs(a)                     ///< @return absolute value, |a| = sqrt( real(a)^2 + imag(a)^2 ).
#define ABS1(a)       (fabs((a).x) + fabs((a).y))   ///< @return 1-norm absolute value, | real(a) | + | imag(a) |.
#define CONJ(a)       cuConj(a)                     ///< @return conjugate of a.

// =============================================================================
// names to match C++ std complex functions

// real component of complex number x; x for real number.
__host__ __device__ static inline double real(const DoubleComplex& x) { return REAL(x); }

__host__ __device__  static inline double real(const double& x) { return x; }


// return imaginary component of complex number x; 0 for real number.

__host__ __device__ static inline double imag(const DoubleComplex& x) { return IMAG(x); }

__host__ __device__ static inline double imag(const double& x) { return 0.; }


//return conjugate of complex number x; x for real number.
__host__ __device__ static inline DoubleComplex conj(const DoubleComplex& x) { return CONJ(x); }

__host__ __device__ static inline double             conj(const double& x) { return x; }



// return 2-norm absolute value of complex number x: sqrt( real(x)^2 + imag(x)^2 ).
//         math.h or cmath provide fabs for real numbers.

__host__ __device__ static inline double fabs(const DoubleComplex& x) { return ABS(x); }

// already have fabs( double ) in math.h

// return 1-norm absolute value of complex nmuber x: | real(x) | + | imag(x) |.

__host__ __device__ static inline double abs1(const DoubleComplex& x) { return ABS1(x); }


__host__ __device__ static inline double abs1(const double& x) { return x >= 0 ? x : -1 * x; }

// =============================================================================
// DoubleComplex

// ---------- negate
__host__ __device__ static inline DoubleComplex
operator - (const DoubleComplex& a)
{
    return MAKE(-real(a),
        -imag(a));
}


// ---------- add
__host__ __device__ static inline DoubleComplex
operator + (const DoubleComplex a, const DoubleComplex b)
{
    return MAKE(real(a) + real(b),
        imag(a) + imag(b));
}

__host__ __device__ static inline DoubleComplex
operator + (const DoubleComplex a, const double s)
{
    return MAKE(real(a) + s,
        imag(a));
}

__host__ __device__ static inline DoubleComplex
operator + (const double s, const DoubleComplex b)
{
    return MAKE(s + real(b),
        imag(b));
}

__host__ __device__ static inline DoubleComplex&
operator += (DoubleComplex& a, const DoubleComplex b)
{
    a = MAKE(real(a) + real(b),
        imag(a) + imag(b));
    return a;
}

__host__ __device__ static inline DoubleComplex&
operator += (DoubleComplex& a, const double s)
{
    a = MAKE(real(a) + s,
        imag(a));
    return a;
}


// ---------- subtract
__host__ __device__ static inline DoubleComplex
operator - (const DoubleComplex a, const DoubleComplex b)
{
    return MAKE(real(a) - real(b),
        imag(a) - imag(b));
}

__host__ __device__ static inline DoubleComplex
operator - (const DoubleComplex a, const double s)
{
    return MAKE(real(a) - s,
        imag(a));
}

__host__ __device__ static inline DoubleComplex
operator - (const double s, const DoubleComplex b)
{
    return MAKE(s - real(b),
        -imag(b));
}

__host__ __device__ static inline DoubleComplex&
operator -= (DoubleComplex& a, const DoubleComplex b)
{
    a = MAKE(real(a) - real(b),
        imag(a) - imag(b));
    return a;
}

__host__ __device__ static inline DoubleComplex&
operator -= (DoubleComplex& a, const double s)
{
    a = MAKE(real(a) - s,
        imag(a));
    return a;
}


// ---------- multiply
__host__ __device__ static inline DoubleComplex
operator * (const DoubleComplex a, const DoubleComplex b)
{
    return MAKE(real(a) * real(b) - imag(a) * imag(b),
        imag(a) * real(b) + real(a) * imag(b));
}

__host__ __device__ static inline DoubleComplex
operator * (const DoubleComplex a, const double s)
{
    return MAKE(real(a) * s,
        imag(a) * s);
}

__host__ __device__ static inline DoubleComplex
operator * (const double s, const DoubleComplex a)
{
    return MAKE(real(a) * s,
        imag(a) * s);
}

__host__ __device__ static inline DoubleComplex&
operator *= (DoubleComplex& a, const DoubleComplex b)
{
    a = MAKE(real(a) * real(b) - imag(a) * imag(b),
        imag(a) * real(b) + real(a) * imag(b));
    return a;
}

__host__ __device__ static inline DoubleComplex&
operator *= (DoubleComplex& a, const double s)
{
    a = MAKE(real(a) * s,
        imag(a) * s);
    return a;
}


// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__ static inline DoubleComplex
operator / (const DoubleComplex x, const DoubleComplex y)
{
    double a = real(x);
    double b = imag(x);
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if (fabs(d) < fabs(c)) {
        e = d / c;
        f = c + d * e;
        p = (a + b * e) / f;
        q = (b - a * e) / f;
    }
    else {
        e = c / d;
        f = d + c * e;
        p = (b + a * e) / f;
        q = (-a + b * e) / f;
    }
    return MAKE(p, q);
}

__host__ __device__ static inline DoubleComplex
operator / (const DoubleComplex a, const double s)
{
    return MAKE(real(a) / s,
        imag(a) / s);
}

__host__ __device__ static inline DoubleComplex
operator / (const double a, const DoubleComplex y)
{
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if (fabs(d) < fabs(c)) {
        e = d / c;
        f = c + d * e;
        p = a / f;
        q = -a * e / f;
    }
    else {
        e = c / d;
        f = d + c * e;
        p = a * e / f;
        q = -a / f;
    }
    return MAKE(p, q);
}

__host__ __device__ static inline DoubleComplex&
operator /= (DoubleComplex& a, const DoubleComplex b)
{
    a = a / b;
    return a;
}

__host__ __device__ static inline DoubleComplex&
operator /= (DoubleComplex& a, const double s)
{
    a = MAKE(real(a) / s,
        imag(a) / s);
    return a;
}


// ---------- equality
__host__ __device__ static inline bool
operator == (const DoubleComplex a, const DoubleComplex b)
{
    return (real(a) == real(b) &&
        imag(a) == imag(b));
}

__host__ __device__ static inline bool
operator == (const DoubleComplex a, const double s)
{
    return (real(a) == s &&
        imag(a) == 0.);
}

__host__ __device__ static inline bool
operator == (const double s, const DoubleComplex a)
{
    return (real(a) == s &&
        imag(a) == 0.);
}


// ---------- not equality
__host__ __device__ static inline bool
operator != (const DoubleComplex a, const DoubleComplex b)
{
    return !(a == b);
}

__host__ __device__ static inline bool
operator != (const DoubleComplex a, const double s)
{
    return !(a == s);
}

__host__ __device__ static inline bool
operator != (const double s, const DoubleComplex a)
{
    return !(a == s);
}



