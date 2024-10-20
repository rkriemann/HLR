#ifndef HLR_UTILS_FLOPS_HH
#define HLR_UTILS_FLOPS_HH
//
//
// @file flops.h
//
//  File provided by Univ. of Tennessee,
//
// @version 1.0.0
// @author Mathieu Faverge
// @date November 2014
//
// @version 1.0.1
// @author Ronald Kriemann
// @date November 2024
// @changes C++ version with uint64_t instead of double
//
//
// This file provide the flops formula for all Level 3 BLAS and some
// Lapack routines.  Each macro uses the same size parameters as the
// function associated and provide one formula for additions and one
// for multiplications. Example to use these macros:
//
//    FLOPS_ZGEMM( m, n, k )
//
// All the formula are reported in the LAPACK Lawn 41:
//     http://www.netlib.org/lapack/lawns/lawn41.ps
//

#include <hpro/blas/flops.hh>

#if 0
//
//           Generic formula coming from LAWN 41
//
using flops_t = double;

//
// Level 2 BLAS
//
#define FMULS_GEMV(m_, n_) ((m_) * (n_) + 2 * (m_))
#define FADDS_GEMV(m_, n_) ((m_) * (n_)           )

#define FMULS_SYMV(n_) FMULS_GEMV( (n_), (n_) )
#define FADDS_SYMV(n_) FADDS_GEMV( (n_), (n_) )
#define FMULS_HEMV FMULS_SYMV
#define FADDS_HEMV FADDS_SYMV

//
// Level 3 BLAS
//
#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

#define FMULS_SYMM(side_, m_, n_) ( ( (side_) == from_left ) ? FMULS_GEMM((m_), (m_), (n_)) : FMULS_GEMM((m_), (n_), (n_)) )
#define FADDS_SYMM(side_, m_, n_) ( ( (side_) == from_left ) ? FADDS_GEMM((m_), (m_), (n_)) : FADDS_GEMM((m_), (n_), (n_)) )
#define FMULS_HEMM FMULS_SYMM
#define FADDS_HEMM FADDS_SYMM

#define FMULS_SYRK(k_, n_) (0.5 * (k_) * (n_) * ((n_)+1))
#define FADDS_SYRK(k_, n_) (0.5 * (k_) * (n_) * ((n_)+1))
#define FMULS_HERK FMULS_SYRK
#define FADDS_HERK FADDS_SYRK

#define FMULS_SYR2K(k_, n_) ((k_) * (n_) * (n_)        )
#define FADDS_SYR2K(k_, n_) ((k_) * (n_) * (n_) + (n_))
#define FMULS_HER2K FMULS_SYR2K
#define FADDS_HER2K FADDS_SYR2K

#define FMULS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))


#define FMULS_TRMM(side_, m_, n_) ( ( (side_) == from_left ) ? FMULS_TRMM_2((m_), (n_)) : FMULS_TRMM_2((n_), (m_)) )
#define FADDS_TRMM(side_, m_, n_) ( ( (side_) == from_left ) ? FADDS_TRMM_2((m_), (n_)) : FADDS_TRMM_2((n_), (m_)) )

#define FMULS_TRSM FMULS_TRMM
#define FADDS_TRSM FADDS_TRMM

//
// Lapack
//
#define FMULS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_) - 1 ) + (n_)) + (2 / 3.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_) - 1 ) + (m_)) + (2 / 3.) * (n_)) )
#define FADDS_GETRF(m_, n_) ( ((m_) < (n_)) \
    ? (0.5 * (m_) * ((m_) * ((n_) - (1./3.) * (m_)      ) - (n_)) + (1 / 6.) * (m_)) \
    : (0.5 * (n_) * ((n_) * ((m_) - (1./3.) * (n_)      ) - (m_)) + (1 / 6.) * (n_)) )

#define FMULS_GETRI(n_) ( (n_) * ((5 / 6.) + (n_) * ((2 / 3.) * (n_) + 0.5)) )
#define FADDS_GETRI(n_) ( (n_) * ((5 / 6.) + (n_) * ((2 / 3.) * (n_) - 1.5)) )

#define FMULS_GETRS(n_, nrhs_) ((nrhs_) * (n_) *  (n_)      )
#define FADDS_GETRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) - 1 ))

#define FMULS_POTRF(n_) ((n_) * (((1 / 6.) * (n_) + 0.5) * (n_) + (1 / 3.)))
#define FADDS_POTRF(n_) ((n_) * (((1 / 6.) * (n_)      ) * (n_) - (1 / 6.)))

#define FMULS_POTRI(n_) ( (n_) * ((2 / 3.) + (n_) * ((1 / 3.) * (n_) + 1 )) )
#define FADDS_POTRI(n_) ( (n_) * ((1 / 6.) + (n_) * ((1 / 3.) * (n_) - 0.5)) )

#define FMULS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) + 1 ))
#define FADDS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) - 1 ))

//SPBTRF
//SPBTRS
//SSYTRF
//SSYTRI
//SSYTRS

#define FMULS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_)) +    (m_) + 23 / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) + 2.*(n_) + 23 / 6.)) )
#define FADDS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_))           +  5 / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) +    (n_) +  5 / 6.)) )

#define FMULS_GEQRT(m_, n_) (0.5 * (m_)*(n_))
#define FADDS_GEQRT(m_, n_) (0.5 * (m_)*(n_))

#define FMULS_GEQLF(m_, n_) FMULS_GEQRF(m_, n_)
#define FADDS_GEQLF(m_, n_) FADDS_GEQRF(m_, n_)

#define FMULS_GERQF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_)) +    (m_) + 29 / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) + 2.*(n_) + 29 / 6.)) )
#define FADDS_GERQF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * ( -0.5-(1./3.) * (n_) + (m_)) +    (m_) +  5 / 6.)) \
    : ((m_) * ((m_) * (  0.5-(1./3.) * (m_) + (n_)) +         +  5 / 6.)) )

#define FMULS_GELQF(m_, n_) FMULS_GERQF(m_, n_)
#define FADDS_GELQF(m_, n_) FADDS_GERQF(m_, n_)

#define FMULS_UNGQR(m_, n_, k_) ((k_) * (2.* (m_) * (n_) +   2 * (n_) - 5./3 + (k_) * ( 2./3 * (k_) - ((m_) + (n_)) - 1.)))
#define FADDS_UNGQR(m_, n_, k_) ((k_) * (2.* (m_) * (n_) + (n_) - (m_) + 1./3 + (k_) * ( 2./3 * (k_) - ((m_) + (n_))     )))
#define FMULS_ORGQR FMULS_UNGQR
#define FADDS_ORGQR FADDS_UNGQR

#define FMULS_UNGQL FMULS_UNGQR
#define FADDS_UNGQL FADDS_UNGQR
#define FMULS_ORGQL FMULS_UNGQR
#define FADDS_ORGQL FADDS_UNGQR

#define FMULS_UNGRQ(m_, n_, k_) ((k_) * (2.* (m_) * (n_) + (m_) + (n_) - 2./3 + (k_) * ( 2./3 * (k_) - ((m_) + (n_)) - 1.)))
#define FADDS_UNGRQ(m_, n_, k_) ((k_) * (2.* (m_) * (n_) + (m_) - (n_) + 1./3 + (k_) * ( 2./3 * (k_) - ((m_) + (n_))     )))
#define FMULS_ORGRQ FMULS_UNGRQ
#define FADDS_ORGRQ FADDS_UNGRQ

#define FMULS_UNGLQ FMULS_UNGRQ
#define FADDS_UNGLQ FADDS_UNGRQ
#define FMULS_ORGLQ FMULS_UNGRQ
#define FADDS_ORGLQ FADDS_UNGRQ

#define FMULS_GEQRS(m_, n_, nrhs_) ((nrhs_) * ((n_) * ( 2.* (m_) - 0.5 * (n_) + 2.5)))
#define FADDS_GEQRS(m_, n_, nrhs_) ((nrhs_) * ((n_) * ( 2.* (m_) - 0.5 * (n_) + 0.5)))

#define FMULS_UNMQR(m_, n_, k_, side_) (( (side_) == from_left ) \
    ?  (2.*(n_)*(m_)*(k_) - (n_)*(k_)*(k_) + 2.*(n_)*(k_)) \
    :  (2.*(n_)*(m_)*(k_) - (m_)*(k_)*(k_) + (m_)*(k_) + (n_)*(k_) - 0.5*(k_)*(k_) + 0.5*(k_)))
#define FADDS_UNMQR(m_, n_, k_, side_) (( ((side_)) == from_left ) \
    ?  (2.*(n_)*(m_)*(k_) - (n_)*(k_)*(k_) + (n_)*(k_)) \
    :  (2.*(n_)*(m_)*(k_) - (m_)*(k_)*(k_) + (m_)*(k_)))
#define FMULS_ORMQR FMULS_UNMQR
#define FADDS_ORMQR FADDS_UNMQR

#define FMULS_UNMQL FMULS_UNMQR
#define FADDS_UNMQL FADDS_UNMQR
#define FMULS_ORMQL FMULS_UNMQR
#define FADDS_ORMQL FADDS_UNMQR

#define FMULS_UNMRQ FMULS_UNMQR
#define FADDS_UNMRQ FADDS_UNMQR
#define FMULS_ORMRQ FMULS_UNMQR
#define FADDS_ORMRQ FADDS_UNMQR

#define FMULS_UNMLQ FMULS_UNMQR
#define FADDS_UNMLQ FADDS_UNMQR
#define FMULS_ORMLQ FMULS_UNMQR
#define FADDS_ORMLQ FADDS_UNMQR

#define FMULS_TRTRI(n_) ((n_) * ((n_) * ( 1./6 * (n_) + 0.5 ) + 1./3.))
#define FADDS_TRTRI(n_) ((n_) * ((n_) * ( 1./6 * (n_) - 0.5 ) + 1./3.))

#define FMULS_GEHRD(n_) ( (n_) * ((n_) * (5./3 *(n_) + 0.5) - 7./6.) - 13 )
#define FADDS_GEHRD(n_) ( (n_) * ((n_) * (5./3 *(n_) - 1 ) - 2./3.) -  8 )

#define FMULS_SYTRD(n_) ( (n_) *  ( (n_) * ( 2./3 * (n_) + 2.5 ) - 1./6 ) - 15.)
#define FADDS_SYTRD(n_) ( (n_) *  ( (n_) * ( 2./3 * (n_) + 1  ) - 8./3 ) -  4.)
#define FMULS_HETRD FMULS_SYTRD
#define FADDS_HETRD FADDS_SYTRD

#define FMULS_GEBRD(m_, n_) ( ((m_) >= (n_)) \
    ? ((n_) * ((n_) * (2 * (m_) - 2./3 * (n_) + 2 )         + 20./3.)) \
    : ((m_) * ((m_) * (2 * (n_) - 2./3 * (m_) + 2 )         + 20./3.)) )
#define FADDS_GEBRD(m_, n_) ( ((m_) >= (n_)) \
    ? ((n_) * ((n_) * (2 * (m_) - 2./3 * (n_) + 1 ) - (m_) +  5./3.)) \
    : ((m_) * ((m_) * (2 * (n_) - 2./3 * (m_) + 1 ) - (n_) +  5./3.)) )

#define FMULS_LARFG(n_) (2*n_)
#define FADDS_LARFG(n_) (  n_)


//
//               Users functions
//*****************************************************************************/

//
// Level 2 BLAS
//
#define FLOPS_ZGEMV(m_, n_) (6 * FMULS_GEMV(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEMV(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGEMV(m_, n_) (6 * FMULS_GEMV(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEMV(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGEMV(m_, n_) (     FMULS_GEMV(flops_t(m_), flops_t(n_)) +       FADDS_GEMV(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGEMV(m_, n_) (     FMULS_GEMV(flops_t(m_), flops_t(n_)) +       FADDS_GEMV(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZHEMV(n_) (6 * FMULS_HEMV(flops_t(n_)) + 2 * FADDS_HEMV(flops_t(n_)) )
#define FLOPS_CHEMV(n_) (6 * FMULS_HEMV(flops_t(n_)) + 2 * FADDS_HEMV(flops_t(n_)) )

#define FLOPS_ZSYMV(n_) (6 * FMULS_SYMV(flops_t(n_)) + 2 * FADDS_SYMV(flops_t(n_)) )
#define FLOPS_CSYMV(n_) (6 * FMULS_SYMV(flops_t(n_)) + 2 * FADDS_SYMV(flops_t(n_)) )
#define FLOPS_DSYMV(n_) (     FMULS_SYMV(flops_t(n_)) +       FADDS_SYMV(flops_t(n_)) )
#define FLOPS_SSYMV(n_) (     FMULS_SYMV(flops_t(n_)) +       FADDS_SYMV(flops_t(n_)) )

//
// Level 3 BLAS
//
#define FLOPS_ZGEMM(m_, n_, k_) (6 * FMULS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_CGEMM(m_, n_, k_) (6 * FMULS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_DGEMM(m_, n_, k_) (     FMULS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_SGEMM(m_, n_, k_) (     FMULS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_GEMM(flops_t(m_), flops_t(n_), flops_t(k_)) )

#define FLOPS_ZHEMM(side_, m_, n_) (6 * FMULS_HEMM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_HEMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_CHEMM(side_, m_, n_) (6 * FMULS_HEMM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_HEMM(side_, flops_t(m_), flops_t(n_)) )

#define FLOPS_ZSYMM(side_, m_, n_) (6 * FMULS_SYMM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_SYMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_CSYMM(side_, m_, n_) (6 * FMULS_SYMM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_SYMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_DSYMM(side_, m_, n_) (     FMULS_SYMM(side_, flops_t(m_), flops_t(n_)) +       FADDS_SYMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_SSYMM(side_, m_, n_) (     FMULS_SYMM(side_, flops_t(m_), flops_t(n_)) +       FADDS_SYMM(side_, flops_t(m_), flops_t(n_)) )

#define FLOPS_ZHERK(k_, n_) (6 * FMULS_HERK(flops_t(k_), flops_t(n_)) + 2 * FADDS_HERK(flops_t(k_), flops_t(n_)) )
#define FLOPS_CHERK(k_, n_) (6 * FMULS_HERK(flops_t(k_), flops_t(n_)) + 2 * FADDS_HERK(flops_t(k_), flops_t(n_)) )

#define FLOPS_ZSYRK(k_, n_) (6 * FMULS_SYRK(flops_t(k_), flops_t(n_)) + 2 * FADDS_SYRK(flops_t(k_), flops_t(n_)) )
#define FLOPS_CSYRK(k_, n_) (6 * FMULS_SYRK(flops_t(k_), flops_t(n_)) + 2 * FADDS_SYRK(flops_t(k_), flops_t(n_)) )
#define FLOPS_DSYRK(k_, n_) (     FMULS_SYRK(flops_t(k_), flops_t(n_)) +       FADDS_SYRK(flops_t(k_), flops_t(n_)) )
#define FLOPS_SSYRK(k_, n_) (     FMULS_SYRK(flops_t(k_), flops_t(n_)) +       FADDS_SYRK(flops_t(k_), flops_t(n_)) )

#define FLOPS_ZHER2K(k_, n_) (6 * FMULS_HER2K(flops_t(k_), flops_t(n_)) + 2 * FADDS_HER2K(flops_t(k_), flops_t(n_)) )
#define FLOPS_CHER2K(k_, n_) (6 * FMULS_HER2K(flops_t(k_), flops_t(n_)) + 2 * FADDS_HER2K(flops_t(k_), flops_t(n_)) )

#define FLOPS_ZSYR2K(k_, n_) (6 * FMULS_SYR2K(flops_t(k_), flops_t(n_)) + 2 * FADDS_SYR2K(flops_t(k_), flops_t(n_)) )
#define FLOPS_CSYR2K(k_, n_) (6 * FMULS_SYR2K(flops_t(k_), flops_t(n_)) + 2 * FADDS_SYR2K(flops_t(k_), flops_t(n_)) )
#define FLOPS_DSYR2K(k_, n_) (     FMULS_SYR2K(flops_t(k_), flops_t(n_)) +       FADDS_SYR2K(flops_t(k_), flops_t(n_)) )
#define FLOPS_SSYR2K(k_, n_) (     FMULS_SYR2K(flops_t(k_), flops_t(n_)) +       FADDS_SYR2K(flops_t(k_), flops_t(n_)) )

#define FLOPS_ZTRMM(side_, m_, n_) (6 * FMULS_TRMM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_TRMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_CTRMM(side_, m_, n_) (6 * FMULS_TRMM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_TRMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_DTRMM(side_, m_, n_) (     FMULS_TRMM(side_, flops_t(m_), flops_t(n_)) +       FADDS_TRMM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_STRMM(side_, m_, n_) (     FMULS_TRMM(side_, flops_t(m_), flops_t(n_)) +       FADDS_TRMM(side_, flops_t(m_), flops_t(n_)) )

#define FLOPS_ZTRSM(side_, m_, n_) (6 * FMULS_TRSM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_TRSM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_CTRSM(side_, m_, n_) (6 * FMULS_TRSM(side_, flops_t(m_), flops_t(n_)) + 2 * FADDS_TRSM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_DTRSM(side_, m_, n_) (     FMULS_TRSM(side_, flops_t(m_), flops_t(n_)) +       FADDS_TRSM(side_, flops_t(m_), flops_t(n_)) )
#define FLOPS_STRSM(side_, m_, n_) (     FMULS_TRSM(side_, flops_t(m_), flops_t(n_)) +       FADDS_TRSM(side_, flops_t(m_), flops_t(n_)) )

//
// Lapack
//
#define FLOPS_ZGETRF(m_, n_) (6 * FMULS_GETRF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GETRF(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGETRF(m_, n_) (6 * FMULS_GETRF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GETRF(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGETRF(m_, n_) (     FMULS_GETRF(flops_t(m_), flops_t(n_)) +       FADDS_GETRF(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGETRF(m_, n_) (     FMULS_GETRF(flops_t(m_), flops_t(n_)) +       FADDS_GETRF(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZGETRI(n_) (6 * FMULS_GETRI(flops_t(n_)) + 2 * FADDS_GETRI(flops_t(n_)) )
#define FLOPS_CGETRI(n_) (6 * FMULS_GETRI(flops_t(n_)) + 2 * FADDS_GETRI(flops_t(n_)) )
#define FLOPS_DGETRI(n_) (     FMULS_GETRI(flops_t(n_)) +       FADDS_GETRI(flops_t(n_)) )
#define FLOPS_SGETRI(n_) (     FMULS_GETRI(flops_t(n_)) +       FADDS_GETRI(flops_t(n_)) )

#define FLOPS_ZGETRS(n_, nrhs_) (6 * FMULS_GETRS(flops_t(n_), flops_t(nrhs_)) + 2 * FADDS_GETRS(flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_CGETRS(n_, nrhs_) (6 * FMULS_GETRS(flops_t(n_), flops_t(nrhs_)) + 2 * FADDS_GETRS(flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_DGETRS(n_, nrhs_) (     FMULS_GETRS(flops_t(n_), flops_t(nrhs_)) +       FADDS_GETRS(flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_SGETRS(n_, nrhs_) (     FMULS_GETRS(flops_t(n_), flops_t(nrhs_)) +       FADDS_GETRS(flops_t(n_), flops_t(nrhs_)) )

#define FLOPS_ZPOTRF(n_) (6 * FMULS_POTRF(flops_t(n_)) + 2 * FADDS_POTRF(flops_t(n_)) )
#define FLOPS_CPOTRF(n_) (6 * FMULS_POTRF(flops_t(n_)) + 2 * FADDS_POTRF(flops_t(n_)) )
#define FLOPS_DPOTRF(n_) (     FMULS_POTRF(flops_t(n_)) +       FADDS_POTRF(flops_t(n_)) )
#define FLOPS_SPOTRF(n_) (     FMULS_POTRF(flops_t(n_)) +       FADDS_POTRF(flops_t(n_)) )

#define FLOPS_ZPOTRI(n_) (6 * FMULS_POTRI(flops_t(n_)) + 2 * FADDS_POTRI(flops_t(n_)) )
#define FLOPS_CPOTRI(n_) (6 * FMULS_POTRI(flops_t(n_)) + 2 * FADDS_POTRI(flops_t(n_)) )
#define FLOPS_DPOTRI(n_) (     FMULS_POTRI(flops_t(n_)) +       FADDS_POTRI(flops_t(n_)) )
#define FLOPS_SPOTRI(n_) (     FMULS_POTRI(flops_t(n_)) +       FADDS_POTRI(flops_t(n_)) )

#define FLOPS_ZPOTRS(n_, nrhs_) (6 * FMULS_POTRS(flops_t(n_), flops_t(nrhs_)) + 2 * FADDS_POTRS(flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_CPOTRS(n_, nrhs_) (6 * FMULS_POTRS(flops_t(n_), flops_t(nrhs_)) + 2 * FADDS_POTRS(flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_DPOTRS(n_, nrhs_) (     FMULS_POTRS(flops_t(n_), flops_t(nrhs_)) +       FADDS_POTRS(flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_SPOTRS(n_, nrhs_) (     FMULS_POTRS(flops_t(n_), flops_t(nrhs_)) +       FADDS_POTRS(flops_t(n_), flops_t(nrhs_)) )

#define FLOPS_ZGEQRF(m_, n_) (6 * FMULS_GEQRF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEQRF(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGEQRF(m_, n_) (6 * FMULS_GEQRF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEQRF(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGEQRF(m_, n_) (     FMULS_GEQRF(flops_t(m_), flops_t(n_)) +       FADDS_GEQRF(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGEQRF(m_, n_) (     FMULS_GEQRF(flops_t(m_), flops_t(n_)) +       FADDS_GEQRF(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZGEQRT(m_, n_) (6 * FMULS_GEQRT(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEQRT(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGEQRT(m_, n_) (6 * FMULS_GEQRT(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEQRT(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGEQRT(m_, n_) (     FMULS_GEQRT(flops_t(m_), flops_t(n_)) +       FADDS_GEQRT(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGEQRT(m_, n_) (     FMULS_GEQRT(flops_t(m_), flops_t(n_)) +       FADDS_GEQRT(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZGEQLF(m_, n_) (6 * FMULS_GEQLF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEQLF(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGEQLF(m_, n_) (6 * FMULS_GEQLF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEQLF(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGEQLF(m_, n_) (     FMULS_GEQLF(flops_t(m_), flops_t(n_)) +       FADDS_GEQLF(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGEQLF(m_, n_) (     FMULS_GEQLF(flops_t(m_), flops_t(n_)) +       FADDS_GEQLF(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZGERQF(m_, n_) (6 * FMULS_GERQF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GERQF(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGERQF(m_, n_) (6 * FMULS_GERQF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GERQF(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGERQF(m_, n_) (     FMULS_GERQF(flops_t(m_), flops_t(n_)) +       FADDS_GERQF(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGERQF(m_, n_) (     FMULS_GERQF(flops_t(m_), flops_t(n_)) +       FADDS_GERQF(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZGELQF(m_, n_) (6 * FMULS_GELQF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GELQF(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGELQF(m_, n_) (6 * FMULS_GELQF(flops_t(m_), flops_t(n_)) + 2 * FADDS_GELQF(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGELQF(m_, n_) (     FMULS_GELQF(flops_t(m_), flops_t(n_)) +       FADDS_GELQF(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGELQF(m_, n_) (     FMULS_GELQF(flops_t(m_), flops_t(n_)) +       FADDS_GELQF(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZUNGQR(m_, n_, k_) (6 * FMULS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_CUNGQR(m_, n_, k_) (6 * FMULS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_DORGQR(m_, n_, k_) (     FMULS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_SORGQR(m_, n_, k_) (     FMULS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGQR(flops_t(m_), flops_t(n_), flops_t(k_)) )

#define FLOPS_ZUNGQL(m_, n_, k_) (6 * FMULS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_CUNGQL(m_, n_, k_) (6 * FMULS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_DORGQL(m_, n_, k_) (     FMULS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_SORGQL(m_, n_, k_) (     FMULS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGQL(flops_t(m_), flops_t(n_), flops_t(k_)) )

#define FLOPS_ZUNGRQ(m_, n_, k_) (6 * FMULS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_CUNGRQ(m_, n_, k_) (6 * FMULS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_DORGRQ(m_, n_, k_) (     FMULS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_SORGRQ(m_, n_, k_) (     FMULS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGRQ(flops_t(m_), flops_t(n_), flops_t(k_)) )

#define FLOPS_ZUNGLQ(m_, n_, k_) (6 * FMULS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_CUNGLQ(m_, n_, k_) (6 * FMULS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) + 2 * FADDS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_DORGLQ(m_, n_, k_) (     FMULS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) )
#define FLOPS_SORGLQ(m_, n_, k_) (     FMULS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) +       FADDS_UNGLQ(flops_t(m_), flops_t(n_), flops_t(k_)) )

#define FLOPS_ZUNMQR(m_, n_, k_, side_) (6 * FMULS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_CUNMQR(m_, n_, k_, side_) (6 * FMULS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_DORMQR(m_, n_, k_, side_) (     FMULS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_SORMQR(m_, n_, k_, side_) (     FMULS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMQR(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )

#define FLOPS_ZUNMQL(m_, n_, k_, side_) (6 * FMULS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_CUNMQL(m_, n_, k_, side_) (6 * FMULS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_DORMQL(m_, n_, k_, side_) (     FMULS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_SORMQL(m_, n_, k_, side_) (     FMULS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMQL(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )

#define FLOPS_ZUNMRQ(m_, n_, k_, side_) (6 * FMULS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_CUNMRQ(m_, n_, k_, side_) (6 * FMULS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_DORMRQ(m_, n_, k_, side_) (     FMULS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_SORMRQ(m_, n_, k_, side_) (     FMULS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMRQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )

#define FLOPS_ZUNMLQ(m_, n_, k_, side_) (6 * FMULS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_CUNMLQ(m_, n_, k_, side_) (6 * FMULS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) + 2 * FADDS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_DORMLQ(m_, n_, k_, side_) (     FMULS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )
#define FLOPS_SORMLQ(m_, n_, k_, side_) (     FMULS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) +       FADDS_UNMLQ(flops_t(m_), flops_t(n_), flops_t(k_), (side_)) )

#define FLOPS_ZGEQRS(m_, n_, nrhs_) (6 * FMULS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) + 2 * FADDS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_CGEQRS(m_, n_, nrhs_) (6 * FMULS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) + 2 * FADDS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_DGEQRS(m_, n_, nrhs_) (     FMULS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) +       FADDS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) )
#define FLOPS_SGEQRS(m_, n_, nrhs_) (     FMULS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) +       FADDS_GEQRS(flops_t(m_), flops_t(n_), flops_t(nrhs_)) )

#define FLOPS_ZTRTRI(n_) (6 * FMULS_TRTRI(flops_t(n_)) + 2 * FADDS_TRTRI(flops_t(n_)) )
#define FLOPS_CTRTRI(n_) (6 * FMULS_TRTRI(flops_t(n_)) + 2 * FADDS_TRTRI(flops_t(n_)) )
#define FLOPS_DTRTRI(n_) (     FMULS_TRTRI(flops_t(n_)) +       FADDS_TRTRI(flops_t(n_)) )
#define FLOPS_STRTRI(n_) (     FMULS_TRTRI(flops_t(n_)) +       FADDS_TRTRI(flops_t(n_)) )

#define FLOPS_ZGEHRD(n_) (6 * FMULS_GEHRD(flops_t(n_)) + 2 * FADDS_GEHRD(flops_t(n_)) )
#define FLOPS_CGEHRD(n_) (6 * FMULS_GEHRD(flops_t(n_)) + 2 * FADDS_GEHRD(flops_t(n_)) )
#define FLOPS_DGEHRD(n_) (     FMULS_GEHRD(flops_t(n_)) +       FADDS_GEHRD(flops_t(n_)) )
#define FLOPS_SGEHRD(n_) (     FMULS_GEHRD(flops_t(n_)) +       FADDS_GEHRD(flops_t(n_)) )

#define FLOPS_ZHETRD(n_) (6 * FMULS_HETRD(flops_t(n_)) + 2 * FADDS_HETRD(flops_t(n_)) )
#define FLOPS_CHETRD(n_) (6 * FMULS_HETRD(flops_t(n_)) + 2 * FADDS_HETRD(flops_t(n_)) )

#define FLOPS_ZSYTRD(n_) (6 * FMULS_SYTRD(flops_t(n_)) + 2 * FADDS_SYTRD(flops_t(n_)) )
#define FLOPS_CSYTRD(n_) (6 * FMULS_SYTRD(flops_t(n_)) + 2 * FADDS_SYTRD(flops_t(n_)) )
#define FLOPS_DSYTRD(n_) (     FMULS_SYTRD(flops_t(n_)) +       FADDS_SYTRD(flops_t(n_)) )
#define FLOPS_SSYTRD(n_) (     FMULS_SYTRD(flops_t(n_)) +       FADDS_SYTRD(flops_t(n_)) )

#define FLOPS_ZGEBRD(m_, n_) (6 * FMULS_GEBRD(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEBRD(flops_t(m_), flops_t(n_)) )
#define FLOPS_CGEBRD(m_, n_) (6 * FMULS_GEBRD(flops_t(m_), flops_t(n_)) + 2 * FADDS_GEBRD(flops_t(m_), flops_t(n_)) )
#define FLOPS_DGEBRD(m_, n_) (    FMULS_GEBRD(flops_t(m_), flops_t(n_)) +     FADDS_GEBRD(flops_t(m_), flops_t(n_)) )
#define FLOPS_SGEBRD(m_, n_) (    FMULS_GEBRD(flops_t(m_), flops_t(n_)) +     FADDS_GEBRD(flops_t(m_), flops_t(n_)) )

#define FLOPS_ZLARFG(n_) (6 * FMULS_LARFG(flops_t(n_)) + 2 * FADDS_LARFG(flops_t(n_)) )
#define FLOPS_CLARFG(n_) (6 * FMULS_LARFG(flops_t(n_)) + 2 * FADDS_LARFG(flops_t(n_)) )
#define FLOPS_DLARFG(n_) (    FMULS_LARFG(flops_t(n_)) +     FADDS_LARFG(flops_t(n_)) )
#define FLOPS_SLARFG(n_) (    FMULS_LARFG(flops_t(n_)) +     FADDS_LARFG(flops_t(n_)) )

#endif // if 0

#endif // HLR_UTILS_FLOPS_HH
