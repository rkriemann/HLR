#ifndef __HLR_ARITH_BLAS_DEF_HH
#define __HLR_ARITH_BLAS_DEF_HH
//
// Project     : HLR
// Module      : arith/blas_def
// Description : definition of various BLAS/LAPACK functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/blas/lapack.hh>

namespace hlr { namespace blas {

namespace hpro = HLIB;

using hpro::blas_int_t;

extern "C"
{
void slarfg_ ( const blas_int_t *      n,
               const float *           alpha,
               const float *           x,
               const blas_int_t *      incx,
               const float *           tau );
void dlarfg_ ( const blas_int_t *      n,
               const double *          alpha,
               const double *          x,
               const blas_int_t *      incx,
               const double *          tau );
void clarfg_ ( const blas_int_t *             n,
               const std::complex< float > *  alpha,
               const std::complex< float > *  x,
               const blas_int_t *             incx,
               const std::complex< float > *  tau );
void zlarfg_ ( const blas_int_t *             n,
               const std::complex< double > * alpha,
               const std::complex< double > * x,
               const blas_int_t *             incx,
               const std::complex< double > * tau );
}

#define HLR_BLAS_LARFG( type, func )            \
    inline                                      \
    void larfg ( const blas_int_t  n,           \
                 type &            alpha,       \
                 type *            x,           \
                 const blas_int_t  incx,        \
                 type &            tau )        \
    {                                           \
        func( & n, & alpha, x, & incx, & tau ); \
    }

HLR_BLAS_LARFG( float,             slarfg_ )
HLR_BLAS_LARFG( double,            dlarfg_ )
HLR_BLAS_LARFG( std::complex< float >,  clarfg_ )
HLR_BLAS_LARFG( std::complex< double >, zlarfg_ )

#undef HLR_BLAS_LARFG


extern "C"
{
void slarf_  ( const char *            side,
               const blas_int_t *      n,
               const blas_int_t *      m,
               const float *           V,
               const blas_int_t *      incv,
               const float *           tau,
               float *                 C,
               const blas_int_t *      ldc,
               const float *           work );
void dlarf_  ( const char *            side,
               const blas_int_t *      n,
               const blas_int_t *      m,
               const double *          V,
               const blas_int_t *      incv,
               const double *          tau,
               double *                C,
               const blas_int_t *      ldc,
               const double *          work );
void clarf_  ( const char *                    side,
               const blas_int_t *              n,
               const blas_int_t *              m,
               const std::complex< float > *   V,
               const blas_int_t *              incv,
               const std::complex< float > *   tau,
               std::complex< float > *         C,
               const blas_int_t *              ldc,
               const std::complex< float > *   work );
void zlarf_  ( const char *                    side,
               const blas_int_t *              n,
               const blas_int_t *              m,
               const std::complex< double > *  V,
               const blas_int_t *              incv,
               const std::complex< double > *  tau,
               std::complex< double > *        C,
               const blas_int_t *              ldc,
               const std::complex< double > *  work );
}

#define HLR_BLAS_LARF( type, func )                                 \
    inline                                                          \
    void larf ( const char        side,                             \
                const blas_int_t  n,                                \
                const blas_int_t  m,                                \
                const type *      V,                                \
                const blas_int_t  incv,                             \
                const type        tau,                              \
                type *            C,                                \
                const blas_int_t  ldc,                              \
                type *            work )                            \
    {                                                               \
        func( & side, & n, & m, V, & incv, & tau, C, & ldc, work ); \
    }

HLR_BLAS_LARF( float,                  slarf_ )
HLR_BLAS_LARF( double,                 dlarf_ )
HLR_BLAS_LARF( std::complex< float >,  clarf_ )
HLR_BLAS_LARF( std::complex< double >, zlarf_ )

#undef HLIB_LARF_FUNC


extern "C"
{
void sgeqr_ ( const blas_int_t *  n,
              const blas_int_t *  m,
              float *             A,
              const blas_int_t *  ldA,
              float *             T,
              const blas_int_t *  tsize,
              float *             work,
              const blas_int_t *  lwork,
              blas_int_t *        info );

void dgeqr_ ( const blas_int_t *  n,
              const blas_int_t *  m,
              double *            A,
              const blas_int_t *  ldA,
              double *            T,
              const blas_int_t *  tsize,
              double *            work,
              const blas_int_t *  lwork,
              blas_int_t *        info );

void cgeqr_ ( const blas_int_t *        n,
              const blas_int_t *        m,
              std::complex< float > *   A,
              const blas_int_t *        ldA,
              std::complex< float > *   T,
              const blas_int_t *        tsize,
              std::complex< float > *   work,
              const blas_int_t *        lwork,
              blas_int_t *              info );

void zgeqr_ ( const blas_int_t *        n,
              const blas_int_t *        m,
              std::complex< double > *  A,
              const blas_int_t *        ldA,
              std::complex< double > *  T,
              const blas_int_t *        tsize,
              std::complex< double > *  work,
              const blas_int_t *        lwork,
              blas_int_t *              info );
}

#define HLR_BLAS_GEQR( type, func )                                     \
    inline                                                              \
    void geqr ( const blas_int_t  n,                                    \
                const blas_int_t  m,                                    \
                type *            A,                                    \
                const blas_int_t  ldA,                                  \
                type *            T,                                    \
                const blas_int_t  tsize,                                \
                type *            work,                                 \
                const blas_int_t  lwork,                                \
                blas_int_t &      info )                                \
    {                                                                   \
        func( & n, & m, A, & ldA, T, & tsize, work, & lwork, & info );  \
    }

HLR_BLAS_GEQR( float,                  sgeqr_ )
HLR_BLAS_GEQR( double,                 dgeqr_ )
HLR_BLAS_GEQR( std::complex< float >,  cgeqr_ )
HLR_BLAS_GEQR( std::complex< double >, zgeqr_ )

#undef HLR_BLAS_GEQR


extern "C"
{
void sgeqr2_ ( const blas_int_t *  nrows,
               const blas_int_t *  ncols,
               float *             A,
               const blas_int_t *  lda,
               float *             tau,
               float *             work,
               const blas_int_t *  info );
void dgeqr2_ ( const blas_int_t *  nrows,
               const blas_int_t *  ncols,
               double *            A,
               const blas_int_t *  lda,
               double *            tau,
               double *            work,
               const blas_int_t *  info );
void cgeqr2_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               std::complex< float > *   A,
               const blas_int_t *        lda,
               std::complex< float > *   tau,
               std::complex< float > *   work,
               const blas_int_t *        info );
void zgeqr2_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               std::complex< double > *  A,
               const blas_int_t *        lda,
               std::complex< double > *  tau,
               std::complex< double > *  work,
               const blas_int_t *        info );
}

#define HLR_BLAS_GEQR2( type, func )                    \
    inline                                              \
    void geqr2 ( const blas_int_t  n,                   \
                 const blas_int_t  m,                   \
                 type *            A,                   \
                 const blas_int_t  ldA,                 \
                 type *            tau,                 \
                 type *            work,                \
                 blas_int_t &      info )               \
    {                                                   \
        func( & n, & m, A, & ldA, tau, work, & info );  \
    }

HLR_BLAS_GEQR2( float,                  sgeqr2_ )
HLR_BLAS_GEQR2( double,                 dgeqr2_ )
HLR_BLAS_GEQR2( std::complex< float >,  cgeqr2_ )
HLR_BLAS_GEQR2( std::complex< double >, zgeqr2_ )

#undef HLR_BLAS_GEQR2

extern "C"
{
void sgemqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const float *       A,
               const blas_int_t *  ldA,
               const float *       T,
               const blas_int_t *  tsize,
               float *             C,
               const blas_int_t *  ldC,
               float *             work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void dgemqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const double *      A,
               const blas_int_t *  ldA,
               const double *      T,
               const blas_int_t *  tsize,
               double *            C,
               const blas_int_t *  ldC,
               double *            work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void cgemqr_ ( const char *                   side,
               const char *                   trans,
               const blas_int_t *             n,
               const blas_int_t *             m,
               const blas_int_t *             k,
               const std::complex< float > *  A,
               const blas_int_t *             ldA,
               const std::complex< float > *  T,
               const blas_int_t *             tsize,
               std::complex< float > *        C,
               const blas_int_t *             ldC,
               std::complex< float > *        work,
               const blas_int_t *             lwork,
               blas_int_t *                   info );

void zgemqr_ ( const char *                   side,
               const char *                   trans,
               const blas_int_t *             n,
               const blas_int_t *             m,
               const blas_int_t *             k,
               const std::complex< double > * A,
               const blas_int_t *             ldA,
               const std::complex< double > * T,
               const blas_int_t *             tsize,
               std::complex< double > *       C,
               const blas_int_t *             ldC,
               std::complex< double > *       work,
               const blas_int_t *             lwork,
               blas_int_t *                   info );
}

#define HLR_BLAS_GEMQR( type, func )                                    \
    inline                                                              \
    void gemqr ( const char        side,                                \
                 const char        trans,                               \
                 const blas_int_t  n,                                   \
                 const blas_int_t  m,                                   \
                 const blas_int_t  k,                                   \
                 const type *      A,                                   \
                 const blas_int_t  ldA,                                 \
                 const type *      T,                                   \
                 const blas_int_t  tsize,                               \
                 type *            C,                                   \
                 const blas_int_t  ldC,                                 \
                 type *            work,                                \
                 const blas_int_t  lwork,                               \
                 blas_int_t &      info )                               \
    {                                                                   \
        func( & side, & trans, & n, & m, & k, A, & ldA, T, & tsize, C, & ldC, work, & lwork, & info ); \
    }

HLR_BLAS_GEMQR( float,                  sgemqr_ )
HLR_BLAS_GEMQR( double,                 dgemqr_ )
HLR_BLAS_GEMQR( std::complex< float >,  cgemqr_ )
HLR_BLAS_GEMQR( std::complex< double >, zgemqr_ )

#undef HLR_BLAS_GEMQR


extern "C"
{
void sormqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const float *       A,
               const blas_int_t *  ldA,
               const float *       tau,
               float *             C,
               const blas_int_t *  ldC,
               float *             work,
               const blas_int_t *  lwork,
               blas_int_t *        info );

void dormqr_ ( const char *        side,
               const char *        trans,
               const blas_int_t *  n,
               const blas_int_t *  m,
               const blas_int_t *  k,
               const double *      A,
               const blas_int_t *  ldA,
               const double *      tau,
               double *            C,
               const blas_int_t *  ldC,
               double *            work,
               const blas_int_t *  lwork,
               blas_int_t *        info );
void cunmqr_ ( const char *                    side,
               const char *                    trans,
               const blas_int_t *              n,
               const blas_int_t *              m,
               const blas_int_t *              k,
               const std::complex< float > *   A,
               const blas_int_t *              ldA,
               const std::complex< float > *   tau,
               std::complex< float > *         C,
               const blas_int_t *              ldC,
               std::complex< float > *         work,
               const blas_int_t *              lwork,
               blas_int_t *                    info );

void zunmqr_ ( const char *                    side,
               const char *                    trans,
               const blas_int_t *              n,
               const blas_int_t *              m,
               const blas_int_t *              k,
               const std::complex< double > *  A,
               const blas_int_t *              ldA,
               const std::complex< double > *  tau,
               std::complex< double > *        C,
               const blas_int_t *              ldC,
               std::complex< double > *        work,
               const blas_int_t *              lwork,
               blas_int_t *                    info );
}

#define HLR_BLAS_UNMQR( type, func )                                    \
    inline                                                              \
    void unmqr ( const char        side,                                \
                 const char        trans,                               \
                 const blas_int_t  n,                                   \
                 const blas_int_t  m,                                   \
                 const blas_int_t  k,                                   \
                 const type *      A,                                   \
                 const blas_int_t  ldA,                                 \
                 const type *      tau,                                 \
                 type *            C,                                   \
                 const blas_int_t  ldC,                                 \
                 type *            work,                                \
                 const blas_int_t  lwork,                               \
                 blas_int_t &      info )                               \
    {                                                                   \
        func( & side, & trans, & n, & m, & k, A, & ldA, tau, C, & ldC, work, & lwork, & info ); \
    }

HLR_BLAS_UNMQR( float,                  sormqr_ )
HLR_BLAS_UNMQR( double,                 dormqr_ )
HLR_BLAS_UNMQR( std::complex< float >,  cunmqr_ )
HLR_BLAS_UNMQR( std::complex< double >, zunmqr_ )

#undef HLR_BLAS_UNMQR


extern "C"
{
void sorg2r_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nref,
               float *                   A,
               const blas_int_t *        lda,
               float *                   tau,
               float *                   work,
               const blas_int_t *        info );
void dorg2r_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nref,
               double *                  A,
               const blas_int_t *        lda,
               double *                  tau,
               double *                  work,
               const blas_int_t *        info );
void cung2r_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nref,
               std::complex< float > *   A,
               const blas_int_t *        lda,
               std::complex< float > *   tau,
               std::complex< float > *   work,
               const blas_int_t *        info );
void zung2r_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nref,
               std::complex< double > *  A,
               const blas_int_t *        lda,
               std::complex< double > *  tau,
               std::complex< double > *  work,
               const blas_int_t *        info );
}

#define HLR_BLAS_UNG2R( type, func )                    \
    inline                                              \
    void ung2r ( const blas_int_t  n,                   \
                 const blas_int_t  m,                   \
                 const blas_int_t  k,                   \
                 type *            A,                   \
                 const blas_int_t  ldA,                 \
                 type *            tau,                 \
                 type *            work,                \
                 blas_int_t &      info )               \
    {                                                   \
        func( & n, & m, & k, A, & ldA, tau, work, & info );  \
    }

HLR_BLAS_UNG2R( float,                  sorg2r_ )
HLR_BLAS_UNG2R( double,                 dorg2r_ )
HLR_BLAS_UNG2R( std::complex< float >,  cung2r_ )
HLR_BLAS_UNG2R( std::complex< double >, zung2r_ )

#undef HLR_BLAS_UNG2R

}}// hlr::blas

#endif // __HLR_ARITH_BLAS_DEF_HH
