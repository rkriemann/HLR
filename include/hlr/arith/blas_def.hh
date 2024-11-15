#ifndef __HLR_ARITH_BLAS_DEF_HH
#define __HLR_ARITH_BLAS_DEF_HH
//
// Project     : HLR
// Module      : arith/blas_def
// Description : definition of various BLAS/LAPACK functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/blas/lapack.hh>

namespace hlr { namespace blas {

using Hpro::blas_int_t;

extern "C"
{
float
slange_ ( const char *        norm,
          const blas_int_t *  nrows,
          const blas_int_t *  ncols,
          const float *       M,
          const blas_int_t *  ldM,
          float *             work );
double
dlange_ ( const char *        norm,
          const blas_int_t *  nrows,
          const blas_int_t *  ncols,
          const double *      M,
          const blas_int_t *  ldM,
          double *            work );
double
clange_ ( const char *                   norm,
          const blas_int_t *             nrows,
          const blas_int_t *             ncols,
          const std::complex< float > *  M,
          const blas_int_t *             ldM,
          float *                        work );
double
zlange_ ( const char *                    norm,
          const blas_int_t *              nrows,
          const blas_int_t *              ncols,
          const std::complex< double > *  M,
          const blas_int_t *              ldM,
          double *                        work );
}


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

#undef HLR_BLAS_LARF


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
void sgeqrt_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nb,
               float *                   A,
               const blas_int_t *        ldA,
               float *                   T,
               const blas_int_t *        ldT,
               float *                   work,
               const blas_int_t *        info );
void dgeqrt_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nb,
               double *                  A,
               const blas_int_t *        ldA,
               double *                  T,
               const blas_int_t *        ldT,
               double *                  work,
               const blas_int_t *        info );
void cgeqrt_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nb,
               std::complex< float > *   A,
               const blas_int_t *        ldA,
               std::complex< float > *   T,
               const blas_int_t *        ldT,
               std::complex< float > *   work,
               const blas_int_t *        info );
void zgeqrt_ ( const blas_int_t *        nrows,
               const blas_int_t *        ncols,
               const blas_int_t *        nb,
               std::complex< double > *  A,
               const blas_int_t *        ldA,
               std::complex< double > *  T,
               const blas_int_t *        ldT,
               std::complex< double > *  work,
               const blas_int_t *        info );
}

#define HLR_BLAS_GEQRT( type, func )                    \
    inline                                              \
    void geqrt ( const blas_int_t  nrows,               \
                 const blas_int_t  ncols,               \
                 const blas_int_t  nb,                  \
                 type *            A,                   \
                 const blas_int_t  ldA,                 \
                 type *            T,                   \
                 const blas_int_t  ldT,                 \
                 type *            work,                \
                 blas_int_t &      info )               \
    {                                                   \
        func( & nrows, & ncols, & nb, A, & ldA, T, & ldT, work, & info ); \
    }

HLR_BLAS_GEQRT( float,                  sgeqrt_ )
HLR_BLAS_GEQRT( double,                 dgeqrt_ )
HLR_BLAS_GEQRT( std::complex< float >,  cgeqrt_ )
HLR_BLAS_GEQRT( std::complex< double >, zgeqrt_ )

#undef HLR_BLAS_GEQRT


extern "C"
{
void slatsqr_ ( const blas_int_t *        nrows,
                const blas_int_t *        ncols,
                const blas_int_t *        nbrow,
                const blas_int_t *        nbcol,
                float *                   A,
                const blas_int_t *        ldA,
                float *                   T,
                const blas_int_t *        ldT,
                float *                   work,
                const blas_int_t *        lwork,
                const blas_int_t *        info );
void dlatsqr_ ( const blas_int_t *        nrows,
                const blas_int_t *        ncols,
                const blas_int_t *        nbrow,
                const blas_int_t *        nbcol,
                double *                  A,
                const blas_int_t *        ldA,
                double *                  T,
                const blas_int_t *        ldT,
                double *                  work,
                const blas_int_t *        lwork,
                const blas_int_t *        info );
void clatsqr_ ( const blas_int_t *        nrows,
                const blas_int_t *        ncols,
                const blas_int_t *        nbrow,
                const blas_int_t *        nbcol,
                std::complex< float > *   A,
                const blas_int_t *        ldA,
                std::complex< float > *   T,
                const blas_int_t *        ldT,
                std::complex< float > *   work,
                const blas_int_t *        lwork,
                const blas_int_t *        info );
void zlatsqr_ ( const blas_int_t *        nrows,
                const blas_int_t *        ncols,
                const blas_int_t *        nbrow,
                const blas_int_t *        nbcol,
                std::complex< double > *  A,
                const blas_int_t *        ldA,
                std::complex< double > *  T,
                const blas_int_t *        ldT,
                std::complex< double > *  work,
                const blas_int_t *        lwork,
                const blas_int_t *        info );
}

#define HLR_BLAS_LATSQR( type, func )                   \
    inline                                              \
    void latsqr ( const blas_int_t  nrows,              \
                  const blas_int_t  ncols,              \
                  const blas_int_t  nbrow,              \
                  const blas_int_t  nbcol,              \
                  type *            A,                  \
                  const blas_int_t  ldA,                \
                  type *            T,                  \
                  const blas_int_t  ldT,                \
                  type *            work,               \
                  const blas_int_t  lwork,              \
                  blas_int_t &      info )              \
    {                                                   \
        func( & nrows, & ncols, & nbrow, & nbcol,       \
              A, & ldA, T, & ldT, work, & lwork, & info );  \
    }

HLR_BLAS_LATSQR( float,                  slatsqr_ )
HLR_BLAS_LATSQR( double,                 dlatsqr_ )
HLR_BLAS_LATSQR( std::complex< float >,  clatsqr_ )
HLR_BLAS_LATSQR( std::complex< double >, zlatsqr_ )

#undef HLR_BLAS_LATSQR


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
void sorg2r_ ( const blas_int_t *             nrows,
               const blas_int_t *             ncols,
               const blas_int_t *             nref,
               float *                        A,
               const blas_int_t *             lda,
               const float *                  tau,
               float *                        work,
               const blas_int_t *             info );
void dorg2r_ ( const blas_int_t *             nrows,
               const blas_int_t *             ncols,
               const blas_int_t *             nref,
               double *                       A,
               const blas_int_t *             lda,
               const double *                 tau,
               double *                       work,
               const blas_int_t *             info );
void cung2r_ ( const blas_int_t *             nrows,
               const blas_int_t *             ncols,
               const blas_int_t *             nref,
               std::complex< float > *        A,
               const blas_int_t *             lda,
               const std::complex< float > *  tau,
               std::complex< float > *        work,
               const blas_int_t *             info );
void zung2r_ ( const blas_int_t *             nrows,
               const blas_int_t *             ncols,
               const blas_int_t *             nref,
               std::complex< double > *       A,
               const blas_int_t *             lda,
               const std::complex< double > * tau,
               std::complex< double > *       work,
               const blas_int_t *             info );
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


extern "C"
{
void sorgtsqr_ ( const blas_int_t *        nrows,
                 const blas_int_t *        ncols,
                 const blas_int_t *        nbrow,
                 const blas_int_t *        nbcol,
                 float *                   A,
                 const blas_int_t *        ldA,
                 const float *             T,
                 const blas_int_t *        ldT,
                 float *                   work,
                 const blas_int_t *        lwork,
                 const blas_int_t *        info );
void dorgtsqr_ ( const blas_int_t *        nrows,
                 const blas_int_t *        ncols,
                 const blas_int_t *        nbrow,
                 const blas_int_t *        nbcol,
                 double *                  A,
                 const blas_int_t *        ldA,
                 const double *            T,
                 const blas_int_t *        ldT,
                 double *                  work,
                 const blas_int_t *        lwork,
                 const blas_int_t *        info );
void cungtsqr_ ( const blas_int_t *        nrows,
                 const blas_int_t *        ncols,
                 const blas_int_t *        nbrow,
                 const blas_int_t *        nbcol,
                 std::complex< float > *   A,
                 const blas_int_t *        ldA,
                 const std::complex< float > * T,
                 const blas_int_t *        ldT,
                 std::complex< float > *   work,
                 const blas_int_t *        lwork,
                 const blas_int_t *        info );
void zungtsqr_ ( const blas_int_t *        nrows,
                 const blas_int_t *        ncols,
                 const blas_int_t *        nbrow,
                 const blas_int_t *        nbcol,
                 std::complex< double > *  A,
                 const blas_int_t *        ldA,
                 const std::complex< double > * T,
                 const blas_int_t *        ldT,
                 std::complex< double > *  work,
                 const blas_int_t *        lwork,
                 const blas_int_t *        info );
}

#define HLR_BLAS_UNGTSQR( type, func )                  \
    inline                                              \
    void ungtsqr ( const blas_int_t  nrows,             \
                   const blas_int_t  ncols,             \
                   const blas_int_t  nbrow,             \
                   const blas_int_t  nbcol,             \
                   type *            A,                 \
                   const blas_int_t  ldA,               \
                   const type *      T,                 \
                   const blas_int_t  ldT,               \
                   type *            work,              \
                   const blas_int_t  lwork,             \
                   blas_int_t &      info )             \
    {                                                   \
        func( & nrows, & ncols, & nbrow, & nbcol,       \
              A, & ldA, T, & ldT, work, & lwork, & info );  \
    }

HLR_BLAS_UNGTSQR( float,                  sorgtsqr_ )
HLR_BLAS_UNGTSQR( double,                 dorgtsqr_ )
HLR_BLAS_UNGTSQR( std::complex< float >,  cungtsqr_ )
HLR_BLAS_UNGTSQR( std::complex< double >, zungtsqr_ )

#undef HLR_BLAS_UNGTSQR



extern "C"
{
void slarfb_ ( const char *                    side,
               const char *                    trans,
               const char *                    direct,
               const char *                    storeV,
               const blas_int_t *              nrows,
               const blas_int_t *              ncols,
               const blas_int_t *              norder,
               const float *                   V,
               const blas_int_t *              ldV,
               const float *                   T,
               const blas_int_t *              ldT,
               float *                         C,
               const blas_int_t *              ldC,
               float *                         work,
               const blas_int_t *              ldwork );
void dlarfb_ ( const char *                    side,
               const char *                    trans,
               const char *                    direct,
               const char *                    storeV,
               const blas_int_t *              nrows,
               const blas_int_t *              ncols,
               const blas_int_t *              norder,
               const double *                  V,
               const blas_int_t *              ldV,
               const double *                  T,
               const blas_int_t *              ldT,
               double *                        C,
               const blas_int_t *              ldC,
               double *                        work,
               const blas_int_t *              ldwork );
void clarfb_ ( const char *                    side,
               const char *                    trans,
               const char *                    direct,
               const char *                    storeV,
               const blas_int_t *              nrows,
               const blas_int_t *              ncols,
               const blas_int_t *              norder,
               const std::complex< float > *   V,
               const blas_int_t *              ldV,
               const std::complex< float > *   T,
               const blas_int_t *              ldT,
               std::complex< float > *         C,
               const blas_int_t *              ldC,
               std::complex< float > *         work,
               const blas_int_t *              ldwork );
void zlarfb_ ( const char *                    side,
               const char *                    trans,
               const char *                    direct,
               const char *                    storeV,
               const blas_int_t *              nrows,
               const blas_int_t *              ncols,
               const blas_int_t *              norder,
               const std::complex< double > *  V,
               const blas_int_t *              ldV,
               const std::complex< double > *  T,
               const blas_int_t *              ldT,
               std::complex< double > *        C,
               const blas_int_t *              ldC,
               std::complex< double > *        work,
               const blas_int_t *              ldwork );
}

#define HLR_BLAS_LARFB( type, func )                    \
    inline                                              \
    void larfb ( const char        side,                \
                 const char        trans,               \
                 const char        direct,              \
                 const char        storeV,              \
                 const blas_int_t  nrows,               \
                 const blas_int_t  ncols,               \
                 const blas_int_t  norder,              \
                 const type *      V,                   \
                 const blas_int_t  ldV,                 \
                 const type *      T,                   \
                 const blas_int_t  ldT,                 \
                 type *            C,                   \
                 const blas_int_t  ldC,                 \
                 type *            work,                \
                 const blas_int_t  lwork )              \
    {                                                   \
        func( & side, & trans, & direct, & storeV,      \
              & nrows, & ncols, & norder, V, & ldV,     \
              T, & ldT, C, & ldC, work, & lwork );      \
    }

HLR_BLAS_LARFB( float,                  slarfb_ )
HLR_BLAS_LARFB( double,                 dlarfb_ )
HLR_BLAS_LARFB( std::complex< float >,  clarfb_ )
HLR_BLAS_LARFB( std::complex< double >, zlarfb_ )

#undef HLR_BLAS_LARFB


extern "C"
{
void sbdsvdx_ ( const char *        uplo,
                const char *        jobz,
                const char *        range,
                const blas_int_t *  n,
                const float *       D,
                const float *       E,
                const float *       vl,
                const float *       vu,
                const blas_int_t *  il,
                const blas_int_t *  iu,
                const blas_int_t *  ns,
                float *             S,
                float *             Z,
                const blas_int_t *  ldZ,
                float *             work,
                blas_int_t *        iwork,
                const blas_int_t *  info );
void dbdsvdx_ ( const char *        uplo,
                const char *        jobz,
                const char *        range,
                const blas_int_t *  n,
                const double *      D,
                const double *      E,
                const double *      vl,
                const double *      vu,
                const blas_int_t *  il,
                const blas_int_t *  iu,
                const blas_int_t *  ns,
                double *            S,
                double *            Z,
                const blas_int_t *  ldZ,
                double *            work,
                blas_int_t *        iwork,
                const blas_int_t *  info );
}

#define HLR_BLAS_BDSVD( type, func )                    \
    inline                                              \
    void bdsvd ( const char          uplo,              \
                 const char          jobz,              \
                 const char          range,             \
                 const blas_int_t    n,                 \
                 const type *        D,                 \
                 const type *        E,                 \
                 const type          vl,                \
                 const type          vu,                \
                 const blas_int_t    il,                \
                 const blas_int_t    iu,                \
                 const blas_int_t &  ns,                \
                 type *              S,                 \
                 type *              Z,                 \
                 const blas_int_t    ldZ,               \
                 type *              work,              \
                 blas_int_t *        iwork,             \
                 blas_int_t &        info )             \
    {                                                   \
        func( & uplo, & jobz, & range, & n, D, E,       \
              & vl, & vu, & il, & iu, & ns, S, Z,       \
              & ldZ, work, iwork, & info );             \
    }

HLR_BLAS_BDSVD( float,  sbdsvdx_ )
HLR_BLAS_BDSVD( double, dbdsvdx_ )

#undef HLR_BLAS_BDSVD


extern "C"
{
void sbdsqr_ ( const char *        uplo,
               const blas_int_t *  n,
               const blas_int_t *  ncvt,
               const blas_int_t *  nru,
               const blas_int_t *  ncc,
               float *             D,
               float *             E,
               float *             VT,
               const blas_int_t *  ldVT,
               float *             U,
               const blas_int_t *  ldU,
               float *             C,
               const blas_int_t *  ldC,
               float *             work,
               const blas_int_t *  info );
void dbdsqr_ ( const char *        uplo,
               const blas_int_t *  n,
               const blas_int_t *  ncvt,
               const blas_int_t *  nru,
               const blas_int_t *  ncc,
               double *            D,
               double *            E,
               double *            VT,
               const blas_int_t *  ldVT,
               double *            U,
               const blas_int_t *  ldU,
               double *            C,
               const blas_int_t *  ldC,
               double *            work,
               const blas_int_t *  info );
}

#define HLR_BLAS_BDSQR( type, func )                    \
    inline                                              \
    void bdsqr ( const char          uplo,              \
                 const blas_int_t    n,                 \
                 const blas_int_t    ncvt,              \
                 const blas_int_t    nru,               \
                 const blas_int_t    ncc,               \
                 type *              D,                 \
                 type *              E,                 \
                 type *              VT,                \
                 const blas_int_t    ldVT,              \
                 type *              U,                 \
                 const blas_int_t    ldU,               \
                 type *              C,                 \
                 const blas_int_t    ldC,               \
                 type *              work,              \
                 blas_int_t &        info )             \
    {                                                   \
        func( & uplo, & n, & ncvt, & nru, & ncc, D, E,  \
              VT, & ldVT, U, & ldU, C, & ldC, work,     \
              & info );                                 \
    }

HLR_BLAS_BDSQR( float,  sbdsqr_ )
HLR_BLAS_BDSQR( double, dbdsqr_ )

#undef HLR_BLAS_BDSQR


extern "C"
{
void
ssyev_   ( const char *         jobz,
           const char *         uplo,
           const blas_int_t *   n,
           float *              A,
           const blas_int_t *   lda,
           float *              w,
           float *              work,
           const blas_int_t *   lwork,
           blas_int_t *         info );
void
dsyev_   ( const char *         jobz,
           const char *         uplo,
           const blas_int_t *   n,
           double *             A,
           const blas_int_t *   lda,
           double *             w,
           double *             work,
           const blas_int_t *   lwork,
           blas_int_t *         info );
void
cheev_   ( const char *             jobz,
           const char *             uplo,
           const blas_int_t *       n,
           std::complex< float> *   A,
           const blas_int_t *       lda,
           float *                  w,
           std::complex< float> *   work,
           const blas_int_t *       lwork,
           float *                  rwork,
           blas_int_t *             info );
void
zheev_   ( const char *             jobz,
           const char *             uplo,
           const blas_int_t *       n,
           std::complex< double> *  A,
           const blas_int_t *       lda,
           double *                 w,
           std::complex< double> *  work,
           const blas_int_t *       lwork,
           double *                 rwork,
           blas_int_t *             info );
}

#define HLR_BLAS_HEEV( type, func )                     \
    inline                                              \
    void heev ( const char          jobz,               \
                const char          uplo,               \
                const blas_int_t    n,                  \
                type *              A,                  \
                const blas_int_t    ldA,                \
                Hpro::real_type< type >::type_t *  w,   \
                type *              work,               \
                const blas_int_t    lwork,              \
                Hpro::real_type< type >::type_t *,      \
                blas_int_t &        info )              \
    {                                                   \
        func( & jobz, & uplo, & n, A, & ldA, w, work,   \
              & lwork, & info  );                       \
    }

HLR_BLAS_HEEV( float,                  ssyev_ )
HLR_BLAS_HEEV( double,                 dsyev_ )

#undef HLR_BLAS_HEEV

#define HLR_BLAS_HEEV( type, func )                     \
    inline                                              \
    void heev ( const char          jobz,               \
                const char          uplo,               \
                const blas_int_t    n,                  \
                type *              A,                  \
                const blas_int_t    ldA,                \
                Hpro::real_type< type >::type_t *  w,   \
                type *              work,               \
                const blas_int_t    lwork,              \
                Hpro::real_type< type >::type_t *  rwork,   \
                blas_int_t &        info )              \
    {                                                   \
        func( & jobz, & uplo, & n, A, & ldA, w, work,   \
              & lwork, rwork, & info  );                \
    }


HLR_BLAS_HEEV( std::complex< float >,  cheev_ )
HLR_BLAS_HEEV( std::complex< double >, zheev_ )

#undef HLR_BLAS_HEEV

}}// hlr::blas

#endif // __HLR_ARITH_BLAS_DEF_HH
