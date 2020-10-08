#include <iostream>
#include <complex>

#include <cuComplex.h>

namespace hlr { namespace blas { namespace cuda {

//
// type trait for providing real valued type forming base of T
//
template <typename T>   struct real_type                            { using  type_t = T; };
template <>             struct real_type< cuFloatComplex >          { using  type_t = float; };
template <>             struct real_type< cuDoubleComplex >         { using  type_t = double; };

namespace
{

//
// compute Q = I - α·Θ⊗M with Θ_ij = 1 / ( m_ii - m_jj )
//
template < typename value_t >
__global__ void
hmul_theta_gpu ( const int        nrows,
                 const int        ncols,
                 const typename real_type< value_t >::type_t  alpha,
                 const value_t *  diag_M,
                 const value_t *  M,
                 value_t *        Q )
{
    const int  col = blockIdx.x * blockDim.x + threadIdx.x;
    const int  row = blockIdx.y * blockDim.y + threadIdx.y;
    const int  idx = col * nrows + row;

    // due to grid/block layout, more threads are used than entries in matrix
    if (( row < nrows ) && ( col < ncols ))
    {
        if ( col == row )
            Q[ idx ] = value_t(1);
        else
            Q[ idx ] = - alpha * M[ idx ] / ( diag_M[row] - diag_M[col] );
    }// if
}

template <>
__global__ void
hmul_theta_gpu< cuFloatComplex > ( const int               nrows,
                                   const int               ncols,
                                   const float             alpha,
                                   const cuFloatComplex *  diag_M,
                                   const cuFloatComplex *  M,
                                   cuFloatComplex *        Q )
{
    const int  col = blockIdx.x * blockDim.x + threadIdx.x;
    const int  row = blockIdx.y * blockDim.y + threadIdx.y;
    const int  idx = col * nrows + row;

    // due to grid/block layout, more threads are used than entries in matrix
    if (( row < nrows ) && ( col < ncols ))
    {
        if ( col == row )
            Q[ idx ] = make_cuFloatComplex( 1, 0 );
        else
        {
            Q[ idx ] = cuCmulf( cuCmulf( make_cuFloatComplex( -alpha, 0 ), M[ idx ] ),
                                cuCsubf( diag_M[row], diag_M[col] ) );
        }// else
    }// if
}

template <>
__global__ void
hmul_theta_gpu< cuDoubleComplex > ( const int                nrows,
                                    const int                ncols,
                                    const double             alpha,
                                    const cuDoubleComplex *  diag_M,
                                    const cuDoubleComplex *  M,
                                    cuDoubleComplex *        Q )
{
    const int  col = blockIdx.x * blockDim.x + threadIdx.x;
    const int  row = blockIdx.y * blockDim.y + threadIdx.y;
    const int  idx = col * nrows + row;

    // due to grid/block layout, more threads are used than entries in matrix
    if (( row < nrows ) && ( col < ncols ))
    {
        if ( col == row )
            Q[ idx ] = make_cuDoubleComplex( 1, 0 );
        else
        {
            Q[ idx ] = cuCmul( cuCmul( make_cuDoubleComplex( -alpha, 0 ), M[ idx ] ),
                               cuCsub( diag_M[row], diag_M[col] ) );
        }// else
    }// if
}

template < typename value_t >
void
hmul_theta_wrapper ( const int                                    nrows,
                     const int                                    ncols,
                     const typename real_type< value_t >::type_t  alpha,
                     const value_t *                              diag_M,
                     const value_t *                              M,
                     value_t *                                    Q )
{
    
    auto  block = dim3( std::min( 16, nrows ), std::min( 16, ncols ) );
    auto  nrg   = ( nrows % 16 == 0 ? nrows / 16 : nrows / 16 + 1 );
    auto  ncg   = ( ncols % 16 == 0 ? ncols / 16 : ncols / 16 + 1 );
    auto  grid  = dim3( nrg, ncg );
        
    hmul_theta_gpu<<< grid, block, 0 >>>( nrows, ncols, alpha, diag_M, M, Q );
}

}// namespace detail

//
// template instantiation did not work correctly, so
// define standard functions
//
#define HMUL_THETA( type )                                          \
    void                                                            \
    hmul_theta ( const int      nrows,                              \
                 const int      ncols,                              \
                 const typename real_type< type >::type_t  alpha,   \
                 const type *   diag_M,                             \
                 const type *   M,                                  \
                 type *         Q )                                 \
    {                                                               \
        hmul_theta_wrapper( nrows, ncols, alpha, diag_M, M, Q );    \
    }

HMUL_THETA( float )
HMUL_THETA( double )
HMUL_THETA( cuFloatComplex )
HMUL_THETA( cuDoubleComplex )

}}}// namespace hlr::blas::cuda
