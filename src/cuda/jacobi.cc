#include <cuda.h>
#include <cuda_bf16.h>

namespace hlr { namespace blas { namespace cuda {

//
// signum function for bfloat16
//
__device__
__nv_bfloat16
sign ( const __nv_bfloat16  f )
{
    return ( f >= __nv_bfloat16( float(0) ) ? __nv_bfloat16( float(1) ) : __nv_bfloat16( float(-1) ) );
}

//
// Jacobi eigenvalue computation for bfloat16 format
//
__global__
void
jacobi ( const int        n,
         __nv_bfloat16 *  M,
         __nv_bfloat16 *  V,
         __nv_bfloat16 *  E )
{
    for ( int  i = 0; i < n; i++ )
        V[ i*n + i ] = __nv_bfloat16( float(1) );

    float  eps        = float(1e-7);
    float  tol        = float(1e-4);
    int    max_sweeps = n;
    int    sweep      = 0;
    bool   converged  = false;

    while ( ! converged && ( sweep < max_sweeps ))
    {
        __nv_bfloat16  max_err = 0.0;
        
        sweep++;
        converged = true;
                
        for ( int  i = 0; i < n-1; i++ )
        {
            for ( int j = i + 1; j < n; j++ )
            {
                //
                // compute Jacobi rotation diagonalizing ⎧ M_ii  M_ij ⎫
                //                                       ⎩ M_ji  M_jj ⎭
                //

                const auto  c = M[ j*n + i ];

                if ( abs( float(c) ) <= eps )
                    continue;

                const auto  a   = M[ i*n + i ];
                const auto  b   = M[ j*n + i ];
                const auto  err = abs( float(c) ) / sqrt( abs( float( a*b )) );
                
                if (  err > tol )
                    converged = false;

                max_err = max( err, max_err );
                
                //
                // compute Jacobi rotation which diagonalises │a c│
                //                                            │c b│
                //

                const float          xi = (b - a) / ( __nv_bfloat16( float(2) ) * c );
                const __nv_bfloat16  t  = __nv_bfloat16( float( sign( xi ) ) / float( abs( xi ) + sqrt( float(1) + xi*xi ) ) );
                const __nv_bfloat16  cs = __nv_bfloat16( float(1) / sqrt( float(1) + float(t*t) ) );
                const __nv_bfloat16  sn = cs * t;

                M[ i*n + i ] = a - c * t;
                M[ j+n + j ] = b + c * t;
                M[ j*n + i ] = float(0);
                M[ i*n + j ] = float(0);
                
                //
                // update columns i and j of A (apply rotation)
                //

                for ( int  k = 0; k < n; k++ )
                {
                    if (( k == i ) || ( k == j ))
                        continue;
                    
                    const __nv_bfloat16  m_ik = M[ k*n + i ];
                    const __nv_bfloat16  m_jk = M[ k*n + j ];

                    M[ i*n + k ] = M[ k*n + i ] = cs * m_ik - sn * m_jk;
                    M[ j*n + k ] = M[ k*n + j ] = sn * m_ik + cs * m_jk;
                }// for
                
                //
                // update V (apply rotation)
                //
                
                for ( int  k = 0; k < n; k++ )
                {
                    const __nv_bfloat16  v_ki = V[ i*n + k ];
                    const __nv_bfloat16  v_kj = V[ j*n + k ];
                    
                    M[ i*n + k ] = cs * v_ki - sn * v_kj;
                    M[ j*n + k ] = sn * v_ki + cs * v_kj;
                }// for
            }// for
        }// for
    }// while

    //
    // extract eigenvalues as diagonal entries of M
    //
    
    for ( int  i = 0; i < n; i++ )
        E[ i ] = M[ i*n + i ];
}
         
void
jacobi_bf16 ( const int        n,
              __nv_bfloat16 *  M,
              __nv_bfloat16 *  V,
              __nv_bfloat16 *  E )
{
    jacobi<<< 1, 1, 0 >>>( n, M, V, E );
}

}}}// namespace hlr::blas::cuda
