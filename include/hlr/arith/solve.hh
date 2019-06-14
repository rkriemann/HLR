#ifndef __HLR_ARITH_SOLVE_HH
#define __HLR_ARITH_SOLVE_HH
//
// Project     : HLib
// File        : solve.hh
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/TMatrix.hh>
#include <matrix/TDenseMatrix.hh>
#include <matrix/TRkMatrix.hh>

namespace hlr
{

using namespace HLIB;

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const TDenseMatrix *  U,
         TMatrix *             X )
{
    if ( verbose( 4 ) )
        DBG::printf( "trsmuh( %d, %d )", U->id(), X->id() );
    
    if ( is_lowrank( X ) )
    {
        auto  RX = ptrcast( X, TRkMatrix );
        auto  Y  = copy( blas_mat_B< value_t >( RX ) );

        BLAS::prod( value_t(1), BLAS::adjoint( blas_mat< value_t >( U ) ), Y,
                    value_t(0), blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( X, TDenseMatrix );
        auto  Y  = copy( blas_mat< value_t >( DX ) );
    
        BLAS::prod( value_t(1), Y, blas_mat< value_t >( U ),
                    value_t(0), blas_mat< value_t >( DX ) );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_SOLVE_HH
