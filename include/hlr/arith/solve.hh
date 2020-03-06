#ifndef __HLR_ARITH_SOLVE_HH
#define __HLR_ARITH_SOLVE_HH
//
// Project     : HLib
// File        : solve.hh
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/utils/log.hh>

namespace hlr
{

namespace hpro = HLIB;

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const hpro::TDenseMatrix *  U,
         hpro::TMatrix *             X )
{
    HLR_LOG( 4, hpro::to_string( "trsmuh( %d, %d )", U->id(), X->id() ) );
    
    if ( is_lowrank( X ) )
    {
        auto  RX = ptrcast( X, hpro::TRkMatrix );
        auto  Y  = blas::copy( hpro::blas_mat_B< value_t >( RX ) );

        blas::prod( value_t(1), blas::adjoint( hpro::blas_mat< value_t >( U ) ), Y,
                    value_t(0), hpro::blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( X, hpro::TDenseMatrix );
        auto  Y  = copy( hpro::blas_mat< value_t >( DX ) );
    
        blas::prod( value_t(1), Y, hpro::blas_mat< value_t >( U ),
                    value_t(0), hpro::blas_mat< value_t >( DX ) );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_SOLVE_HH
