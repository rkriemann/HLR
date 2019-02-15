#ifndef __HLR_TLR_HH
#define __HLR_TLR_HH
//
// Project     : HLib
// File        : tlr.hh
// Description : TLR arithmetic functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <utility>

#include <hlib.hh>

#include "approx.hh"

namespace TLR
{

///////////////////////////////////////////////////////////////////////
//
// clustering
//
///////////////////////////////////////////////////////////////////////

std::pair< std::unique_ptr< HLIB::TClusterTree >,
           std::unique_ptr< HLIB::TBlockClusterTree > >
cluster ( HLIB::TCoordinate *  coords,
          const size_t         ntile );

///////////////////////////////////////////////////////////////////////
//
// arithmetic
//
///////////////////////////////////////////////////////////////////////

//
// solve X U = M
// - on input, X stores M
//
template < typename value_t >
void
trsmuh ( const HLIB::TDenseMatrix *  U,
         HLIB::TMatrix *             X )
{
    if ( HLIB::verbose( 4 ) )
        HLIB::DBG::printf( "trsmuh( %d, %d )", U->id(), X->id() );
    
    if ( HLIB::is_lowrank( X ) )
    {
        auto  RX = ptrcast( X, HLIB::TRkMatrix );
        auto  Y  = copy( HLIB::blas_mat_B< value_t >( RX ) );

        HLIB::BLAS::prod( value_t(1), HLIB::BLAS::adjoint( HLIB::blas_mat< value_t >( U ) ), Y, value_t(0), HLIB::blas_mat_B< value_t >( RX ) );
    }// else
    else if ( is_dense( X ) )
    {
        auto  DX = ptrcast( X, HLIB::TDenseMatrix );
        auto  Y  = copy( HLIB::blas_mat< value_t >( DX ) );
    
        HLIB::BLAS::prod( value_t(1), Y, HLIB::blas_mat< value_t >( U ), value_t(0), HLIB::blas_mat< value_t >( DX ) );
    }// else
}

}// namespace TLR

#endif // __HLR_TLR_HH
