#ifndef __HLR_ARITH_INVERT_HH
#define __HLR_ARITH_INVERT_HH
//
// Project     : HLib
// File        : multiply.hh
// Description : matrix inversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TDenseMatrix.hh>

#include "hlr/arith/blas.hh"

namespace hlr
{

//
// in-place matrix inversion M ← M^-1
//
template < typename value_t >
void
invert ( hpro::TDenseMatrix &  M )
{
    blas::invert( hpro::blas_mat< value_t >( M ) );
}

//
// return inverse of M
//
template < typename value_t >
std::unique_ptr< hpro::TDenseMatrix >
inverse ( const hpro::TDenseMatrix &  M )
{
    auto  I = std::unique_ptr< hpro::TDenseMatrix >( ptrcast( M.copy().release(), hpro::TDenseMatrix ) );

    invert< value_t >( *I );

    return I;
}

}// namespace hlr

#endif // __HLR_ARITH_INVERT_HH
