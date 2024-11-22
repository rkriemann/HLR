#ifndef __HLR_MATRIX_LINOP_HH
#define __HLR_MATRIX_LINOP_HH
//
// Project     : HLR
// Module      : matrix/linop
// Description : further definitions for linear operators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/matrix/TLinearOperator.hh>
#include <hpro/vector/TVector.hh>

#include <hlr/arith/operator_wrapper.hh>
#include <hlr/utils/traits.hh>

namespace hlr {

//
// concept for types having apply functions
//
template < typename T >
concept has_apply_func = requires ( const T &                                        linop,
                                    const blas::vector< Hpro::value_type_t< T > > &  x,
                                    blas::vector< Hpro::value_type_t< T > > &        y,
                                    const Hpro::matop_t                              matop )
{
    { apply( linop, x, y, matop ) };
};
    
//
// concept for linear operators
//
template < typename T >
concept linear_operator_type = requires
{
    requires has_value_type< T >;
    requires has_apply_func< T >;
};

}// namespace hlr

#endif // __HLR_MATRIX_LINOP_HH
