#ifndef __HLR_MATRIX_LINOP_HH
#define __HLR_MATRIX_LINOP_HH
//
// Project     : HLR
// Module      : matrix/linop
// Description : further definitions for linear operators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TLinearOperator.hh>
#include <hpro/vector/TVector.hh>

namespace hlr {

//
// concept for types with value type
//
template < typename T > concept has_value_type = requires
{
    typename T::value_t;
};
    
//
// concept for types having apply functions
//
template < typename T > concept has_apply_func = 
    requires ( T  linop,
               const Hpro::TVector< Hpro::value_type_t< T > > *  x,
               Hpro::TVector< Hpro::value_type_t< T > > *        y,
               const Hpro::matop_t                               matop )
    {
        { linop.apply( x, y, matop ) };
    };
    
//
// concept for linear operators
//
template < typename T > concept is_linear_operator = requires
{
    requires has_value_type< T >;
    requires has_apply_func< T >;
};

}// namespace hlr

#endif // __HLR_MATRIX_LINOP_HH
