#ifndef __HLR_MATRIX_SUM_HH
#define __HLR_MATRIX_SUM_HH
//
// Project     : HLR
// Module      : matrix/sum
// Description : provides operator for matrix sums
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/matrix/TMatrixSum.hh>

namespace hlr { namespace matrix {

//
// functions to return matrix sum objects
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const Hpro::TLinearOperator< value_t > *  A0,
      const Hpro::TLinearOperator< value_t > *  A1,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( value_t(1), apply_normal, A0,
                                                               value_t(1), apply_normal, A1,
                                                               is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const Hpro::TLinearOperator< value_t > *  A0,
      const value_t                             alpha1,
      const Hpro::TLinearOperator< value_t > *  A1,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, apply_normal, A0,
                                                            alpha1, apply_normal, A1,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const matop_t                             op0,
      const Hpro::TLinearOperator< value_t > *  A0,
      const value_t                             alpha1,
      const matop_t                             op1,
      const Hpro::TLinearOperator< value_t > *  A1,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, op0, A0,
                                                            alpha1, op1, A1,
                                                            is_owner );
}
    

template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const Hpro::TLinearOperator< value_t > *  A0,
      const Hpro::TLinearOperator< value_t > *  A1,
      const Hpro::TLinearOperator< value_t > *  A2,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( value_t(1), apply_normal, A0,
                                                               value_t(1), apply_normal, A1,
                                                               value_t(1), apply_normal, A2,
                                                               is_owner );
}
    
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const Hpro::TLinearOperator< value_t > *  A0,
      const value_t                             alpha1,
      const Hpro::TLinearOperator< value_t > *  A1,
      const value_t                             alpha2,
      const Hpro::TLinearOperator< value_t > *  A2,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, apply_normal, A0,
                                                            alpha1, apply_normal, A1,
                                                            alpha2, apply_normal, A2,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const matop_t                             op0,
      const Hpro::TLinearOperator< value_t > *  A0,
      const value_t                             alpha1,
      const matop_t                             op1,
      const Hpro::TLinearOperator< value_t > *  A1,
      const value_t                             alpha2,
      const matop_t                             op2,
      const Hpro::TLinearOperator< value_t > *  A2,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, op0, A0,
                                                            alpha1, op1, A1,
                                                            alpha2, op2, A2,
                                                            is_owner );
}

template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const Hpro::TLinearOperator< value_t > &  A0,
      const Hpro::TLinearOperator< value_t > &  A1,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( value_t(1), apply_normal, & A0,
                                                            value_t(1), apply_normal, & A1,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const Hpro::TLinearOperator< value_t > &  A1,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, apply_normal, & A0,
                                                            alpha1, apply_normal, & A1,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const matop_t                             op0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const matop_t                             op1,
      const Hpro::TLinearOperator< value_t > &  A1,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, op0, & A0,
                                                            alpha1, op1, & A1,
                                                            is_owner );
}
    

template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const Hpro::TLinearOperator< value_t > &  A0,
      const Hpro::TLinearOperator< value_t > &  A1,
      const Hpro::TLinearOperator< value_t > &  A2,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( value_t(1), apply_normal, & A0,
                                                            value_t(1), apply_normal, & A1,
                                                            value_t(1), apply_normal, & A2,
                                                            is_owner );
}
    
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const Hpro::TLinearOperator< value_t > &  A1,
      const value_t                             alpha2,
      const Hpro::TLinearOperator< value_t > &  A2,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, apply_normal, & A0,
                                                            alpha1, apply_normal, & A1,
                                                            alpha2, apply_normal, & A2,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< Hpro::TMatrixSum< value_t > >
sum ( const value_t                             alpha0,
      const matop_t                             op0,
      const Hpro::TLinearOperator< value_t > &  A0,
      const value_t                             alpha1,
      const matop_t                             op1,
      const Hpro::TLinearOperator< value_t > &  A1,
      const value_t                             alpha2,
      const matop_t                             op2,
      const Hpro::TLinearOperator< value_t > &  A2,
      const bool                                is_owner = false )
{
    return std::make_unique< Hpro::TMatrixSum< value_t > >( alpha0, op0, & A0,
                                                            alpha1, op1, & A1,
                                                            alpha2, op2, & A2,
                                                            is_owner );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_SUM_HH
