#ifndef __HLR_MATRIX_SUM_HH
#define __HLR_MATRIX_SUM_HH
//
// Project     : HLR
// File        : sum.hh
// Description : provides operator for matrix sums
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/matrix/TMatrixSum.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

//
// functions to return matrix sum objects
//

inline 
std::unique_ptr< hpro::TMatrixSum< hpro::real > >
sum ( const hpro::TLinearOperator *  A0,
      const hpro::TLinearOperator *  A1,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< hpro::real > >( hpro::real(1), apply_normal, A0,
                                                               hpro::real(1), apply_normal, A1,
                                                               is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const hpro::TLinearOperator *  A0,
      const value_t                  alpha1,
      const hpro::TLinearOperator *  A1,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, apply_normal, A0,
                                                            alpha1, apply_normal, A1,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const matop_t                  op0,
      const hpro::TLinearOperator *  A0,
      const value_t                  alpha1,
      const matop_t                  op1,
      const hpro::TLinearOperator *  A1,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, op0, A0,
                                                            alpha1, op1, A1,
                                                            is_owner );
}
    

inline 
std::unique_ptr< hpro::TMatrixSum< hpro::real > >
sum ( const hpro::TLinearOperator *  A0,
      const hpro::TLinearOperator *  A1,
      const hpro::TLinearOperator *  A2,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< hpro::real > >( hpro::real(1), apply_normal, A0,
                                                               hpro::real(1), apply_normal, A1,
                                                               hpro::real(1), apply_normal, A2,
                                                               is_owner );
}
    
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const hpro::TLinearOperator *  A0,
      const value_t                  alpha1,
      const hpro::TLinearOperator *  A1,
      const value_t                  alpha2,
      const hpro::TLinearOperator *  A2,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, apply_normal, A0,
                                                            alpha1, apply_normal, A1,
                                                            alpha2, apply_normal, A2,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const matop_t                  op0,
      const hpro::TLinearOperator *  A0,
      const value_t                  alpha1,
      const matop_t                  op1,
      const hpro::TLinearOperator *  A1,
      const value_t                  alpha2,
      const matop_t                  op2,
      const hpro::TLinearOperator *  A2,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, op0, A0,
                                                            alpha1, op1, A1,
                                                            alpha2, op2, A2,
                                                            is_owner );
}

inline 
std::unique_ptr< hpro::TMatrixSum< hpro::real > >
sum ( const hpro::TLinearOperator &  A0,
      const hpro::TLinearOperator &  A1,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< hpro::real > >( hpro::real(1), apply_normal, & A0,
                                                               hpro::real(1), apply_normal, & A1,
                                                               is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const hpro::TLinearOperator &  A0,
      const value_t                  alpha1,
      const hpro::TLinearOperator &  A1,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, apply_normal, & A0,
                                                            alpha1, apply_normal, & A1,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const matop_t                  op0,
      const hpro::TLinearOperator &  A0,
      const value_t                  alpha1,
      const matop_t                  op1,
      const hpro::TLinearOperator &  A1,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, op0, & A0,
                                                            alpha1, op1, & A1,
                                                            is_owner );
}
    

inline 
std::unique_ptr< hpro::TMatrixSum< hpro::real > >
sum ( const hpro::TLinearOperator &  A0,
      const hpro::TLinearOperator &  A1,
      const hpro::TLinearOperator &  A2,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< hpro::real > >( hpro::real(1), apply_normal, & A0,
                                                               hpro::real(1), apply_normal, & A1,
                                                               hpro::real(1), apply_normal, & A2,
                                                               is_owner );
}
    
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const hpro::TLinearOperator &  A0,
      const value_t                  alpha1,
      const hpro::TLinearOperator &  A1,
      const value_t                  alpha2,
      const hpro::TLinearOperator &  A2,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, apply_normal, & A0,
                                                            alpha1, apply_normal, & A1,
                                                            alpha2, apply_normal, & A2,
                                                            is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixSum< value_t > >
sum ( const value_t                  alpha0,
      const matop_t                  op0,
      const hpro::TLinearOperator &  A0,
      const value_t                  alpha1,
      const matop_t                  op1,
      const hpro::TLinearOperator &  A1,
      const value_t                  alpha2,
      const matop_t                  op2,
      const hpro::TLinearOperator &  A2,
      const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixSum< value_t > >( alpha0, op0, & A0,
                                                            alpha1, op1, & A1,
                                                            alpha2, op2, & A2,
                                                            is_owner );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_SUM_HH
