#ifndef __HLR_MATRIX_PRODUCT_HH
#define __HLR_MATRIX_PRODUCT_HH
//
// Project     : HLR
// File        : product.hh
// Description : provides operator for matrix products
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/matrix/TMatrixProduct.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;

//
// functions to return matrix product objects
//

inline
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::TLinearOperator *  A0,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( hpro::real(1), A0, is_owner );
}
                 
inline                 
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::real               alpha0,
          const hpro::TLinearOperator *  A0,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( alpha0, A0, is_owner );
}

inline                 
std::unique_ptr< hpro::TMatrixProduct< complex > >
product ( const complex                  alpha0,
          const hpro::TLinearOperator *  A0,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< complex > >( alpha0, A0, is_owner );
}
                 
inline
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::TLinearOperator *  A0,
          const hpro::TLinearOperator *  A1,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( hpro::real(1), A0, hpro::real(1), A1, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
          const hpro::TLinearOperator *  A0,
          const value_t                  alpha1,
          const hpro::TLinearOperator *  A1,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, A0, alpha1, A1, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
          const matop_t                  op0,
          const hpro::TLinearOperator *  A0,
          const value_t                  alpha1,
          const matop_t                  op1,
          const hpro::TLinearOperator *  A1,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, op0, A0, alpha1, op1, A1, is_owner );
}
                 
inline
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::TLinearOperator *  A0,
          const hpro::TLinearOperator *  A1,
          const hpro::TLinearOperator *  A2,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( hpro::real(1), A0, hpro::real(1), A1, hpro::real(1), A2, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
          const hpro::TLinearOperator *  A0,
          const value_t                  alpha1,
          const hpro::TLinearOperator *  A1,
          const value_t                  alpha2,
          const hpro::TLinearOperator *  A2,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, A0, alpha1, A1, alpha2, A2, is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
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
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, op0, A0, alpha1, op1, A1, alpha2, op2, A2, is_owner );
}


inline
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::TLinearOperator &  A0,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( hpro::real(1), & A0, is_owner );
}
                 
inline                 
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::real               alpha0,
          const hpro::TLinearOperator &  A0,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( alpha0, & A0, is_owner );
}

inline                 
std::unique_ptr< hpro::TMatrixProduct< complex > >
product ( const complex                  alpha0,
          const hpro::TLinearOperator &  A0,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< complex > >( alpha0, & A0, is_owner );
}
                 
inline
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::TLinearOperator &  A0,
          const hpro::TLinearOperator &  A1,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( hpro::real(1), & A0,
                                                                   hpro::real(1), & A1,
                                                                   is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
          const hpro::TLinearOperator &  A0,
          const value_t                  alpha1,
          const hpro::TLinearOperator &  A1,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, & A0,
                                                                alpha1, & A1,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
          const matop_t                  op0,
          const hpro::TLinearOperator &  A0,
          const value_t                  alpha1,
          const matop_t                  op1,
          const hpro::TLinearOperator &  A1,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, op0, & A0,
                                                                alpha1, op1, & A1,
                                                                is_owner );
}
                 
inline
std::unique_ptr< hpro::TMatrixProduct< hpro::real > >
product ( const hpro::TLinearOperator &  A0,
          const hpro::TLinearOperator &  A1,
          const hpro::TLinearOperator &  A2,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< hpro::real > >( hpro::real(1), & A0,
                                                                   hpro::real(1), & A1,
                                                                   hpro::real(1), & A2,
                                                                   is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
          const hpro::TLinearOperator &  A0,
          const value_t                  alpha1,
          const hpro::TLinearOperator &  A1,
          const value_t                  alpha2,
          const hpro::TLinearOperator &  A2,
          const bool                     is_owner = false )
{
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, & A0,
                                                                alpha1, & A1,
                                                                alpha2, & A2,
                                                                is_owner );
}
                 
template < typename value_t >
std::unique_ptr< hpro::TMatrixProduct< value_t > >
product ( const value_t                  alpha0,
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
    return std::make_unique< hpro::TMatrixProduct< value_t > >( alpha0, op0, & A0,
                                                                alpha1, op1, & A1,
                                                                alpha2, op2, & A2,
                                                                is_owner );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_PRODUCT_HH
