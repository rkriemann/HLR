//
// Project     : HLR
// File        : lduinv_eval.cc
// Description : evaluation operator for the inverse of LDU factorizations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

#include <hlr/arith/solve.hh>
#include <hlr/matrix/lduinv_eval.hh>

namespace hlr { namespace matrix {

//
// ctor
//

//
// linear operator mapping
//

//
// mapping function of linear operator A, e.g. y ≔ A(x).
// Depending on \a op, either A, A^T or A^H is applied.
//
template < typename value_t >
void
lduinv_eval< value_t >::apply  ( const Hpro::TVector< value_t > *  x,
                                 Hpro::TVector< value_t > *        y,
                                 const matop_t                     op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );
    HLR_ASSERT( is_scalar_all( x, y ) );
    
    x->copy_to( y );

    if ( op == apply_normal )
    {
        hlr::solve_lower_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), unit_diag );
        hlr::solve_diag(      op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), general_diag );
        hlr::solve_upper_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), unit_diag );
    }// if
    else
    {
        hlr::solve_upper_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), unit_diag );
        hlr::solve_diag(      op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), general_diag );
        hlr::solve_lower_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), unit_diag );
    }// else
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
template < typename value_t >
void
lduinv_eval< value_t >::apply_add  ( const value_t                     alpha,
                                     const Hpro::TVector< value_t > *  x,
                                     Hpro::TVector< value_t > *        y,
                                     const matop_t                     op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

    Hpro::TScalarVector< value_t >  t;

    apply( x, & t, op );
    y->axpy( alpha, & t );
}

template < typename value_t >
void
lduinv_eval< value_t >::apply_add  ( const value_t                     /* alpha */,
                                     const Hpro::TMatrix< value_t > *  /* X */,
                                     Hpro::TMatrix< value_t > *        /* Y */,
                                     const matop_t                     /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

//
// same as above but only the dimension of the vector spaces is tested,
// not the corresponding index sets
//
template < typename value_t >
void
lduinv_eval< value_t >::apply_add   ( const value_t                    alpha,
                                      const blas::vector< value_t > &  x,
                                      blas::vector< value_t > &        y,
                                      const matop_t                    op ) const
{
    Hpro::TScalarVector< value_t >  sx( _mat.row_is(), x );
    Hpro::TScalarVector< value_t >  sy( _mat.row_is(), y );
    
    apply_add( alpha, & sx, & sy, op );
}

template < typename value_t >
void
lduinv_eval< value_t >::apply_add   ( const value_t                    /* alpha */,
                                      const blas::matrix< value_t > &  /* x */,
                                      blas::matrix< value_t > &        /* y */,
                                      const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

}} // namespace hlr::matrix
