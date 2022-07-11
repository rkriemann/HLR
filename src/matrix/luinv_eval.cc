//
// Project     : HLR
// File        : luinv_eval.cc
// Description : evaluation operator for the inverse of LU factorizations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

#include <hlr/arith/solve.hh>
#include <hlr/matrix/luinv_eval.hh>

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
luinv_eval< value_t >::apply  ( const Hpro::TVector< value_t > *  x,
                     Hpro::TVector< value_t > *        y,
                     const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );
    HLR_ASSERT( is_scalar_all( x, y ) );
    
    x->copy_to( y );

    if ( op == apply_normal )
    {
        // Hpro::solve_lower( op, &_mat, y, Hpro::solve_option_t( block_wise, unit_diag,    store_inverse ) );
        // Hpro::solve_upper( op, &_mat, y, Hpro::solve_option_t( block_wise, general_diag, store_inverse ) );

        hlr::solve_lower_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), unit_diag );
        hlr::solve_upper_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), general_diag );
    }// if
    else
    {
        // Hpro::solve_upper( op, &_mat, y, Hpro::solve_option_t( block_wise, general_diag, store_inverse ) );
        // Hpro::solve_lower( op, &_mat, y, Hpro::solve_option_t( block_wise, unit_diag,    store_inverse ) );

        hlr::solve_upper_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), general_diag );
        hlr::solve_lower_tri( op, _mat, * ptrcast( y, Hpro::TScalarVector< value_t > ), unit_diag );
    }// else
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
template < typename value_t >
void
luinv_eval< value_t >::apply_add  ( const value_t       alpha,
                         const Hpro::TVector< value_t > *  x,
                         Hpro::TVector< value_t > *        y,
                         const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

    Hpro::TScalarVector< value_t >  t;

    apply( x, & t, op );
    y->axpy( alpha, & t );
}

template < typename value_t >
void
luinv_eval< value_t >::apply_add  ( const value_t                     /* alpha */,
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
luinv_eval< value_t >::apply_add   ( const value_t                    alpha,
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
luinv_eval< value_t >::apply_add   ( const value_t                    /* alpha */,
                                     const blas::matrix< value_t > &  /* x */,
                                     blas::matrix< value_t > &        /* y */,
                                     const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

}} // namespace hlr::matrix
