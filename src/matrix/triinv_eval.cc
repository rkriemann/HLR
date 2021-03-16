//
// Project     : HLR
// File        : triinv_eval.cc
// Description : evaluation operator for the inverse of triangular matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/algebra/solve_tri.hh>

#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/seq/dag.hh>

#include <hlr/matrix/triinv_eval.hh>

namespace hlr { namespace matrix {

using namespace HLIB;

//
// linear operator mapping
//

//
// mapping function of linear operator A, e.g. y ≔ A(x).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
triinv_eval::apply  ( const TVector *  x,
                      TVector *        y,
                      const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

    x->copy_to( y );

    if ( _shape == upper_triangular )
        hpro::solve_upper( op, &_mat, y, hpro::solve_option_t( block_wise, _diag, store_inverse ) );
    else
        hpro::solve_lower( op, &_mat, y, hpro::solve_option_t( block_wise, _diag, store_inverse ) );
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
triinv_eval::apply_add  ( const real       alpha,
                          const TVector *  x,
                          TVector *        y,
                          const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

    TScalarVector  t;

    apply( x, & t, op );
    y->axpy( alpha, & t );
}

void
triinv_eval::capply_add  ( const complex    alpha,
                           const TVector *  x,
                           TVector *        y,
                           const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

    TScalarVector  t;
    
    apply( x, & t, op );
    y->caxpy( alpha, & t );
}

void
triinv_eval::apply_add  ( const real       /* alpha */,
                          const TMatrix *  /* X */,
                          TMatrix *        /* Y */,
                          const matop_t    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

//
// same as above but only the dimension of the vector spaces is tested,
// not the corresponding index sets
//
void
triinv_eval::apply_add   ( const real                    alpha,
                           const blas::vector< real > &  x,
                           blas::vector< real > &        y,
                           const matop_t                 op ) const
{
    TScalarVector  sx( _mat.row_is(), x );
    TScalarVector  sy( _mat.row_is(), y );
    
    apply_add( alpha, & sx, & sy, op );
}

void
triinv_eval::apply_add   ( const complex                    alpha,
                           const blas::vector< complex > &  x,
                           blas::vector< complex > &        y,
                           const matop_t                    op ) const
{
    TScalarVector  sx( _mat.row_is(), x );
    TScalarVector  sy( _mat.row_is(), y );
    
    capply_add( alpha, & sx, & sy, op );
}

void
triinv_eval::apply_add   ( const real                       /* alpha */,
                           const blas::matrix< real > &     /* x */,
                           blas::matrix< real > &           /* y */,
                           const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

void
triinv_eval::apply_add   ( const complex                    /* alpha */,
                           const blas::matrix< complex > &  /* x */,
                           blas::matrix< complex > &        /* y */,
                           const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

}} // namespace hlr::matrix
