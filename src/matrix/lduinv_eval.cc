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

using namespace HLIB;

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
void
lduinv_eval::apply  ( const TVector *  x,
                      TVector *        y,
                      const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );
    HLR_ASSERT( is_scalar_all( x, y ) );
    
    x->copy_to( y );

    if ( op == apply_normal )
    {
        hlr::solve_lower_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), unit_diag );
        hlr::solve_diag(      op, _mat, * ptrcast( y, hpro::TScalarVector ), general_diag );
        hlr::solve_upper_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), unit_diag );
    }// if
    else
    {
        hlr::solve_upper_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), unit_diag );
        hlr::solve_diag(      op, _mat, * ptrcast( y, hpro::TScalarVector ), general_diag );
        hlr::solve_lower_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), unit_diag );
    }// else
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
lduinv_eval::apply_add  ( const real       alpha,
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
lduinv_eval::capply_add  ( const complex    alpha,
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
lduinv_eval::apply_add  ( const real       /* alpha */,
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
lduinv_eval::apply_add   ( const real                       alpha,
                           const blas::vector< real > &     x,
                           blas::vector< real > &           y,
                           const matop_t                    op ) const
{
    TScalarVector  sx( _mat.row_is(), x );
    TScalarVector  sy( _mat.row_is(), y );
    
    apply_add( alpha, & sx, & sy, op );
}

void
lduinv_eval::apply_add   ( const complex                    alpha,
                           const blas::vector< complex > &  x,
                           blas::vector< complex > &        y,
                           const matop_t                    op ) const
{
    TScalarVector  sx( _mat.row_is(), x );
    TScalarVector  sy( _mat.row_is(), y );
    
    capply_add( alpha, & sx, & sy, op );
}

void
lduinv_eval::apply_add   ( const real                       /* alpha */,
                           const blas::matrix< real > &     /* x */,
                           blas::matrix< real > &           /* y */,
                           const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

void
lduinv_eval::apply_add   ( const complex                    /* alpha */,
                           const blas::matrix< complex > &  /* x */,
                           blas::matrix< complex > &        /* y */,
                           const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

}} // namespace hlr::matrix
