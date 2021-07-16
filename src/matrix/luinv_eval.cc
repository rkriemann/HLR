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
luinv_eval::apply  ( const TVector *  x,
                     TVector *        y,
                     const matop_t    op ) const
{
    HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );
    HLR_ASSERT( is_scalar_all( x, y ) );
    
    x->copy_to( y );

    if ( op == apply_normal )
    {
        // hpro::solve_lower( op, &_mat, y, hpro::solve_option_t( block_wise, unit_diag,    store_inverse ) );
        // hpro::solve_upper( op, &_mat, y, hpro::solve_option_t( block_wise, general_diag, store_inverse ) );

        hlr::solve_lower_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), unit_diag );
        hlr::solve_upper_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), general_diag );
    }// if
    else
    {
        // hpro::solve_upper( op, &_mat, y, hpro::solve_option_t( block_wise, general_diag, store_inverse ) );
        // hpro::solve_lower( op, &_mat, y, hpro::solve_option_t( block_wise, unit_diag,    store_inverse ) );

        hlr::solve_upper_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), general_diag );
        hlr::solve_lower_tri( op, _mat, * ptrcast( y, hpro::TScalarVector ), unit_diag );
    }// else
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
luinv_eval::apply_add  ( const real       alpha,
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
luinv_eval::capply_add  ( const complex    alpha,
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
luinv_eval::apply_add  ( const real       /* alpha */,
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
luinv_eval::apply_add   ( const real                       alpha,
                          const blas::vector< real > &     x,
                          blas::vector< real > &           y,
                          const matop_t                    op ) const
{
    TScalarVector  sx( _mat.row_is(), x );
    TScalarVector  sy( _mat.row_is(), y );
    
    apply_add( alpha, & sx, & sy, op );
}

void
luinv_eval::apply_add   ( const complex                    alpha,
                          const blas::vector< complex > &  x,
                          blas::vector< complex > &        y,
                          const matop_t                    op ) const
{
    TScalarVector  sx( _mat.row_is(), x );
    TScalarVector  sy( _mat.row_is(), y );
    
    capply_add( alpha, & sx, & sy, op );
}

void
luinv_eval::apply_add   ( const real                       /* alpha */,
                          const blas::matrix< real > &     /* x */,
                          blas::matrix< real > &           /* y */,
                          const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

void
luinv_eval::apply_add   ( const complex                    /* alpha */,
                          const blas::matrix< complex > &  /* x */,
                          blas::matrix< complex > &        /* y */,
                          const matop_t                    /* op */ ) const
{
    HLR_ERROR( "not implemented" );
}

}} // namespace hlr::matrix
