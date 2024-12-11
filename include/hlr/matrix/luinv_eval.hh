#ifndef __HLR_MATRIX_LUINV_EVAL_HH
#define __HLR_MATRIX_LUINV_EVAL_HH
//
// Project     : HLR
// Module      : luinv_eval.hh
// Description : evaluation operator for the inverse of LU factorizations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <memory>
#include <map>
#include <mutex>

#include <hpro/matrix/TLinearOperator.hh>

#include <hlr/arith/blas.hh>
#include <hlr/arith/solve.hh>

namespace hlr { namespace matrix {

// local matrix type
DECLARE_TYPE( luinv_eval );

//
// implements vector solving for LU factored matrices
//
template < typename T_value >
class luinv_eval : public Hpro::TLinearOperator< T_value >
{
public:
    using  value_t = T_value;
    
private:
    // matrix containing LU data
    const Hpro::TMatrix< value_t > &  _mat;

public:
    //
    // ctor
    //

    luinv_eval ( const Hpro::TMatrix< value_t > &  M )
            : _mat( M )
    {}
    
    //
    // linear operator properties
    //

    // return true, of operator is self adjoint
    bool  is_self_adjoint () const
    {
        return false;
    }
    
    //
    // linear operator mapping
    //

    //
    // mapping function of linear operator A, e.g. y ≔ A(x).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply       ( const Hpro::TVector< value_t > *  x,
                                Hpro::TVector< value_t > *        y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    //
    // mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply_add   ( const value_t                     alpha,
                                const Hpro::TVector< value_t > *  x,
                                Hpro::TVector< value_t > *        y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    virtual void  apply_add   ( const value_t                     alpha,
                                const Hpro::TMatrix< value_t > *  X,
                                Hpro::TMatrix< value_t > *        Y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                     alpha,
                                const blas::vector< value_t > &   x,
                                blas::vector< value_t > &         y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    virtual void  apply_add   ( const value_t                     alpha,
                                const blas::matrix< value_t > &   X,
                                blas::matrix< value_t > &         Y,
                                const Hpro::matop_t               op = Hpro::apply_normal ) const;

    // same as above but use given arithmetic object for computation
    template < typename arithmetic_t >
    void          apply_add   ( arithmetic_t &&                  arithmetic,
                                const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;

    //
    // access vector space data
    //

    // return dimension of domain
    virtual size_t  domain_dim     () const { return _mat.nrows(); }
    
    // return dimension of range
    virtual size_t  range_dim      () const { return _mat.ncols(); }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< Hpro::TVector< value_t > > { return _mat.row_vector(); }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< Hpro::TVector< value_t > > { return _mat.col_vector(); }

    //
    // misc.
    //

    // RTTI
    HPRO_RTTI_DERIVED( luinv_eval, Hpro::TLinearOperator< value_t > )
};

//
// mapping function of linear operator A, e.g. y ≔ A(x).
// Depending on \a op, either A, A^T or A^H is applied.
//
template < typename value_t >
void
luinv_eval< value_t >::apply  ( const Hpro::TVector< value_t > *  x,
                                Hpro::TVector< value_t > *        y,
                                const matop_t                     op ) const
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
luinv_eval< value_t >::apply_add  ( const value_t                     alpha,
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

template < typename value_t >
template < typename arithmetic_t >
void
luinv_eval< value_t >::apply_add  ( arithmetic_t &&                  arithmetic,
                                    const value_t                    alpha,
                                    const blas::vector< value_t > &  x,
                                    blas::vector< value_t > &        y,
                                    const matop_t                    op ) const
{
    Hpro::TScalarVector< value_t >  t( _mat.row_is() );
    
    blas::copy( x, blas::vec( t ) );

    if ( op == apply_normal )
    {
        arithmetic.solve_lower_tri( op, _mat, t, unit_diag );
        arithmetic.solve_upper_tri( op, _mat, t, general_diag );
    }// if
    else
    {
        arithmetic.solve_upper_tri( op, _mat, t, general_diag );
        arithmetic.solve_lower_tri( op, _mat, t, unit_diag );
    }// else

    blas::add( alpha, blas::vec( t ), y );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_LUINV_EVAL_HH
