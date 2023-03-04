#ifndef __HLR_MATRIX_IDENTITY_HH
#define __HLR_MATRIX_IDENTITY_HH
//
// Project     : HLR
// Module      : identity.hh
// Description : provides identity operator
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TLinearOperator.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/checks.hh>

namespace hlr { namespace matrix {

// map HLIB types to HLR 
using  indexset       = Hpro::TIndexSet;
using  block_indexset = Hpro::TBlockIndexSet;

// local matrix type
DECLARE_TYPE( identity_operator );

//
// implements vector solving for LU using DAGs
//
template < typename T_value >
class identity_operator : public Hpro::TLinearOperator< T_value >
{
public:
    //
    // value type
    //

    using  value_t = T_value;
    
private:
    // index set of identity
    block_indexset  _bis;
    
public:
    //
    // ctor
    //

    identity_operator ( const block_indexset  bis )
            : _bis( bis )
    {
        HLR_ASSERT( bis.row_is() == bis.col_is() );
    }
    
    //
    // linear operator properties
    //

    // return true, of operator is self adjoint
    bool  is_self_adjoint () const
    {
        return true;
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
                                const Hpro::matop_t               /* op */ = Hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

        x->copy_to( y );
    }

    //
    // mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply_add   ( const value_t                     alpha,
                                const Hpro::TVector< value_t > *  x,
                                Hpro::TVector< value_t > *        y,
                                const Hpro::matop_t               /* op */ = Hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

        y->axpy( alpha, x );
    }

    virtual void  apply_add   ( const value_t                     /* alpha */,
                                const Hpro::TMatrix< value_t > *  X,
                                Hpro::TMatrix< value_t > *        Y,
                                const Hpro::matop_t               /* op */ = Hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( X ) && ! is_null( Y ) );
        
        HLR_ERROR( "TO BE DONE" );
    }
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const Hpro::matop_t              /* op */ = Hpro::apply_normal ) const
    {
        blas::add( alpha, x, y );
    }

    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::matrix< value_t > &  X,
                                blas::matrix< value_t > &        Y,
                                const Hpro::matop_t              /* op */ = Hpro::apply_normal ) const
    {
        blas::add( alpha, X, Y );
    }

    //
    // access vector space data
    //

    // return dimension of domain
    virtual size_t  domain_dim     () const { return _bis.col_is().size(); }
    
    // return dimension of range
    virtual size_t  range_dim      () const { return _bis.row_is().size(); }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< Hpro::TVector< value_t > >
    {
        return std::make_unique< Hpro::TScalarVector< value_t > >( _bis.col_is() );
    }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< Hpro::TVector< value_t > >
    {
        return std::make_unique< Hpro::TScalarVector< value_t > >( _bis.row_is() );
    }

    //
    // misc.
    //

    // RTTI
    HPRO_RTTI_DERIVED( identity_operator, Hpro::TLinearOperator< value_t > )
};

//
// return operator representing identity
//
template < typename value_t >
std::unique_ptr< identity_operator< value_t > >
identity ( const block_indexset &  bis )
{
    return std::make_unique< identity_operator< value_t > >( bis );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_IDENTITY_HH
