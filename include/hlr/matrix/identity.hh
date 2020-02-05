#ifndef __HLR_MATRIX_IDENTITY_HH
#define __HLR_MATRIX_IDENTITY_HH
//
// Project     : HLR
// File        : identity.hh
// Description : provides identity operator
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TLinearOperator.hh>
#include <hpro/vector/TScalarVector.hh>

#include <hlr/utils/checks.hh>

namespace hlr { namespace matrix {

namespace hpro = HLIB;
namespace blas = HLIB::BLAS;

// map HLIB types to HLR 
using  indexset       = hpro::TIndexSet;
using  block_indexset = hpro::TBlockIndexSet;

// local matrix type
DECLARE_TYPE( identity_operator );

//
// implements vector solving for LU using DAGs
//
class identity_operator : public hpro::TLinearOperator
{
private:
    // index set of identity
    indexset  _is;
    
public:
    //
    // ctor
    //

    identity_operator ( const indexset  is )
            : _is( is )
    {}
    
    //
    // linear operator properties
    //

    // return true, if field type is complex
    bool  is_complex      () const
    {
        return false;
    }
    
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
    virtual void  apply       ( const hpro::TVector *  x,
                                hpro::TVector *        y,
                                const hpro::matop_t    /* op */ = hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

        x->copy_to( y );
    }

    //
    // mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
    // Depending on \a op, either A, A^T or A^H is applied.
    //
    virtual void  apply_add   ( const hpro::real       alpha,
                                const hpro::TVector *  x,
                                hpro::TVector *        y,
                                const hpro::matop_t    /* op */ = hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

        y->axpy( alpha, x );
    }
    virtual void  capply_add  ( const hpro::complex    alpha,
                                const hpro::TVector *  x,
                                hpro::TVector *        y,
                                const hpro::matop_t    /* op */ = hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( x ) && ! is_null( y ) );

        y->caxpy( alpha, x );
    }

    virtual void  apply_add   ( const hpro::real       /* alpha */,
                                const hpro::TMatrix *  X,
                                hpro::TMatrix *        Y,
                                const hpro::matop_t    /* op */ = hpro::apply_normal ) const
    {
        HLR_ASSERT( ! is_null( X ) && ! is_null( Y ) );

        throw "TO BE DONE";
    }
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const hpro::real                       alpha,
                                const blas::Vector< hpro::real > &     x,
                                blas::Vector< hpro::real > &           y,
                                const hpro::matop_t                    /* op */ = hpro::apply_normal ) const
    {
        blas::add( alpha, x, y );
    }
    
    virtual void  apply_add   ( const hpro::complex                    alpha,
                                const blas::Vector< hpro::complex > &  x,
                                blas::Vector< hpro::complex > &        y,
                                const hpro::matop_t                    /* op */ = hpro::apply_normal ) const
    {
        blas::add( alpha, x, y );
    }

    //
    // access vector space data
    //

    // return dimension of domain
    virtual size_t  domain_dim     () const { return _is.size(); }
    
    // return dimension of range
    virtual size_t  range_dim      () const { return _is.size(); }
    
    // return vector in domain space
    virtual auto    domain_vector  () const -> std::unique_ptr< hpro::TVector > { return std::make_unique< hpro::TScalarVector >( _is ); }

    // return vector in range space
    virtual auto    range_vector   () const -> std::unique_ptr< hpro::TVector > { return std::make_unique< hpro::TScalarVector >( _is ); }

    //
    // misc.
    //

    // RTTI
    HLIB_RTTI_DERIVED( identity_operator, hpro::TLinearOperator )
};

//
// return operator representing identity
//
inline
std::unique_ptr< identity_operator >
identity ( const indexset &  is )
{
    return std::make_unique< identity_operator >( is );
}

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_IDENTITY_HH
