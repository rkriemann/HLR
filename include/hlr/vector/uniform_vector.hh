#ifndef __HLR_VECTOR_UNIFORM_VECTOR_HH
#define __HLR_VECTOR_UNIFORM_VECTOR_HH
//
// Project     : HLR
// Module      : vector/uniform_vector
// Description : vector using given cluster bases
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <type_traits>

#include <hpro/vector/TVector.hh>

#include <hlr/utils/checks.hh>
#include <hlr/arith/blas.hh>

namespace hlr { namespace vector {

namespace hpro = HLIB;

// local vector type
DECLARE_TYPE( uniform_vector );

//
// Represents a vector with associated basis, e.g.,
//
//     x = V·s
//
// with basis V and coefficients s.
//
template < typename T_clusterbasis >
class uniform_vector : public hpro::TVector
{
public:
    //
    // export local types
    //

    using  cluster_basis_t = T_clusterbasis;
    using  value_t         = typename cluster_basis_t::value_t;

private:
    // local index set of vector
    indexset                 _is;
    
    // associated cluster basis
    const cluster_basis_t *  _basis;

    // coefficients within basis
    blas::vector< value_t >  _coeffs;
    
public:
    //
    // ctors
    //

    uniform_vector ()
            : TVector( 0, hpro::value_type< value_t >::value )
            , _is( 0, 0 )
            , _basis( nullptr )
    {}
    
    uniform_vector ( const indexset &  ais )
            : TVector( ais.first(), hpro::value_type< value_t >::value )
            , _is( ais )
            , _basis( nullptr )
    {}

    uniform_vector ( const indexset           ais,
                     const cluster_basis_t &  acb )
            : TVector( ais.first(), hpro::value_type< value_t >::value )
            , _is( ais )
            , _basis( &acb )
    {}

    uniform_vector ( const indexset              ais,
                     const cluster_basis_t &     acb,
                     blas::vector< value_t > &&  acoeff )
            : TVector( ais.first(), hpro::value_type< value_t >::value )
            , _is( ais )
            , _basis( &acb )
            , _coeffs( std::move( acoeff ) )
    {
        HLR_ASSERT( ! _coeffs.length() == _basis->rank() );
    }


    // dtor
    virtual ~uniform_vector ()
    {}
    
    //
    // access internal data
    //

    const tile< value_t > &          basis  () const { return *_basis; }
    
    blas::vector< value_t > &        coeffs ()       { return _coeffs; }
    const blas::vector< value_t > &  coeffs () const { return _coeffs; }
    
    //
    // vector data
    //
    
    virtual size_t  size  () const { return _is.size(); }
    
    ////////////////////////////////////////////////
    //
    // BLAS functionality (real valued)
    //

    // fill with constant
    virtual void fill ( const real  a )
    {
        assert( false );
    }

    // fill with random numbers
    virtual void fill_rand ( const uint  /* seed */ )
    {
        HLR_ASSERT( false );
    }

    // scale vector by constant factor
    virtual void scale ( const real  alpha )
    {
        blas::scale( value_t(alpha), _coeffs );
    }

    // this ≔ a · vector
    virtual void assign ( const real             /* a */,
                          const hpro::TVector *  /* v */ )
    {
        HLR_ASSERT( false );
    }

    // copy operator for all vectors
    hpro::TVector &
    operator = ( const hpro::TVector &  v )
    {
        assign( real(1), & v );
        return *this;
    }
    
    // return euclidean norm
    virtual real norm2 () const
    {
        // assuming orthonormal basis
        return blas::norm2( _coeffs );
    }

    // return infimum norm
    virtual real norm_inf () const
    {
        HLR_ASSERT( false );
        return real(0);
    }
    
    // this ≔ this + α·x
    virtual void axpy ( const real             alpha,
                        const hpro::TVector *  x )
    {
        if ( ! IS_TYPE( x, uniform_vector ) )
            blas::add( value_t(alpha), cptrcast( x, uniform_vector )->coeffs(), _coeffs );
        else
            HLR_ASSERT( false );
    }
    
    ////////////////////////////////////////////////
    //
    // BLAS functionality (complex valued)
    //

    // conjugate entries
    virtual void conjugate ()
    {
        // assuming cluster basis was modified accordingly
        blas::conj( _coeffs );
    }
        
    // fill with constant
    virtual void    cfill   ( const complex & )       { HLR_ASSERT( false ); }

    // scale vector by constant factor
    virtual void    cscale  ( const complex & )       { HLR_ASSERT( false ); }

    // this ≔ f · vector
    virtual void    cassign ( const complex &,
                              const TVector * )       { HLR_ASSERT( false ); }

    // return dot-product, <x,y> = x^H · y, where x = this
    virtual complex dot     ( const TVector * ) const { HLR_ASSERT( false ); return complex(0); }

    // return dot-product, <x,y> = x^T · y, where x = this
    virtual complex dotu    ( const TVector * ) const { HLR_ASSERT( false ); return complex(0); }

    // this ≔ this + α·x
    virtual void    caxpy   ( const complex &,
                              const TVector * )       { HLR_ASSERT( false ); } 

    //
    // RTTI
    //

    HLIB_RTTI_DERIVED( uniform_vector, TVector )

    //
    // virtual constructor
    //

    // return vector of same class (but no content)
    virtual
    std::unique_ptr< hpro::TVector >
    create () const
    {
        return std::make_unique< uniform_vector >();
    }

    // return copy of vector
    virtual
    std::unique_ptr< hpro::TVector >
    copy () const
    {
        return std::make_unique< uniform_vector >( _is, *_basis, std::move( blas::copy( _coeffs ) ) );
    }

    // copy vector data to A
    virtual
    void
    copy_to ( hpro::TVector *  v ) const
    {
        if ( ! IS_TYPE( v, uniform_vector ) )
        {
            auto  u = ptrcast( v, uniform_vector );

            u->_is     = _is;
            u->_basis  = _basis;
            u->_coeffs = std::move( blas::copy( _coeffs ) );
        }// if
        else
            HLR_ASSERT( false );
    }
    
    //
    // misc.
    //

    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        return ( TVector::byte_size() +
                 sizeof(_is) + sizeof(_basis) +
                 sizeof(_coeffs) + sizeof(value_t) * _coeffs.length() );
    }

protected:
    //
    // change field type 
    //
    
    virtual void  to_real     () { HLR_ASSERT( false ); }
    virtual void  to_complex  () { HLR_ASSERT( false ); }
};

}}// namespace hlr::vector

#endif // __HLR_VECTOR_UNIFORM_VECTOR_HH
