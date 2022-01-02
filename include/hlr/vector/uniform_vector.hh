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
#include <mutex>

#include <hpro/vector/TVector.hh>

#include <hlr/utils/checks.hh>
#include <hlr/arith/blas.hh>

namespace hlr { namespace vector {

namespace hpro = HLIB;

using indexset = hpro::TIndexSet;

// local vector type
DECLARE_TYPE( uniform_vector );

//
// Represents a vector with associated basis, e.g.,
//
//     x = V·s
//
// with basis V and coefficients s. Furthermore, it is structured
// with the same structure as the cluster basis.
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
    using  sub_block_t     = uniform_vector< cluster_basis_t >;

private:
    // local index set of vector
    indexset                      _is;
    
    // associated cluster basis
    const cluster_basis_t *       _basis;

    // coefficients within basis
    blas::vector< value_t >       _coeffs;
    
    // sub blocks
    std::vector< sub_block_t * >  _blocks;

    // for mutual exclusion
    std::mutex                    _mutex;
    
public:
    //
    // ctors
    //

    uniform_vector ()
            : TVector( 0, hpro::value_type_v< value_t > )
            , _is( 0, 0 )
            , _basis( nullptr )
    {}
    
    uniform_vector ( const indexset &  ais )
            : TVector( ais.first(), hpro::value_type_v< value_t > )
            , _is( ais )
            , _basis( nullptr )
    {}

    uniform_vector ( const indexset           ais,
                     const cluster_basis_t &  acb )
            : TVector( ais.first(), hpro::value_type_v< value_t > )
            , _is( ais )
            , _basis( &acb )
            , _coeffs( acb.rank() )
            , _blocks( acb.nsons() )
    {}

    uniform_vector ( const indexset              ais,
                     const cluster_basis_t &     acb,
                     blas::vector< value_t > &&  acoeff )
            : TVector( ais.first(), hpro::value_type_v< value_t > )
            , _is( ais )
            , _basis( &acb )
            , _coeffs( std::move( acoeff ) )
            , _blocks( acb.nsons() )
    {
        HLR_ASSERT( ! ( _coeffs.length() == _basis->rank() ) );
    }


    // dtor
    virtual ~uniform_vector ()
    {
        for ( auto  v : _blocks )
            delete v;
    }
    
    //
    // access basis and coefficients
    //

    const cluster_basis_t &          basis  () const { return *_basis; }
    
    blas::vector< value_t > &        coeffs ()       { return _coeffs; }
    const blas::vector< value_t > &  coeffs () const { return _coeffs; }

    void
    set_coeffs ( blas::vector< value_t > &&  acoeffs )
    {
        _coeffs = std::move( acoeffs );
    }
    
    //
    // access sub blocks
    //

    uint                 nblocks    () const                   { return _blocks.size(); }
    
    sub_block_t *        block      ( const uint     i )       { return _blocks[i]; }
    const sub_block_t *  block      ( const uint     i ) const { return _blocks[i]; }

    void                 set_block  ( const uint     i,
                                      sub_block_t *  v )
    {
        HLR_ASSERT( i < nblocks() );

        if (( _blocks[i] != nullptr ) && ( _blocks[i] != v ))
            delete _blocks[i];
        
        _blocks[i] = v;
    }
    
    //
    // access mutex
    //

    std::mutex &  mutex () { return _mutex; }
    
    //
    // general vector data
    //
    
    virtual size_t  size  () const { return _is.size(); }
    
    ////////////////////////////////////////////////
    //
    // BLAS functionality (real valued)
    //

    // fill with constant
    virtual void fill ( const hpro::real  a )
    {
        assert( false );

        for ( auto  v : _blocks )
            if ( v != nullptr )
                v->fill( a );
    }

    // fill with random numbers
    virtual void fill_rand ( const uint  seed )
    {
        HLR_ASSERT( false );

        for ( auto  v : _blocks )
            if ( v != nullptr )
                v->fill_rand( seed );
    }

    // scale vector by constant factor
    virtual void scale ( const hpro::real  alpha )
    {
        blas::scale( value_t(alpha), _coeffs );

        for ( auto  v : _blocks )
            if ( v != nullptr )
                v->scale( alpha );
    }

    // this ≔ a · vector
    virtual void assign ( const hpro::real       alpha,
                          const hpro::TVector *  v )
    {
        if ( IS_TYPE( v, uniform_vector ) )
        {
            auto  u = cptrcast( v, uniform_vector );

            _is     = u->_is;
            _basis  = u->_basis;
            _coeffs = std::move( blas::copy( u->_coeffs ) );

            if ( alpha != hpro::real(1) )
                blas::scale( value_t(alpha), _coeffs );
                
            HLR_ASSERT( nblocks() == u->nblocks() );

            for ( uint  i = 0; i < nblocks(); ++i )
            {
                if ( block(i) != nullptr )
                {
                    HLR_ASSERT( u->block(i) != nullptr );
                    
                    block(i)->assign( alpha, u->block(i) );
                }// if
            }// for
        }// if

        HLR_ASSERT( false );
    }

    // copy operator for all vectors
    hpro::TVector &
    operator = ( const hpro::TVector &  v )
    {
        assign( hpro::real(1), & v );
        return *this;
    }
    
    // return euclidean norm
    virtual hpro::real norm2 () const
    {
        // assuming orthonormal basis
        auto  square = [] ( const auto  f ) { return f*f; };
            
        // assuming orthonormal basis
        auto  val = square( blas::norm2( _coeffs ) );

        for ( auto  v : _blocks )
            val += square( v->norm2() );

        return hpro::Math::sqrt( val );
    }

    // return infimum norm
    virtual hpro::real norm_inf () const
    {
        HLR_ASSERT( false );
        return hpro::real(0);
    }
    
    // this ≔ this + α·x
    virtual void axpy ( const hpro::real       alpha,
                        const hpro::TVector *  v )
    {
        if ( ! IS_TYPE( v, uniform_vector ) )
        {
            auto  u = cptrcast( v, uniform_vector );
            
            blas::add( value_t(alpha), u->_coeffs, _coeffs );
            
            HLR_ASSERT( nblocks() == u->nblocks() );

            for ( uint  i = 0; i < nblocks(); ++i )
            {
                if ( block(i) != nullptr )
                {
                    HLR_ASSERT( u->block(i) != nullptr );
                    
                    block(i)->axpy( alpha, u->block(i) );
                }// if
            }// for
        }// if
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

        for ( auto  v : _blocks )
            v->conjugate();
    }
        
    // fill with constant
    virtual void    cfill   ( const hpro::complex & )       { HLR_ASSERT( false ); }

    // scale vector by constant factor
    virtual void    cscale  ( const hpro::complex & ) { HLR_ASSERT( false ); }

    // this ≔ f · vector
    virtual void    cassign ( const hpro::complex &,
                              const TVector * )       { HLR_ASSERT( false ); }

    // return dot-product, <x,y> = x^H · y, where x = this
    virtual hpro::complex dot     ( const TVector * ) const { HLR_ASSERT( false ); return hpro::complex(0); }

    // return dot-product, <x,y> = x^T · y, where x = this
    virtual hpro::complex dotu    ( const TVector * ) const { HLR_ASSERT( false ); return hpro::complex(0); }

    // this ≔ this + α·x
    virtual void    caxpy   ( const hpro::complex &,
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

            HLR_ASSERT( _blocks.size() == u->_blocks.size() );

            for ( uint  i = 0; i < nblocks(); ++i )
            {
                if ( block(i) != nullptr )
                {
                    HLR_ASSERT( u->block(i) != nullptr );
                    
                    block(i)->copy_to( u->block(i) );
                }// if
            }// for
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
        size_t  n = ( TVector::byte_size() +
                      sizeof(_is) + sizeof(_basis) +
                      sizeof(_coeffs) + sizeof(value_t) * _coeffs.length() +
                      sizeof(_blocks) + sizeof(sub_block_t*) * _blocks.size() );

        for ( auto  v : _blocks )
            if ( v != nullptr )
                n += v->byte_size();

        return n;
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
