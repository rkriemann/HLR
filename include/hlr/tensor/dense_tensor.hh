#ifndef __HLR_TENSOR_DENSE_TENSOR_HH
#define __HLR_TENSOR_DENSE_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/dense_tensor
// Description : dense (full) tensor
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/utils/traits.hh>
#include <hlr/utils/io.hh>
#include <hlr/compress/direct.hh>

#include <hlr/tensor/base_tensor.hh>

namespace hlr { namespace tensor {

//
// implements dense (full) 3D tensor
// - storage layout is column-major
//
template < typename T_value >
class dense_tensor3 : public base_tensor3< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  super_t = base_tensor3< value_t >;

    static constexpr uint  dimension = 3;

private:
    // tensor data
    blas::tensor3< value_t >   _tensor;

    // compressed data
    compress::zarray           _ztensor;
    
public:
    //
    // ctors
    //

    dense_tensor3 ()
    {}

    dense_tensor3 ( const dense_tensor3 &  t )
            : super_t( t )
            , _tensor( t._tensor.size() )
    {
        blas::copy( t._tensor, _tensor );
    }

    dense_tensor3 ( dense_tensor3 &&  t )
            : super_t( std::forward< base_tensor3< value_t > >( t ) )
    {
        std::swap( _tensor, t._tensor );
    }

    dense_tensor3 ( const indexset &  is0,
                    const indexset &  is1,
                    const indexset &  is2 )
            : super_t( is0, is1, is2 )
            , _tensor( is0.size(), is1.size(), is2.size() )
    {}
    
    dense_tensor3 ( const std::array< indexset, 3 > &  ais )
            : super_t{ ais[0], ais[1], ais[2] }
            , _tensor( ais[0].size(), ais[1].size(), ais[2].size() )
    {}
    
    dense_tensor3 ( const indexset &             is0,
                    const indexset &             is1,
                    const indexset &             is2,
                    blas::tensor3< value_t > &&  t )
            : super_t( is0, is1, is2 )
            , _tensor( std::move( t ) )
    {}
    
    // dtor
    virtual ~dense_tensor3 ()
    {}

    // assignment
    dense_tensor3 &  operator = ( const dense_tensor3 &  t )
    {
        super_t::operator = ( t );

        _tensor = blas::copy( t._tensor );
    }

    dense_tensor3 &  operator = ( dense_tensor3 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _tensor = std::move( t._tensor );
    }

    //
    // access internal data
    //

    // blas::tensor3< value_t > &
    // tensor ()
    // {
    //     HLR_ASSERT( ! is_compressed() );
    //
    //     return _tensor;
    // }
    
    // const blas::tensor3< value_t > &
    // tensor () const
    // {
    //     HLR_ASSERT( ! is_compressed() );
    // 
    //     return _tensor;
    // }
    
    blas::tensor3< value_t >
    tensor () const
    {
        if ( is_compressed() )
        {
            auto  dT = blas::tensor3< value_t >( this->dim(0),
                                                 this->dim(1),
                                                 this->dim(2) );
    
            compress::decompress< value_t >( _ztensor, dT );
            
            return dT;
        }// if

        return _tensor;
    }

    value_t          coeff       ( const uint  i,
                                   const uint  j,
                                   const uint  l ) const
    {
        HLR_DBG_ASSERT( ! is_compressed() );
        return this->_tensor(i,j,l);
    }
    value_t &        coeff       ( const uint  i,
                                   const uint  j,
                                   const uint  l )
    {
        HLR_DBG_ASSERT( ! is_compressed() );
        return this->_tensor(i,j,l);
    }
    
    value_t          operator () ( const uint  i,
                                   const uint  j,
                                   const uint  l ) const { return coeff( i, j, l ); }
    value_t &        operator () ( const uint  i,
                                   const uint  j,
                                   const uint  l )       { return coeff( i, j, l ); }
    
    //
    // compression
    //

    // compress internal data based on given accuracy
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! _ztensor.empty();
    }

    //
    // misc
    //
    
    // return copy of local object
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    copy () const
    {
        auto  T = super_t::copy();
        auto  X = ptrcast( T.get(), dense_tensor3< value_t > );

        X->_tensor  = blas::copy( _tensor );

        X->_ztensor = compress::zarray( _ztensor.size() );
        std::copy( _ztensor.begin(), _ztensor.end(), X->_ztensor.begin() );
        
        return T;
    }
    
    // create object of same type but without data
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    create () const
    {
        return std::make_unique< dense_tensor3< value_t > >();
    }
    
    // return size in bytes used by this object
    virtual size_t  byte_size () const
    {
        return super_t::byte_size() + _tensor.byte_size() + hlr::compress::byte_size( _ztensor );
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        if ( is_compressed() )
            return hlr::compress::byte_size( _ztensor );
        else
            return _tensor.data_byte_size();
    }
    
    // return name of type
    virtual std::string  typestr () const { return "dense_tensor3"; }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void  remove_compressed ()
    {
        _ztensor = compress::zarray();
    }
};

//
// compress internal data
// - may result in non-compression if storage does not decrease
//
template < typename value_t >
void
dense_tensor3< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    if ( is_compressed() )
        return;

    if ( this->dim(0) * this->dim(1) * this->dim(2) == 0 )
        return;

    if (( this->is(0) == indexset( 160, 191 ) ) &&
        ( this->is(1) == indexset(   0,  31 ) ) &&
        ( this->is(2) == indexset(   0,  31 ) ))
        hlr::breakpoint();

    auto  T   = this->tensor();
    auto  tol = acc.abs_eps();

    if ( acc.abs_eps() != 0 )
    {
        if      ( acc.norm_mode() == Hpro::spectral_norm  ) tol = acc.abs_eps() / blas::norm_F( T ); // TODO
        else if ( acc.norm_mode() == Hpro::frobenius_norm ) tol = acc.abs_eps() / blas::norm_F( T );
        else
            HLR_ERROR( "unsupported norm mode" );
    }// if
    else if ( acc.rel_eps() != 0 )
    {
        tol = acc.rel_eps();
    }// if
    else
        HLR_ERROR( "zero error" );

    const auto    zconf     = compress::get_config( tol );
    const size_t  mem_dense = sizeof(value_t) * T.size(0) * T.size(1) * T.size(2);
    auto          zT        = compress::compress< value_t >( zconf, T );

    // // DEBUG
    // {
    //     auto  dT = blas::tensor3< value_t >( T.size(0), T.size(1), T.size(2) );

    //     compress::decompress( zT, dT );

    //     io::hdf5::write( T, "T1" );
    //     io::hdf5::write( dT, "T2" );
        
    //     blas::add( -1, T, dT );

    //     std::cout << "D "
    //               << this->is(0) << " × " 
    //               << this->is(1) << " × " 
    //               << this->is(2)
    //               << " : " 
    //               << blas::norm_F( dT ) / blas::norm_F( T )
    //               << " / "
    //               << blas::max_abs_val( dT )
    //               << std::endl;
            
    //     // for ( size_t  i = 0; i < M.nrows() * M.ncols(); ++i )
    //     // {
    //     //     const auto  error = std::abs( (M.data()[i] - dM.data()[i]) / M.data()[i] );

    //     //     if ( error > 1e-6 )
    //     //         std::cout << "D " << i << " : "
    //     //                   << M.data()[i] << " / "
    //     //                   << dM.data()[i] << " / "
    //     //                   << std::abs( error )
    //     //                   << std::endl;
    //     // }// for
    // }
    
    if ( compress::byte_size( zT ) < mem_dense )
    {
        _ztensor = std::move( zT );
        _tensor  = std::move( blas::tensor3< value_t >() );
    }// if
}

//
// decompress internal data
//
template < typename value_t >
void
dense_tensor3< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    this->_tensor = std::move( tensor() );

    remove_compressed();
}

//
// type tests
//
bool
is_dense ( const has_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const dense_tensor3< value_t > * >( &t ) != nullptr;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_DENSE_TENSOR3_HH
