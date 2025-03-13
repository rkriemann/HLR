#ifndef __HLR_TENSOR_STRUCTURED_TENSOR_HH
#define __HLR_TENSOR_STRUCTURED_TENSOR_HH
//
// Project     : HLR
// Module      : tensor/structured_tensor
// Description : structured tensor (with sub-blocks)
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <array>

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>

namespace hlr { namespace tensor {

//
// tensor with sub blocks
//
template < typename T_value >
class structured_tensor3 : public base_tensor3< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  super_t = base_tensor3< value_t >;

    static constexpr uint  dimension = 3;

private:
    // sub structure
    std::array< uint, 3 >     _nblocks;

    // sub blocks
    std::vector< super_t * >  _blocks;

public:
    //
    // ctors
    //

    structured_tensor3 ()
            : _nblocks{ 0, 0, 0 }
    {}

    structured_tensor3 ( const indexset &  is0,
                         const indexset &  is1,
                         const indexset &  is2 )
            : super_t( is0, is1, is2 )
            , _nblocks{ 0, 0, 0 }
    {}
    
    structured_tensor3 ( const std::array< indexset, 3 > &  ais )
            : super_t( ais[0], ais[1], ais[2] )
            , _nblocks{ 0, 0, 0 }
    {}
    
    structured_tensor3 ( const structured_tensor3 &  t )
            : super_t( t )
            , _nblocks{ 0, 0, 0 }
    {
        set_structure( t.nblocks(0),
                       t.nblocks(1),
                       t.nblocks(2) );
    }

    structured_tensor3 ( structured_tensor3 &&  t )
            : super_t( std::forward( t ) )
            , _nblocks{ 0, 0, 0 }
    {
        std::swap( _nblocks, t._nblocks );
        std::swap( _blocks,  t._blocks );
    }

    // dtor
    virtual ~structured_tensor3 ()
    {
        for ( auto  b : _blocks )
            if ( ! is_null( b ) )
                delete b;
    }

    // assignment
    structured_tensor3 &  operator = ( const structured_tensor3 &  t )
    {
        super_t::operator = ( t );

        set_structure( t.nblocks(0),
                       t.nblocks(1),
                       t.nblocks(2) );

        HLR_ERROR( "todo" );
    }

    structured_tensor3 &  operator = ( structured_tensor3 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _nblocks = std::move( t._nblocks );
        _blocks  = std::move( t._blocks );
    }

    //
    // access internal data
    //

    // return number of sub blocks per mode d
    uint  nblocks ( const uint  d ) const { HLR_ASSERT( d < dimension ); return _nblocks[d]; }

    // set new sub block structure (do not preserve blocks!)
    void
    set_structure ( const uint  n0,
                    const uint  n1,
                    const uint  n2 )
    {
        if (( n0 != _nblocks[0] ) ||
            ( n1 != _nblocks[1] ) ||
            ( n2 != _nblocks[2] ))
        {
            for ( auto  b : _blocks )
                if ( ! is_null( b ) )
                    delete b;

            _blocks.resize( n0*n1*n2 );
            _nblocks[0] = n0;
            _nblocks[1] = n1;
            _nblocks[2] = n2;
        }// if
    }
    
    // access sub blocks (column major storage)
    base_tensor3< value_t > *  block ( const uint  i,
                                       const uint  j,
                                       const uint  l )
    {
        HLR_ASSERT( ( i < nblocks(0) ) &&
                    ( j < nblocks(1) ) &&
                    ( l < nblocks(2) ) );
        
        return _blocks[ i + _nblocks[0] * ( j + _nblocks[1] * l ) ];
    }
            
    const base_tensor3< value_t > *  block ( const uint  i,
                                             const uint  j,
                                             const uint  l ) const
    {
        HLR_ASSERT( ( i < nblocks(0) ) &&
                    ( j < nblocks(1) ) &&
                    ( l < nblocks(2) ) );
        
        return _blocks[ i + _nblocks[0] * ( j + _nblocks[1] * l ) ];
    }

    // set sub block
    void
    set_block ( const uint                 i,
                const uint                 j,
                const uint                 l,
                base_tensor3< value_t > *  t )
    {
        HLR_ASSERT( ( i < nblocks(0) ) &&
                    ( j < nblocks(1) ) &&
                    ( l < nblocks(2) ) );

        if ( ! is_null( block(i,j,l) ) )
            delete block(i,j,l);

        _blocks[ i + _nblocks[0] * ( j + _nblocks[1] * l ) ] = t;
    }
    
    //
    // compression
    //

    // same but compress based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc )
    {
        for ( auto  b : _blocks )
        {
            if ( ! is_null( b ) )
                b->compress( acc );
        }// if
    }

    // decompress internal data
    virtual void   decompress    ()
    {
        for ( auto  b : _blocks )
        {
            if ( ! is_null( b ) )
                b->decompress();
        }// if
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        bool  c = false;
        
        for ( auto  b : _blocks )
        {
            if ( ! is_null( b ) )
                if ( b->is_compressed() )
                    c = true;
        }// if

        return  c;
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
        auto  X = ptrcast( T.get(), structured_tensor3< value_t > );

        X->set_structure( nblocks(0), nblocks(1), nblocks(2) );
        
        for ( uint  i = 0; i < nblocks(0)*nblocks(1)*nblocks(2); ++i )
            if ( ! is_null( _blocks[i] ) )
                X->_blocks[i] = _blocks[i]->copy().release();
        
        return T;
    }
    
    // create object of same type but without data
    virtual
    std::unique_ptr< base_tensor3< value_t > >
    create () const
    {
        return std::make_unique< structured_tensor3< value_t > >();
    }
    
    // return size in bytes used by this object (recursively!)
    virtual size_t  byte_size () const
    {
        size_t  s = super_t::byte_size() + sizeof(_nblocks) + sizeof(_blocks) + sizeof(this) * nblocks(0) * nblocks(1) * nblocks(2);

        for ( auto  b : _blocks )
            if ( ! is_null( b ) )
                s += b->byte_size();

        return s;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        size_t  s = 0;

        for ( auto  b : _blocks )
            if ( ! is_null( b ) )
                s += b->data_byte_size();

        return s;
    }
    
    // return name of type
    virtual std::string  typestr () const { return "structured_tensor3"; }
};

//
// type tests
//
bool
is_structured_tensor3 ( const has_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const structured_tensor3< value_t > * >( &t ) != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//
// tensor with sub blocks
//
template < typename T_value >
class structured_tensor4 : public base_tensor4< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  super_t = base_tensor4< value_t >;

    static constexpr uint  dimension = 4;

private:
    // sub structure
    std::array< uint, dimension >  _nblocks;

    // sub blocks
    std::vector< super_t * >       _blocks;

public:
    //
    // ctors
    //

    structured_tensor4 ()
            : _nblocks{ 0, 0, 0, 0 }
    {}

    structured_tensor4 ( const indexset &  is0,
                         const indexset &  is1,
                         const indexset &  is2,
                         const indexset &  is3 )
            : super_t( is0, is1, is2, is3 )
            , _nblocks{ 0, 0, 0, 0 }
    {}
    
    structured_tensor4 ( const std::array< indexset, dimension > &  ais )
            : super_t( ais[0], ais[1], ais[2], ais[3] )
            , _nblocks{ 0, 0, 0, 0 }
    {}
    
    structured_tensor4 ( const structured_tensor4 &  t )
            : super_t( t )
            , _nblocks{ 0, 0, 0, 0 }
    {
        set_structure( t.nblocks(0),
                       t.nblocks(1),
                       t.nblocks(2),
                       t.nblocks(3) );
    }

    structured_tensor4 ( structured_tensor4 &&  t )
            : super_t( std::forward( t ) )
            , _nblocks{ 0, 0, 0, 0 }
    {
        std::swap( _nblocks, t._nblocks );
        std::swap( _blocks,  t._blocks );
    }

    // dtor
    virtual ~structured_tensor4 ()
    {
        for ( auto  b : _blocks )
            if ( ! is_null( b ) )
                delete b;
    }

    // assignment
    structured_tensor4 &  operator = ( const structured_tensor4 &  t )
    {
        super_t::operator = ( t );

        set_structure( t.nblocks(0),
                       t.nblocks(1),
                       t.nblocks(2),
                       t.nblocks(3) );

        HLR_ERROR( "todo" );
    }

    structured_tensor4 &  operator = ( structured_tensor4 &&  t )
    {
        super_t::operator = ( std::forward( t ) );

        _nblocks = std::move( t._nblocks );
        _blocks  = std::move( t._blocks );
    }

    //
    // access internal data
    //

    // return number of sub blocks per mode d
    uint  nblocks ( const uint  d ) const { HLR_ASSERT( d < dimension ); return _nblocks[d]; }

    // set new sub block structure (do not preserve blocks!)
    void
    set_structure ( const uint  n0,
                    const uint  n1,
                    const uint  n2,
                    const uint  n3 )
    {
        if (( n0 != _nblocks[0] ) ||
            ( n1 != _nblocks[1] ) ||
            ( n2 != _nblocks[2] ) ||
            ( n3 != _nblocks[3] ))
        {
            for ( auto  b : _blocks )
                if ( ! is_null( b ) )
                    delete b;

            _blocks.resize( n0*n1*n2*n3 );
            _nblocks[0] = n0;
            _nblocks[1] = n1;
            _nblocks[2] = n2;
            _nblocks[3] = n3;
        }// if
    }
    
    // access sub blocks (column major storage)
    base_tensor4< value_t > *  block ( const uint  i0,
                                       const uint  i1,
                                       const uint  i2,
                                       const uint  i3 )
    {
        HLR_ASSERT( ( i0 < nblocks(0) ) &&
                    ( i1 < nblocks(1) ) &&
                    ( i2 < nblocks(2) ) &&
                    ( i3 < nblocks(3) ) );
        
        return _blocks[ i0 + _nblocks[0] * ( i1 + _nblocks[1] * ( i2 + _nblocks[2] * i3 ) ) ];
    }
            
    const base_tensor4< value_t > *  block ( const uint  i0,
                                             const uint  i1,
                                             const uint  i2,
                                             const uint  i3 ) const
    {
        HLR_ASSERT( ( i0 < nblocks(0) ) &&
                    ( i1 < nblocks(1) ) &&
                    ( i2 < nblocks(2) ) &&
                    ( i3 < nblocks(3) ) );
        
        return _blocks[ i0 + _nblocks[0] * ( i1 + _nblocks[1] * ( i2 + _nblocks[2] * i3 ) ) ];
    }

    // set sub block
    void
    set_block ( const uint                 i0,
                const uint                 i1,
                const uint                 i2,
                const uint                 i3,
                base_tensor4< value_t > *  t )
    {
        HLR_ASSERT( ( i0 < nblocks(0) ) &&
                    ( i1 < nblocks(1) ) &&
                    ( i2 < nblocks(2) ) &&
                    ( i3 < nblocks(3) ) );

        if ( ! is_null( block(i0,i1,i2,i3) ) )
            delete block(i0,i1,i2,i3);

        _blocks[ i0 + _nblocks[0] * ( i1 + _nblocks[1] * ( i2 + _nblocks[2] * i3 ) ) ] = t;
    }
    
    //
    // compression
    //

    // same but compress based on given accuracy
    virtual void   compress      ( const accuracy &  acc )
    {
        for ( auto  b : _blocks )
        {
            if ( ! is_null( b ) )
                b->compress( acc );
        }// if
    }

    // decompress internal data
    virtual void   decompress    ()
    {
        for ( auto  b : _blocks )
        {
            if ( ! is_null( b ) )
                b->decompress();
        }// if
    }

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        bool  c = false;
        
        for ( auto  b : _blocks )
        {
            if ( ! is_null( b ) )
                if ( b->is_compressed() )
                    c = true;
        }// if

        return  c;
    }

    //
    // misc
    //
    
    // return copy of local object
    virtual
    std::unique_ptr< base_tensor4< value_t > >
    copy () const
    {
        auto  T = super_t::copy();
        auto  X = ptrcast( T.get(), structured_tensor4< value_t > );

        X->set_structure( nblocks(0), nblocks(1), nblocks(2), nblocks(3) );
        
        for ( uint  i = 0; i < nblocks(0)*nblocks(1)*nblocks(2)*nblocks(3); ++i )
            if ( ! is_null( _blocks[i] ) )
                X->_blocks[i] = _blocks[i]->copy().release();
        
        return T;
    }
    
    // create object of same type but without data
    virtual
    std::unique_ptr< base_tensor4< value_t > >
    create () const
    {
        return std::make_unique< structured_tensor4< value_t > >();
    }
    
    // return size in bytes used by this object (recursively!)
    virtual size_t  byte_size () const
    {
        size_t  s = super_t::byte_size() + sizeof(_nblocks) + sizeof(_blocks) + sizeof(this) * nblocks(0) * nblocks(1) * nblocks(2) * nblocks(3);

        for ( auto  b : _blocks )
            if ( ! is_null( b ) )
                s += b->byte_size();

        return s;
    }

    // return size of (floating point) data in bytes handled by this object
    virtual size_t data_byte_size () const
    {
        size_t  s = 0;

        for ( auto  b : _blocks )
            if ( ! is_null( b ) )
                s += b->data_byte_size();

        return s;
    }
    
    // return name of type
    virtual std::string  typestr () const { return "structured_tensor4"; }
};

//
// type tests
//
bool
is_structured_tensor4 ( const has_value_type auto &  t )
{
    using type_t  = std::remove_cv_t< std::remove_reference_t< decltype( t ) > >;
    using value_t = typename type_t::value_t;
    
    return dynamic_cast< const structured_tensor4< value_t > * >( &t ) != nullptr;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_STRUCTURED_TENSOR4_HH
