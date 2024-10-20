#ifndef __HLR_COMPRESS_COMPRESSIBLE_HH
#define __HLR_COMPRESS_COMPRESSIBLE_HH
//
// Project     : HLR
// Module      : compress/compressible
// Description : base class for compressible objects
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/compress/direct.hh>

namespace hlr { namespace compress {

//
// general interface for compressible objects
//
struct compressible
{
public:
    //
    // compression interface
    //

    // // compress internal data
    // // - may result in non-compression if storage does not decrease
    // virtual void   compress      ( const compress::zconfig_t &  zconfig ) = 0;

    // compress data based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc ) = 0;

    // decompress internal data
    virtual void   decompress    () = 0;

    // return true if data is compressed
    virtual bool   is_compressed () const = 0;
};

//
// test functions
//
template < typename T >
bool
is_compressible ( const T &  ref )
{
    const auto  C = dynamic_cast< const compressible * >( &ref );
    
    return ! is_null( C );
}

template < typename T >
bool
is_compressible ( const T *  ptr )
{
    const auto  C = dynamic_cast< const compressible * >( ptr );

    return ! is_null( C );
}

template < typename T >
bool
is_compressed ( const T &  ref )
{
    const auto  C = dynamic_cast< const compressible * >( &ref );
    
    return ! is_null( C ) && C->is_compressed();
}

template < typename T >
bool
is_compressed ( const T *  ptr )
{
    const auto  C = dynamic_cast< const compressible * >( ptr );
    
    return ! is_null( C ) && C->is_compressed();
}

}}// hlr::compress

#endif // __HLR_COMPRESS_COMPRESSIBLE_HH
