#ifndef __HLR_MATRIX_COMPRESSIBLE_HH
#define __HLR_MATRIX_COMPRESSIBLE_HH
//
// Project     : HLR
// Module      : matrix/compressible
// Description : defines interface for compressible objects
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>

namespace hlr { namespace matrix {

struct compressible
{
public:
    //
    // compression interface
    //

    // compress internal data
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig ) = 0;

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
    return ! is_null( dynamic_cast< const compressible * >( &ref ) );
}

template < typename T >
bool
is_compressible ( const T *  ptr )
{
    return ! is_null( dynamic_cast< const compressible * >( ptr ) );
}

HLR_TEST_ALL( is_compressible, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_compressible, Hpro::TMatrix< value_t > )

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

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_COMPRESSIBLE_HH
