#ifndef __HLR_UTILS_DETAIL_DUMMY_HH
#define __HLR_UTILS_DETAIL_DUMMY_HH
//
// Project     : HLR
// Module      : utils/detail/dummy
// Description : functions for dummy compression
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// dummy compression with optional info
//
////////////////////////////////////////////////////////////

namespace hlr { namespace compress { namespace dummy {

using byte_t = unsigned char;

struct config
{};

// holds compressed data
using  zarray = std::vector< byte_t >;

inline size_t  byte_size  ( const zarray &  v   ) { return v.size(); }
inline config  get_config ( const double    eps ) { return config{}; }

template < typename value_t >
zarray
compress ( const config &   config,
           value_t *        data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    zarray        zdata( nsize * sizeof(value_t) );
    
    std::copy( reinterpret_cast< const value_t * >( data ),
               reinterpret_cast< const value_t * >( data + nsize ),
               reinterpret_cast< value_t * >( zdata.data() ) );
    
    return zdata;
}

template <>
inline
zarray
compress< double > ( const config &   config,
                     double *         data,
                     const size_t     dim0,
                     const size_t     dim1,
                     const size_t     dim2,
                     const size_t     dim3 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    zarray        zdata( nsize * sizeof(double) );
    
    //
    // analyze data
    //

    double  vmin     = 0;
    double  vmax     = 0;
    bool    has_zero = false;

    {
        size_t  i = 0;

        for ( ; i < nsize; ++i )
        {
            const auto  di = std::abs( data[i] );

            if ( di > double(0) )
            {
                vmin = di;
                vmax = di;
                break;
            }// if
            else
                has_zero = true;
        }// for
        
        for ( ; i < nsize; ++i )
        {
            const auto  di = std::abs( data[i] );

            if ( di > double(0) )
            {
                vmin = std::min( vmin, di );
                vmax = std::max( vmax, di );
            }// if
            else
                has_zero = true;
        }// for
    }

    const auto  ratio    = vmax / vmin;
    const auto  range    = std::ceil( std::log10( vmax / vmin ) );
    const uint  exp_bits = std::max< double >( 1, std::ceil( std::log2( std::log2( vmax / vmin ) ) ) );

    std::cout << exp_bits << " / " << range << " / " << has_zero << std::endl;

    if ( exp_bits > 5 )
        std::cout << vmin << " / " << vmax << std::endl;
        
    //
    // dummy copy
    //
    
    std::copy( reinterpret_cast< const double * >( data ),
               reinterpret_cast< const double * >( data + nsize ),
               reinterpret_cast< double * >( zdata.data() ) );
    
    return zdata;
}

template < typename value_t >
void
decompress ( const zarray &  zdata,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 )
{
    const size_t  nsize = ( dim3 == 0 ? ( dim2 == 0 ? ( dim1 == 0 ? dim0 : dim0 * dim1 ) : dim0 * dim1 * dim2 ) : dim0 * dim1 * dim2 * dim3 );
    
    std::copy( reinterpret_cast< const value_t * >( zdata.data() ),
               reinterpret_cast< const value_t * >( zdata.data() + nsize ),
               dest );
}

}}}// namespace hlr::compress::dummy

#endif // __HLR_UTILS_DETAIL_DUMMY_HH
