#ifndef __HLR_UTILS_DETAIL_SZ_HH
#define __HLR_UTILS_DETAIL_SZ_HH
//
// Project     : HLR
// Module      : utils/detail/sz
// Description : SZ related functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

////////////////////////////////////////////////////////////
//
// SZ related functions and types
//
////////////////////////////////////////////////////////////

#if defined(HLR_HAS_SZ)

#include <string.h>

#if defined(HLR_HAS_SZ)
#include <sz.h>
#include <zlib.h>
#endif

namespace hlr { namespace compress { namespace sz {

using byte_t = unsigned char;

//
// holds compression parameters
//
struct config
{
    int     err_bound_mode;
    double  abs_err_bound;
    double  rel_bound_ratio;
    double  pwr_bound_ratio;
};

// inline config  relative_accuracy ( double  eps ) { return config{ REL, 0.0, eps, 0.0 }; }
inline config  get_config ( double  eps ) { return config{ REL, 0.0, eps, 0.0 }; }

//
// handles arrays allocated _within_ SZ
//
struct zarray 
{
    using value_t = unsigned char;

private:
    byte_t *  _ptr;
    size_t    _size;

public:
    zarray ()
            : _ptr( nullptr )
            , _size( 0 )
    {}

    zarray ( const size_t  asize )
            : _ptr( reinterpret_cast< byte_t * >( ::malloc( asize ) ) )
            , _size( asize )
    {}

    zarray ( byte_t *      aptr,
             const size_t  asize )
            : _ptr( aptr )
            , _size( asize )
    {}

    zarray ( const zarray &  v )
            : _ptr( new byte_t[ v._size ] )
            , _size( v._size )
    {
        std::copy( v.begin(), v.end(), begin() );
    }

    zarray ( zarray &&  v )
            : _ptr( v._ptr )
            , _size( v._size )
    {
        v._ptr  = nullptr;
        v._size = 0;
    }

    ~zarray ()
    {
        free();
    }
    
    zarray &  operator = ( const zarray &  v )
    {
        free();
        
        _ptr  = new byte_t[ v._size ];
        _size = v._size;
        
        std::copy( v.begin(), v.end(), begin() );

        return *this;
    }
    
    zarray &  operator = ( zarray &&  v )
    {
        free();
        
        _ptr  = v._ptr;
        _size = v._size;
        
        v._ptr  = nullptr;
        v._size = 0;

        return *this;
    }
    
    byte_t *  begin () const { return _ptr; }
    byte_t *  end   () const { return _ptr + _size; }
    
    byte_t *  data () const { return _ptr; }
    size_t    size () const { return _size; }

    void  free ()
    {
        ::free( _ptr );
        _ptr  = nullptr;
        _size = 0;
    }
};

inline size_t  byte_size ( const zarray &  v ) { return sizeof(zarray) + v.size(); }

template < typename value_t >
zarray
compress ( const config &   config,
           const value_t *  data,
           const size_t     dim0,
           const size_t     dim1 = 0,
           const size_t     dim2 = 0,
           const size_t     dim3 = 0,
           const size_t     dim4 = 0 );

template <>
inline
zarray
compress< float > ( const config &  config,
                    const float *   data,
                    const size_t    dim0,
                    const size_t    dim1,
                    const size_t    dim2,
                    const size_t    dim3,
                    const size_t    dim4 )
{
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_FLOAT, const_cast< float * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

    return zarray( ptr, csize );
}

template <>
inline
zarray
compress< double > ( const config &  config,
                     const double *  data,
                     const size_t    dim0,
                     const size_t    dim1,
                     const size_t    dim2,
                     const size_t    dim3,
                     const size_t    dim4 )
{
    size_t  csize = 0;
    auto    ptr   = SZ_compress_args( SZ_DOUBLE, const_cast< double * >( data ), & csize,
                                      config.err_bound_mode,
                                      config.abs_err_bound,
                                      config.rel_bound_ratio,
                                      config.pwr_bound_ratio,
                                      dim4, dim3, dim2, dim1, dim0 );

    return zarray( ptr, csize );
}

template <>
inline
zarray
compress< std::complex< float > > ( const config &                 config,
                                    const std::complex< float > *  data,
                                    const size_t                   dim0,
                                    const size_t                   dim1,
                                    const size_t                   dim2,
                                    const size_t                   dim3,
                                    const size_t                   dim4 )
{
    HLR_ERROR( "TO DO" );
}

template <>
inline
zarray
compress< std::complex< double > > ( const config &                 config,
                                     const std::complex< double > * data,
                                     const size_t                   dim0,
                                     const size_t                   dim1,
                                     const size_t                   dim2,
                                     const size_t                   dim3,
                                     const size_t                   dim4 )
{
    HLR_ERROR( "TO DO" );
}

template < typename value_t >
void
decompress ( const byte_t *  zptr,
             const size_t    zsize,
             value_t *       dest,
             const size_t    dim0,
             const size_t    dim1 = 0,
             const size_t    dim2 = 0,
             const size_t    dim3 = 0,
             const size_t    dim4 = 0 );

template <>
inline
void
decompress< float > ( const byte_t *  zptr,
                      const size_t    zsize,
                      float *         dest,
                      const size_t    dim0,
                      const size_t    dim1,
                      const size_t    dim2,
                      const size_t    dim3,
                      const size_t    dim4 )
{
    SZ_decompress_args( SZ_FLOAT, const_cast< byte_t * >( zptr ), zsize,
                        dest,
                        dim4, dim3, dim2, dim1, dim0 );
}

template <>
inline
void
decompress< double > ( const byte_t *  zptr,
                       const size_t    zsize,
                       double *        dest,
                       const size_t    dim0,
                       const size_t    dim1,
                       const size_t    dim2,
                       const size_t    dim3,
                       const size_t    dim4 )
{
    SZ_decompress_args( SZ_DOUBLE, const_cast< byte_t * >( zptr ), zsize,
                        dest,
                        dim4, dim3, dim2, dim1, dim0 );
}

template <>
inline
void
decompress< std::complex< float > > ( const byte_t *            zptr,
                                      const size_t              zsize,
                                      std::complex< float > *   dest,
                                      const size_t              dim0,
                                      const size_t              dim1,
                                      const size_t              dim2,
                                      const size_t              dim3,
                                      const size_t              dim4 )
{
    HLR_ERROR( "TO DO" );
}

template <>
inline
void
decompress< std::complex< double > > ( const byte_t *            zptr,
                                       const size_t              zsize,
                                       std::complex< double > *  dest,
                                       const size_t              dim0,
                                       const size_t              dim1,
                                       const size_t              dim2,
                                       const size_t              dim3,
                                       const size_t              dim4 )
{
    HLR_ERROR( "TO DO" );
}

template < typename value_t >
void
decompress ( const zarray &  v,
             value_t *      dest,
             const size_t   dim0,
             const size_t   dim1 = 0,
             const size_t   dim2 = 0,
             const size_t   dim3 = 0,
             const size_t   dim4 = 0 )
{
    decompress( v.data(), v.size(), dest, dim0, dim1, dim2, dim3, dim4 );
}

//////////////////////////////////////////////////////////////////////////////////////
//
// special version for lowrank matrices
//
//////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
zarray
compress_lr ( const blas::matrix< value_t > &                       U,
              const blas::vector< Hpro::real_type_t< value_t > > &  S )
{
    //
    // first, determine exponent bits and mantissa bits for all
    // columns
    //

    const size_t  n     = U.nrows();
    const size_t  k     = U.ncols();
    size_t        zsize = 0;
    auto          zlist = std::vector< zarray >( k );

    for ( uint  l = 0; l < k; ++l )
    {
        auto  zconf = get_config( S(l) );
        auto  z_i   = compress( zconf, U.data() + l * n, n );

        zsize   += z_i.size();
        zlist[l] = std::move( z_i );
    }// for

    zarray  zdata( zsize + sizeof(uint) * k );
    size_t  pos = 0;

    for ( auto &  z_i : zlist )
    {
        const uint  s_i = z_i.size();
        
        memcpy( zdata.data() + pos, & s_i, sizeof(uint) );
        pos += sizeof(uint);
        
        memcpy( zdata.data() + pos, z_i.data(), s_i );
        pos += s_i;
    }// for

    return zdata;
}

template < typename value_t >
void
decompress_lr ( const zarray &             zdata,
                blas::matrix< value_t > &  U )
{
    const size_t  n   = U.nrows();
    const uint    k   = U.ncols();
    size_t        pos = 0;

    for ( uint  l = 0; l < k; ++l )
    {
        uint  s_i = 0;

        memcpy( & s_i, zdata.data() + pos, sizeof(uint) );
        pos += sizeof(uint);
        
        decompress( zdata.data() + pos, s_i, U.data() + l*n, n );
        pos += s_i;
    }// for
}

}}}// namespace hlr::compress::sz

#endif // HLR_HAS_SZ

#endif // __HLR_UTILS_DETAIL_SZ_HH
