#ifndef __HLR_MATRIX_SHARED_CLUSTER_BASIS_HH
#define __HLR_MATRIX_SHARED_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : matrix/shared_cluster_basis
// Description : (non-nested) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/cluster/TCluster.hh>

#include <hlr/arith/blas.hh>
#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>

namespace hlr
{ 

using indexset     = Hpro::TIndexSet;
using cluster_tree = Hpro::TCluster;
using accuracy     = Hpro::TTruncAcc;

using Hpro::idx_t;

// local matrix type
DECLARE_TYPE( shared_cluster_basis );

namespace matrix
{

#define HLR_USE_APCOMPRESSION  1

//
// represents cluster basis for single cluster with
// additional hierarchy
//
template < typename T_value >
class shared_cluster_basis : public compress::compressible
{
public:

    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  self_t  = shared_cluster_basis< value_t >;
    
private:
    // associated indexset
    const indexset           _is;

    // cluster basis of sub clusters
    std::vector< self_t * >  _sons;
    
    // basis
    blas::matrix< value_t >  _V;

    // mutex for synchronised changes
    std::mutex               _mtx;

    #if HLR_HAS_COMPRESSION == 1
    #if HLR_USE_APCOMPRESSION == 1
    // stores compressed data
    compress::aplr::zarray   _zV;

    // also singular values assoc. with basis vectors
    // in case of adaptive precision compression
    blas::vector< real_t >   _sv;

    #else
        
    // stores compressed data
    compress::zarray         _zV;

    #endif
    #endif
    
public:
    
    // construct cluster basis corresponding to cluster <cl>
    shared_cluster_basis ( const indexset &  ais )
            : _is( ais )
    {}

    // construct cluster basis corresponding to cluster <cl>
    // with basis defined by <V>
    shared_cluster_basis ( const indexset &                      ais,
                           const hlr::blas::matrix< value_t > &  V )
            : _is( ais )
            , _V( V )
    {}

    shared_cluster_basis ( const indexset &                      ais,
                           hlr::blas::matrix< value_t > &&       V )
            : _is( ais )
            , _V( std::move( V ) )
    {}

    // dtor: delete sons
    ~shared_cluster_basis ()
    {
        for ( auto  cb : _sons )
            delete cb;
    }

    //
    // access sons
    //

    // return number of sons
    uint                   nsons     () const                { return _sons.size(); }

    // set number of sons
    void                   set_nsons ( const uint  n )       { _sons.resize( n ); }

    // access <i>'th son
    shared_cluster_basis *        son       ( const uint  i )       { return _sons[i]; }
    const shared_cluster_basis *  son       ( const uint  i ) const { return _sons[i]; }

    void                   set_son   ( const uint       i,
                                       shared_cluster_basis *  cb ) { _sons[i] = cb; }

    //
    // access basis
    //

    // underlying indexset
    const indexset &                      is     () const { return _is; }
    
    // return rank of cluster basis
    uint                                  rank   () const { return _V.ncols(); }
    
    // return local basis
    hlr::blas::matrix< value_t >          basis  () const
    {
        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
        {
            auto  V = blas::matrix< value_t >( _is.size(), rank() );
    
            #if HLR_USE_APCOMPRESSION == 1

            compress::aplr::decompress_lr< value_t >( _zV, V );

            #else
            
            compress::decompress< value_t >( _zV, V );

            #endif
            
            return V;
        }// if

        #endif

        return _V;
    }
    
    // set local basis
    void
    set_basis  ( const blas::matrix< value_t > &  aV )
    {
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( blas::copy( aV ) );
    }
    
    void
    set_basis  ( blas::matrix< value_t > &&  aV )
    {
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( aV );
    }

    // set basis together with singular values
    void
    set_basis  ( const blas::matrix< value_t > &  aV,
                 const blas::vector< real_t > &   asv )
    {
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( blas::copy( aV ) );

        #if HLR_HAS_COMPRESSION == 1 && HLR_USE_APCOMPRESSION == 1
        
        if ( _sv.length() == asv.length() ) blas::copy( asv, _sv );
        else                                _sv = std::move( blas::copy( asv ) );

        #endif
    }
    
    void
    set_basis  ( blas::matrix< value_t > &&  aV,
                 blas::vector< real_t > &&   asv )
    {
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( aV );

        #if HLR_HAS_COMPRESSION == 1 && HLR_USE_APCOMPRESSION == 1
        
        if ( _sv.length() == asv.length() ) blas::copy( asv, _sv );
        else                                _sv = std::move( asv );

        #endif
    }
    
    ///////////////////////////////////////////////////////
    //
    // basis transformation
    //

    //
    // forward transformation, e.g. compute coefficients in local basis
    // - return V^H · arg
    //
    
    hlr::blas::vector< value_t >
    transform_forward  ( const hlr::blas::vector< value_t > &  v ) const
    {
        auto  V = basis();
        
        return blas::mulvec( hlr::blas::adjoint( V ), v );
    }
    
    hlr::blas::matrix< value_t >
    transform_forward  ( const hlr::blas::matrix< value_t > &  M ) const
    {
        auto  V = basis();
        
        return blas::prod( hlr::blas::adjoint( V ), M );
    }
    
    //
    // backward transformation, e.g. compute vector/matrix in standard basis
    // - return V · arg
    //
    
    hlr::blas::vector< value_t >
    transform_backward  ( const hlr::blas::vector< value_t > &  s ) const
    {
        auto  V = basis();
        
        return hlr::blas::mulvec( V, s );
    }
    
    hlr::blas::matrix< value_t >
    transform_backward  ( const hlr::blas::matrix< value_t > &  S ) const
    {
        auto  V = basis();
        
        return hlr::blas::prod( V, S );
    }

    ///////////////////////////////////////////////////////
    //
    // misc.
    //

    // return copy of cluster basis
    std::unique_ptr< shared_cluster_basis >
    copy () const
    {
        auto  cb = std::make_unique< shared_cluster_basis >( _is, std::move( blas::copy( _V ) ) );

        #if HLR_HAS_COMPRESSION == 1
        
        if ( is_compressed() )
            cb->_zV = _zV;

        #if HLR_USE_APCOMPRESSION == 1
        cb->_sv = std::move( blas::copy( _sv ) );
        #endif
        #endif
        
        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy().release() );

        return cb;
    }
    
    // return structural copy (no data) of cluster basis
    std::unique_ptr< shared_cluster_basis >
    copy_struct () const
    {
        auto  cb = std::make_unique< shared_cluster_basis >( _is );

        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy_struct().release() );

        return cb;
    }
    
    // return memory consumption
    size_t
    byte_size () const
    {
        size_t  n = ( sizeof(_is) + sizeof(self_t *) * _sons.size() + _V.byte_size() );

        #if HLR_HAS_COMPRESSION == 1
        #if HLR_USE_APCOMPRESSION  == 1
        n += compress::aplr::byte_size( _zV );
        n += _sv.byte_size();
        #else
        n += hlr::compress::byte_size( _zV );
        #endif
        #endif
        
        for ( auto  son : _sons )
            n += son->byte_size();

        return  n;
    }

    // return depth of cluster basis
    size_t
    depth () const
    {
        size_t  d = 0;

        for ( auto  son : _sons )
            d = std::max( d, son->depth() );

        return d+1;
    }

    //
    // access mutex
    //

    decltype(_mtx) &  mutex () { return _mtx; }

    //
    // compression
    //

    // compress internal data based on given configuration
    // - may result in non-compression if storage does not decrease
    virtual void   compress      ( const compress::zconfig_t &  zconfig );

    // compress internal data based on given accuracy
    virtual void   compress      ( const Hpro::TTruncAcc &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        #if HLR_HAS_COMPRESSION == 1
        return _zV.size() > 0;
        #endif
        
        return false;
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        #if HLR_HAS_COMPRESSION == 1
        #if HLR_USE_APCOMPRESSION  == 1
        _zV = compress::aplr::zarray();
        #else
        _zV = compress::zarray();
        #endif
        #endif
    }
};

//
// copy given cluster basis between different value types
//
template < typename value_dest_t,
           typename value_src_t = value_dest_t >
std::unique_ptr< shared_cluster_basis< value_dest_t > >
copy ( const shared_cluster_basis< value_src_t > &  src )
{
    auto  V_dest = blas::copy< value_dest_t, value_src_t >( src.basis() );
    auto  dest   = std::make_unique< shared_cluster_basis< value_dest_t > >( src.is(), std::move( V_dest ) );

    dest->set_nsons( src.nsons() );
        
    for ( uint  i = 0; i < dest->nsons(); ++i )
    {
        if ( ! is_null( src.son( i ) ) )
            dest->set_son( i, copy< value_dest_t >( * src.son(i) ).release() );
    }// for

    return dest;
}

//
// compress internal data
//
template < typename value_t >
void
shared_cluster_basis< value_t >::compress ( const compress::zconfig_t &  zconfig )
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( is_compressed() )
        return;

    const size_t  mem_dense = sizeof(value_t) * _V.nrows() * _V.ncols();

    #if HLR_USE_APCOMPRESSION == 1

    HLR_ASSERT( false );
    
    #endif
    
    auto  zV = compress::compress< value_t >( zconfig, _V );

    if ( compress::byte_size( zV ) < mem_dense )
    {
        _zV = std::move( zV );
        _V  = std::move( blas::matrix< value_t >( 0, _V.ncols() ) ); // remember rank
    }// if

    #endif
}

template < typename value_t >
void
shared_cluster_basis< value_t >::compress ( const Hpro::TTruncAcc &  acc )
{
    if ( is_compressed() )
        return;

    #if HLR_HAS_COMPRESSION == 1
        
    HLR_ASSERT( acc.rel_eps() == 0 );

    if ( _V.nrows() * _V.ncols() == 0 )
        return;
        
    const size_t  mem_dense = sizeof(value_t) * _V.nrows() * _V.ncols();

    #if HLR_USE_APCOMPRESSION == 1

    {
        //
        // use adaptive precision per basis vector
        //

        HLR_ASSERT( _sv.length() == _V.ncols() );

        // auto  norm = real_t(0);

        // for ( uint  i = 0; i < _sv.length(); ++i )
        //     norm += _sv(0) * _sv(0);

        // norm = std::sqrt( norm );
                
        real_t  tol  = acc.abs_eps() * _sv(0);
        auto    S    = blas::copy( _sv );

        for ( uint  l = 0; l < S.length(); ++l )
            S(l) = tol / S(l);

        auto  zV = compress::aplr::compress_lr< value_t >( _V, S );

        // {
        //     auto  T = blas::copy( _V );

        //     compress::aplr::decompress_lr< value_t >( zV, T );

        //     blas::add( value_t(-1), _V, T );
        //     std::cout << blas::norm_F( T ) << " / " << blas::norm_F( T ) / blas::norm_F( _V ) << std::endl;
        // }
        
        if ( compress::aplr::byte_size( zV ) < mem_dense )
        {
            _zV = std::move( zV );
            _V  = std::move( blas::matrix< value_t >( 0, _V.ncols() ) ); // remember rank
        }// if

        return;
    }// if
    
    #endif

    //
    // use standard compression
    //
    
    auto  zconfig = compress::get_config( acc.abs_eps() );
    auto  zV      = compress::compress< value_t >( zconfig, _V );

    if ( compress::byte_size( zV ) < mem_dense )
    {
        _zV = std::move( zV );
        _V  = std::move( blas::matrix< value_t >( 0, _V.ncols() ) ); // remember rank
    }// if

    #endif
}

//
// decompress internal data
//
template < typename value_t >
void
shared_cluster_basis< value_t >::decompress ()
{
    #if HLR_HAS_COMPRESSION == 1
        
    if ( ! is_compressed() )
        return;

    _V = std::move( basis() );

    remove_compressed();

    #endif
}

//
// return min/avg/max rank of given cluster basis
//
namespace detail
{

template < typename shared_cluster_basis_t >
std::tuple< uint, size_t, uint, size_t >
rank_info_helper_cb ( const shared_cluster_basis_t &  cb )
{
    uint    min_rank = cb.rank();
    uint    max_rank = cb.rank();
    size_t  sum_rank = cb.rank();
    size_t  nnodes   = cb.rank() > 0 ? 1 : 0;

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            auto [ min_i, sum_i, max_i, n_i ] = rank_info_helper_cb( *cb.son(i) );

            if      ( min_rank == 0 ) min_rank = min_i;
            else if ( min_i    != 0 ) min_rank = std::min( min_rank, min_i );
            
            max_rank  = std::max( max_rank, max_i );
            sum_rank += sum_i;
            nnodes   += n_i;
        }// for
    }// if

    return { min_rank, sum_rank, max_rank, nnodes };
}

}// namespace detail

template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const shared_cluster_basis< value_t > &  cb )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_cb( cb );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

#undef HLR_USE_APCOMPRESSION

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_SHARED_CLUSTER_BASIS_HH
