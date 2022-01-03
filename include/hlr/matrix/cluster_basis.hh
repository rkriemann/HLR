#ifndef __HLR_MATRIX_CLUSTER_BASIS_HH
#define __HLR_MATRIX_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : matrix/cluster_basis
// Description : (non-nested) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlib-config.h>

#if defined(USE_LIC_CHECK)
#define HAS_H2
#endif

#include <hpro/cluster/TCluster.hh>
#if defined(HAS_H2)
#include <hpro/cluster/TClusterBasis.hh>
#endif

#include <hlr/utils/checks.hh>
#include <hlr/arith/blas.hh>

namespace hlr
{ 

namespace hpro = HLIB;

using indexset     = hpro::TIndexSet;
using cluster_tree = hpro::TCluster;
using accuracy     = hpro::TTruncAcc;

using hpro::real;
using hpro::complex;
using hpro::idx_t;

// local matrix type
DECLARE_TYPE( cluster_basis );

namespace matrix
{

//
// represents cluster basis for single cluster with
// additional hierarchy
//
template < typename T_value >
class cluster_basis
{
public:

    using  value_t = T_value;
    
private:
    // associated indexset
    const indexset                             _is;

    // cluster basis of sub clusters
    std::vector< cluster_basis< value_t > * >  _sons;
    
    // basis
    hlr::blas::matrix< value_t >               _V;

    // mutex for synchronised changes
    std::mutex                                 _mtx;
    
public:
    
    // construct cluster basis corresponding to cluster <cl>
    cluster_basis ( const indexset &  ais )
            : _is( ais )
    {}

    // construct cluster basis corresponding to cluster <cl>
    // with basis defined by <V>
    cluster_basis ( const indexset &                       ais,
                    const hlr::blas::matrix< value_t > &&  V )
            : _is( ais )
            , _V( V )
    {}

    cluster_basis ( const indexset &                       ais,
                    hlr::blas::matrix< value_t > &&        V )
            : _is( ais )
            , _V( std::move( V ) )
    {}

    // dtor: delete sons
    ~cluster_basis ()
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
    cluster_basis *        son       ( const uint  i )       { return _sons[i]; }
    const cluster_basis *  son       ( const uint  i ) const { return _sons[i]; }

    void                   set_son   ( const uint       i,
                                       cluster_basis *  cb )   { _sons[i] = cb; }

    //
    // access basis
    //

    // underlying indexset
    const indexset &                      is      () const { return _is; }
    
    // return rank of cluster basis
    uint                                  rank    () const { return _V.ncols(); }
    
    // return local basis
    const hlr::blas::matrix< value_t > &  basis   () const { return _V; }
    
    // set local basis
    void
    set_basis  ( const hlr::blas::matrix< value_t > &  aV )
    {
        _V = std::move( blas::copy( aV ) );
    }
    
    void
    set_basis  ( hlr::blas::matrix< value_t > &&  aV )
    {
        // TODO: test "copy" if dimensions are compatible
        _V = std::move( aV );
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
        return blas::mulvec( hlr::blas::adjoint( _V ), v );
    }
    
    hlr::blas::matrix< value_t >
    transform_forward  ( const hlr::blas::matrix< value_t > &  M ) const
    {
        return blas::prod( hlr::blas::adjoint( _V ), M );
    }
    
    //
    // backward transformation, e.g. compute vector/matrix in standard basis
    // - return V · arg
    //
    
    hlr::blas::vector< value_t >
    transform_backward  ( const hlr::blas::vector< value_t > &  s ) const
    {
        return hlr::blas::mulvec( _V, s );
    }
    
    hlr::blas::matrix< value_t >
    transform_backward  ( const hlr::blas::matrix< value_t > &  S ) const
    {
        return hlr::blas::prod( _V, S );
    }

    ///////////////////////////////////////////////////////
    //
    // misc.
    //

    // return copy of cluster basis
    std::unique_ptr< cluster_basis >
    copy () const
    {
        auto  cb = std::make_unique< cluster_basis >( _is, std::move( blas::copy( _V ) ) );

        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy().release() );

        return cb;
    }
    
    // return structural copy (no data) of cluster basis
    std::unique_ptr< cluster_basis >
    copy_struct () const
    {
        auto  cb = std::make_unique< cluster_basis >( _is );

        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy_struct().release() );

        return cb;
    }
    
    // return memory consumption
    size_t
    byte_size () const
    {
        size_t  n = ( sizeof(_is) +
                      sizeof(cluster_basis< value_t > *) * _sons.size() +
                      sizeof(_V) + sizeof(value_t) * _V.nrows() * _V.ncols() );

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
};

//
// copy given cluster basis between different value types
//
template < typename T_value_dest,
           typename T_value_src >
std::unique_ptr< cluster_basis< T_value_dest > >
copy ( const cluster_basis< T_value_src > &  src )
{
    auto  V_dest = blas::copy< T_value_dest >( src.basis() );
    auto  dest   = std::make_unique< cluster_basis< T_value_dest > >( src.is(), std::move( V_dest ) );

    dest->set_nsons( src.nsons() );
        
    for ( uint  i = 0; i < dest->nsons(); ++i )
    {
        if ( ! is_null( src.son( i ) ) )
            dest->set_son( i, copy< T_value_dest >( * src.son(i) ).release() );
    }// for

    return dest;
}

//
// return min/avg/max rank of given cluster basis
//
namespace detail
{

template < typename cluster_basis_t >
std::tuple< uint, size_t, uint, size_t >
rank_info_helper_cb ( const cluster_basis_t &  cb )
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
rank_info ( const cluster_basis< value_t > &  cb )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_cb( cb );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

#if defined(HAS_H2)

template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const hpro::TClusterBasis< value_t > &  cb )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_cb( cb );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

#endif
    
}} // namespace hlr::matrix

#endif // __HLR_MATRIX_CLUSTER_BASIS_HH
