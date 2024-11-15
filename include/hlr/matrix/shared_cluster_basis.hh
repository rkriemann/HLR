#ifndef __HLR_MATRIX_SHARED_CLUSTER_BASIS_HH
#define __HLR_MATRIX_SHARED_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : matrix/shared_cluster_basis
// Description : (non-nested) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/cluster/TCluster.hh>

#include <hlr/arith/blas.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/compress/compressible.hh>
#include <hlr/compress/direct.hh>
#include <hlr/compress/aplr.hh>
#include <hlr/utils/checks.hh>

namespace hlr
{ 

using indexset     = Hpro::TIndexSet;
using cluster_tree = Hpro::TCluster;

using Hpro::idx_t;

// local matrix type
DECLARE_TYPE( shared_cluster_basis );

namespace matrix
{

//
// represents (orthogonal) cluster basis for single cluster
// with additional hierarchy
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

    // unique ID within hierarchical cluster basis
    int                      _id;
    
    // cluster basis of sub clusters
    std::vector< self_t * >  _sons;
    
    // basis
    blas::matrix< value_t >  _V;

    // stores compressed data
    compress::aplr::zarray   _zV;

    // also singular values assoc. with basis vectors
    // in case of adaptive precision compression
    blas::vector< real_t >   _sv;
    
    // mutex for synchronised changes
    std::mutex               _mtx;

public:
    
    // construct cluster basis corresponding to cluster <cl>
    shared_cluster_basis ( const indexset &  ais )
            : _is( ais )
            , _id(-1)
    {}

    // construct cluster basis corresponding to cluster <cl>
    // with basis defined by <V>
    shared_cluster_basis ( const indexset &                 ais,
                           const blas::matrix< value_t > &  V )
            : _is( ais )
            , _id(-1)
            , _V( V )
    {}

    shared_cluster_basis ( const indexset &                 ais,
                           blas::matrix< value_t > &&       V )
            : _is( ais )
            , _id(-1)
            , _V( std::move( V ) )
    {}

    // dtor: delete sons
    ~shared_cluster_basis ()
    {
        for ( auto  cb : _sons )
            delete cb;
    }

    // return ID
    int   id         () const { return _id; }

    // set ID
    void  set_id     ( const int  aid ) { _id = aid; }

    //
    // access sons
    //

    // return number of sons
    uint                          nsons     () const                { return _sons.size(); }

    // set number of sons
    void                          set_nsons ( const uint  n )       { _sons.resize( n ); }

    // access <i>'th son
    shared_cluster_basis *        son       ( const uint  i )       { return _sons[i]; }
    const shared_cluster_basis *  son       ( const uint  i ) const { return _sons[i]; }

    void                          set_son   ( const uint              i,
                                              shared_cluster_basis *  cb ) { _sons[i] = cb; }

    //
    // access basis
    //

    // underlying indexset
    const indexset &                 is     () const { return _is; }
    
    // return rank of cluster basis
    uint                             rank   () const { return _V.ncols(); }
    
    // return local basis
    blas::matrix< value_t >          basis  () const
    {
        if ( is_compressed() )
        {
            auto  V = blas::matrix< value_t >( _is.size(), rank() );
    
            compress::aplr::decompress_lr< value_t >( _zV, V );
            
            return V;
        }// if

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

        if ( _sv.length() == asv.length() ) blas::copy( asv, _sv );
        else                                _sv = std::move( blas::copy( asv ) );
    }
    
    void
    set_basis  ( blas::matrix< value_t > &&  aV,
                 blas::vector< real_t > &&   asv )
    {
        if (( _V.nrows() == aV.nrows() ) && ( _V.ncols() == aV.ncols() )) blas::copy( aV, _V );
        else                                                              _V = std::move( aV );

        if ( _sv.length() == asv.length() ) blas::copy( asv, _sv );
        else                                _sv = std::move( asv );
    }
    
    ///////////////////////////////////////////////////////
    //
    // basis transformation
    //

    //
    // forward transformation, e.g. compute coefficients in local basis
    // - return V^H · arg
    //
    
    blas::vector< value_t >
    transform_forward  ( const blas::vector< value_t > &  v ) const
    {
        #if defined(HLR_HAS_ZBLAS_APLR)
        if ( is_compressed() )
        {
            const auto  k = this->rank();
            auto        t = blas::vector< value_t >( k );

            compress::aplr::zblas::mulvec( _is.size(), k, apply_adjoint, value_t(1), _zV, v.data(), t.data() );

            return t;
        }// if
        else
        #endif
        {
            auto  V = basis();
        
            return blas::mulvec( blas::adjoint( V ), v );
        }// else
    }
    
    blas::matrix< value_t >
    transform_forward  ( const blas::matrix< value_t > &  M ) const
    {
        auto  V = basis();
        
        return blas::prod( blas::adjoint( V ), M );
    }
    
    //
    // backward transformation, e.g. compute vector/matrix in standard basis
    // - return V · arg
    //
    
    blas::vector< value_t >
    transform_backward  ( const blas::vector< value_t > &  s ) const
    {
        #if defined(HLR_HAS_ZBLAS_APLR)
        if ( is_compressed() )
        {
            const auto  n = _is.size();
            auto        t = blas::vector< value_t >( n );

            compress::aplr::zblas::mulvec( n, this->rank(), apply_normal, value_t(1), _zV, s.data(), t.data() );

            return t;
        }// if
        else
        #endif
        {
            auto  V = basis();
        
            return blas::mulvec( V, s );
        }// else
    }
    
    void
    transform_backward  ( const blas::vector< value_t > &  s,
                          blas::vector< value_t > &        v ) const
    {
        #if defined(HLR_HAS_ZBLAS_APLR)
        if ( is_compressed() )
        {
            HLR_ASSERT( v.length() == _is.size() );
            
            compress::aplr::zblas::mulvec( _is.size(), this->rank(), apply_normal, value_t(1), _zV, s.data(), v.data() );
        }// if
        else
        #endif
        {
            auto  V = basis();
        
            blas::mulvec( value_t(1), V, s, value_t(1), v );
        }// else
    }
    
    blas::matrix< value_t >
    transform_backward  ( const blas::matrix< value_t > &  S ) const
    {
        auto  V = basis();
        
        return blas::prod( V, S );
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

        cb->set_id( _id );
        
        if ( is_compressed() )
            cb->_zV = _zV;

        cb->_sv = std::move( blas::copy( _sv ) );

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

        cb->set_id( _id );
        cb->set_nsons( nsons() );

        for ( uint  i = 0; i < nsons(); ++i )
            cb->set_son( i, son(i)->copy_struct().release() );

        return cb;
    }
    
    // return memory consumption
    size_t
    byte_size () const
    {
        size_t  n = ( sizeof(_is) + sizeof(_id) + sizeof(self_t *) * _sons.size() + _V.byte_size() );

        n += compress::aplr::byte_size( _zV );
        n += _sv.byte_size();
        
        for ( auto  son : _sons )
            n += son->byte_size();

        return  n;
    }

    // return size of (floating point) data in bytes handled by this object
    size_t data_byte_size () const
    {
        size_t  n = 0;

        if ( is_compressed() )
            n = hlr::compress::aplr::byte_size( _zV ) + _sv.data_byte_size();
        else
            n = sizeof( value_t ) * _is.size() * rank();

        for ( auto  son : _sons )
            n += son->data_byte_size();

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

    // compress internal data based on given accuracy
    virtual void   compress      ( const accuracy &  acc );

    // decompress internal data
    virtual void   decompress    ();

    // return true if data is compressed
    virtual bool   is_compressed () const
    {
        return ! _zV.empty();
    }

protected:
    // remove compressed storage (standard storage not restored!)
    virtual void   remove_compressed ()
    {
        _zV = compress::aplr::zarray();
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
shared_cluster_basis< value_t >::compress ( const accuracy &  acc )
{
    if ( is_compressed() )
        return;

    if ( _V.nrows() * _V.ncols() == 0 )
        return;

    HLR_ASSERT( _sv.length() == _V.ncols() );
    
    //
    // compute Frobenius norm and set tolerance
    //

    // defaults to absolute error: δ = ε
    auto  lacc = acc( _is, _is ); // TODO: accuracy for just _is
    auto  tol  = lacc.abs_eps();

    if ( lacc.rel_eps() != 0 )
    {
        // use relative error: δ = ε |M|
        real_t  norm = real_t(0);

        if ( lacc.norm_mode() == Hpro::spectral_norm )
            norm = _sv(0);
        else if ( lacc.norm_mode() == Hpro::frobenius_norm )
        {
            for ( uint  i = 0; i < _sv.length(); ++i )
                norm += math::square( _sv(i) );

            norm = math::sqrt( norm );
        }// if
        else
            HLR_ERROR( "unsupported norm mode" );
    
        tol = lacc.rel_eps() * norm;
    }// if
        
    const size_t  mem_dense = sizeof(value_t) * _V.nrows() * _V.ncols();

    //
    // use adaptive precision per basis vector
    //

    HLR_ASSERT( _sv.length() == _V.ncols() );

    // real_t  tol  = acc.abs_eps() * _sv(0);
    auto  S = blas::copy( _sv );

    for ( uint  l = 0; l < S.length(); ++l )
        S(l) = tol / S(l);

    auto  zV   = compress::aplr::compress_lr< value_t >( _V, S );
    auto  zmem = compress::aplr::compressed_size( zV );

    if (( zmem != 0 ) && ( zmem < mem_dense ))
    {
        _zV = std::move( zV );
        _V  = std::move( blas::matrix< value_t >( 0, _V.ncols() ) ); // remember rank
    }// if
}

//
// decompress internal data
//
template < typename value_t >
void
shared_cluster_basis< value_t >::decompress ()
{
    if ( ! is_compressed() )
        return;

    _V = std::move( basis() );

    remove_compressed();
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

////////////////////////////////////////////////////////////////////////////////
//
// level wise representation of shared cluster bases
//

//
// represents hierarchy of shared cluster bases in a level wise way
//
template < typename T_value >
class shared_cluster_basis_hierarchy
{
public:

    using  value_t         = T_value;
    using  real_t          = Hpro::real_type_t< value_t >;
    using  cluster_basis_t = shared_cluster_basis< value_t >;
    
private:
    // basis
    std::vector< std::vector< cluster_basis_t * > >  _hier;
    
public:
    
    // ctor
    shared_cluster_basis_hierarchy ()
    {}

    // ctor with predefined level number
    shared_cluster_basis_hierarchy ( const uint  nlvl )
            : _hier( nlvl )
    {}

    shared_cluster_basis_hierarchy ( shared_cluster_basis_hierarchy &&  hier )
            : _hier( std::move( hier._hier ) )
    {}

    // dtor: delete sons
    ~shared_cluster_basis_hierarchy ()
    {
        // for ( auto  lvl : _hier )
        //     for ( auto  cb : lvl )
        //         delete cb;
    }

    //
    // access sons
    //

    // return number of sons
    uint  nlevel     () const          { return _hier.size(); }

    // set number of sons
    void  set_nlevel ( const uint  n ) { _hier.resize( n ); }

    // access cluster basis at level <lvl> and position <i>
    cluster_basis_t *        cb  ( const uint  lvl,
                                   const uint  i )       { return _hier[lvl][i]; }
    const cluster_basis_t *  cb  ( const uint  lvl,
                                   const uint  i ) const { return _hier[lvl][i]; }

    void  set_cb   ( const uint         lvl,
                     const uint         i,
                     cluster_basis_t *  cb )
    {
        _hier[lvl][i] = cb;
    }

    // return full hierarchy
    auto &        hierarchy ()       { return _hier; }
    const auto &  hierarchy () const { return _hier; }
    
    //
    // compression
    //

    // compress internal data based on given accuracy
    virtual void   compress    ( const accuracy &  acc )
    {
        for ( auto &  lvl : _hier )
            for ( auto  cb : lvl )
                cb->compress( acc );
    }

    // decompress internal data
    virtual void   decompress  ()
    {
        for ( auto &  lvl : _hier )
            for ( auto  cb : lvl )
                cb->decompress();
    }
};

//
// create level wise hierarchy out of tree bases shared cluster basis
//
template < typename value_t >
shared_cluster_basis_hierarchy< value_t >
build_level_hierarchy ( shared_cluster_basis< value_t > &  root_cb )
{
    //
    // traverse in BFS style and construct each level
    //

    auto  hier    = shared_cluster_basis_hierarchy< value_t >( root_cb.depth() );
    auto  current = std::list< shared_cluster_basis< value_t > * >();
    uint  lvl     = 0;

    current.push_back( & root_cb );

    while ( ! current.empty() )
    {
        //
        // add bases on current level to hierarchy
        //

        hier.hierarchy()[lvl].reserve( current.size() );

        for ( auto  cb : current )
            hier.hierarchy()[lvl].push_back( cb );

        //
        // collect bases on next level
        //
        
        auto  sub = std::list< shared_cluster_basis< value_t > * >();
        
        for ( auto  cb : current )
            for ( uint  i = 0; i < cb->nsons(); ++i )
                if ( ! is_null( cb->son(i) ) )
                    sub.push_back( cb->son(i) );

        current = std::move( sub );
        lvl++;
    }// while

    return hier;
}

template < typename value_t >
void
print ( const shared_cluster_basis_hierarchy< value_t > &  hier )
{
    uint  lvl_idx = 0;
    
    for ( auto lvl : hier.hierarchy() )
    {
        uint  idx = 0;
        
        std::cout << lvl_idx++ << std::endl;
        
        for ( auto cb : lvl )
            std::cout << "    " << idx++ << " : " << cb->is() << std::endl;
    }// for
}

//
// return min/avg/max rank of given cluster basis
//
template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const shared_cluster_basis_hierarchy< value_t > &  cb_hier )
{
    uint    min_rank = 0;
    uint    max_rank = 0;
    size_t  sum_rank = 0;
    size_t  nnodes   = 0;
    
    for ( auto &  lvl : cb_hier.hierarchy() )
    {
        for ( auto  cb : lvl )
        {
            const auto  rank = cb.rank();

            if ( rank > 0 )
            {
                if ( min_rank == 0 ) min_rank = rank;
                else                 min_rank = std::min( min_rank, rank );
            
                max_rank  = std::max( max_rank, rank );
                sum_rank += rank;
                ++nnodes;
            }// if
        }// for
    }// for

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}


}} // namespace hlr::matrix

#endif // __HLR_MATRIX_SHARED_CLUSTER_BASIS_HH
