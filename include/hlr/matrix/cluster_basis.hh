#ifndef __HLR_MATRIX_CLUSTER_BASIS_HH
#define __HLR_MATRIX_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : matrix/cluster_basis
// Description : (non-nested) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TCluster.hh>

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

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_CLUSTER_BASIS_HH
