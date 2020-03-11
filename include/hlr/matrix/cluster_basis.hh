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
    // associated cluster
    const cluster_tree *                       _cluster;

    // cluster basis of sub clusters
    std::vector< cluster_basis< value_t > * >  _sons;
    
    // basis
    hlr::blas::matrix< value_t >               _V;

public:
    
    // construct cluster basis corresponding to cluster <cl>
    cluster_basis ( const cluster_tree &  cl )
            : _cluster( &cl )
    {}

    // construct cluster basis corresponding to cluster <cl>
    // with basis defined by <V>
    cluster_basis ( const cluster_tree &                   cl,
                    const hlr::blas::matrix< value_t > &&  V )
            : _cluster( &cl )
            , _V( V )
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

    // underlying cluster tree
    const cluster_tree &                  cluster () const { return *_cluster; }
    
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
        return blas::mulvec( value_t(1), hlr::blas::adjoint( _V ), v );
    }
    
    hlr::blas::matrix< value_t >
    transform_forward  ( const hlr::blas::matrix< value_t > &  M ) const
    {
        return blas::prod( value_t(1), hlr::blas::adjoint( _V ), M );
    }
    
    //
    // backward transformation, e.g. compute vector/matrix in standard basis
    // - return V · arg
    //
    
    hlr::blas::vector< value_t >
    transform_backward  ( const hlr::blas::vector< value_t > &  s ) const
    {
        return hlr::blas::mulvec( value_t(1), _V, s );
    }
    
    hlr::blas::matrix< value_t >
    transform_backward  ( const hlr::blas::matrix< value_t > &  S ) const
    {
        return hlr::blas::prod( value_t(1), _V, S );
    }

    ///////////////////////////////////////////////////////
    //
    // misc.
    //

    // return memory consumption
    size_t
    byte_size () const
    {
        size_t  n = ( sizeof(_cluster) +
                      sizeof(cluster_basis< value_t > *) * _sons.size() +
                      sizeof(_V) + sizeof(value_t) * _V.nrows() * _V.ncols() );

        for ( auto  son : _sons )
            n += son->byte_size();

        return  n;
    }
};

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_CLUSTER_BASIS_HH
