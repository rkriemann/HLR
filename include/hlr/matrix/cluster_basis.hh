#ifndef __HLR_MATRIX_CLUSTER_BASIS_HH
#define __HLR_MATRIX_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : cluster_basis
// Description : (non-nested) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <cassert>
#include <map>

#include <hpro/cluster/TCluster.hh>

#include <hlr/utils/checks.hh>

namespace hlr
{ 

namespace hpro = HLIB;
namespace blas = hpro::BLAS;

using clustertree = hpro::TCluster;
using accuracy    = hpro::TTruncAcc;

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
    const clustertree &                        _cluster;

    // cluster basis of sub clusters
    std::vector< cluster_basis< value_t > * >  _sons;
    
    // basis
    blas::Matrix< value_t >                    _V;

public:
    
    // construct cluster basis corresponding to cluster <cl>
    cluster_basis ( const clustertree &  cl )
            : _cluster( cl )
    {}

    // construct cluster basis corresponding to cluster <cl>
    // with basis defined by <V>
    cluster_basis ( const clustertree &               cl,
                    const blas::Matrix< value_t > &&  V )
            : _cluster( cl )
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

    // return rank of cluster basis
    uint                             rank   () const  { return _V.ncols(); }
    
    // return local basis
    const blas::Matrix< value_t > &  basis  () const { return _V; }
    
    // set local basis
    void
    set_basis  ( const blas::Matrix< value_t > &  aV )
    {
        _V = std::move( blas::copy( aV ) );
    }
    
    void
    set_basis  ( blas::Matrix< value_t > &&  aV )
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
    
    blas::Vector< value_t >
    transform_forward  ( const blas::Vector< value_t > &  v ) const
    {
        return blas::mulvec( value_t(1), blas::adjoint( _V ), v );
    }
    
    blas::Matrix< value_t >
    transform_forward  ( const blas::Matrix< value_t > &  M ) const
    {
        return blas::prod( value_t(1), blas::adjoint( _V ), M );
    }
    
    //
    // backward transformation, e.g. compute vector/matrix in standard basis
    // - return V · arg
    //
    
    blas::Vector< value_t >
    transform_backward  ( const blas::Vector< value_t > &  s ) const
    {
        return blas::mulvec( value_t(1), _V, s );
    }
    
    blas::Matrix< value_t >
    transform_backward  ( const blas::Matrix< value_t > &  S ) const
    {
        return blas::prod( value_t(1), _V, S );
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
                      sizeof(value_t) * _V.nrows() * _V.ncols() );

        for ( auto  son : _sons )
            n += son->byte_size();

        return  n;
    }
};

//
// construct cluster basis for given row/column cluster trees and H matrix
// - only low-rank matrices will contribute to cluster bases
// - cluster bases are not nested
//
template < typename value_t >
std::pair< std::unique_ptr< cluster_basis< value_t > >,
           std::unique_ptr< cluster_basis< value_t > > >
construct_from_H ( const clustertree &    rowct,
                   const clustertree &    colct,
                   const hpro::TMatrix &  M,
                   const accuracy &       acc );

}} // namespace hlr::matrix

#endif // __HLR_MATRIX_CLUSTER_BASIS_HH
