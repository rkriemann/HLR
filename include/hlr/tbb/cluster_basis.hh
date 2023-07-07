#ifndef __HLR_TBB_CLUSTER_BASIS_HH
#define __HLR_TBB_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : tbb/cluster_basis
// Description : functions for cluster bases using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <hlr/utils/log.hh>
#include <hlr/matrix/shared_cluster_basis.hh>
#include <hlr/seq/cluster_basis.hh>

namespace hlr { namespace tbb { namespace matrix {

using namespace hlr::matrix;

//
// forward decl.
//
namespace detail
{

using hlr::seq::matrix::detail::lrmatrix_map_t;
using hlr::seq::matrix::detail::build_lrmatrix_map;

template < typename value_t >
std::unique_ptr< shared_cluster_basis< value_t > >
construct_basis ( const cluster_tree &         ct,
                  lrmatrix_map_t< value_t > &  mat_map,
                  const accuracy &             acc,
                  const bool                   adjoint );

}// namespace detail

//
// construct cluster basis for given row/column cluster trees and H matrix
// - only low-rank matrices will contribute to cluster bases
// - cluster bases are not nested
//
template < typename value_t >
std::pair< std::unique_ptr< shared_cluster_basis< value_t > >,
           std::unique_ptr< shared_cluster_basis< value_t > > >
construct_from_H ( const cluster_tree &              rowct,
                   const cluster_tree &              colct,
                   const Hpro::TMatrix< value_t > &  M,
                   const accuracy &                  acc )
{
    
    //
    // first construct mapping from individual clusters to
    // set of associated matrix blocks in H-matrix
    //

    detail::lrmatrix_map_t< value_t >  row_map, col_map;

    ::tbb::parallel_invoke( [&] { detail::build_lrmatrix_map( rowct, M, row_map, false ); },
                            [&] { detail::build_lrmatrix_map( colct, M, col_map, true  ); } );

    //
    // next, construct cluster basis for each cluster in cluster tree
    //

    std::unique_ptr< shared_cluster_basis< value_t > >  row_cb, col_cb;

    ::tbb::parallel_invoke( [&] { row_cb = detail::construct_basis< value_t >( rowct, row_map, acc, false ); },
                            [&] { col_cb = detail::construct_basis< value_t >( colct, col_map, acc, true  ); } );

    return { std::move( row_cb ), std::move( col_cb ) };
}

namespace detail
{

//
// construct cluster basis for each cluster (block rows)
//
template < typename value_t >
std::unique_ptr< shared_cluster_basis< value_t > >
construct_basis ( const cluster_tree &         ct,
                  lrmatrix_map_t< value_t > &  mat_map,
                  const accuracy &             acc,
                  const bool                   adjoint )
{
    auto  cb = std::make_unique< shared_cluster_basis< value_t > >( ct );
    auto  op = ( adjoint ? apply_adjoint : apply_normal );

    //
    // compute row basis for all blocks
    //

    if ( ! mat_map[ ct ].empty() )
    {
        //
        // first, construct column basis for each block and store coefficients, e.g.,
        // for M, compute M = U·V' = U·(P·C)' with orthogonal P and store C
        //

        auto  condensed_mat = std::list< blas::matrix< value_t > >();
        uint  rank_sum      = 0;
        
        for ( auto  M : mat_map[ ct ] )
        {
            if ( M->rank() > 0 )
            {
                auto  [ Q, C ] = blas::factorise_ortho( M->V( op ) );

                condensed_mat.push_back( std::move( C ) );
                rank_sum += M->rank();
            }// if
        }// for

        if ( rank_sum > 0 )
        {
            //
            // build X_t, the total cluster basis
            //
            //  X_t = [ U₀·C₀', U₁·C₁', ... ]
            //
            
            auto   Xt     = blas::matrix< value_t >( ct.size(), rank_sum );
            auto   iter_M = mat_map[ ct ].begin();
            auto   iter_C = condensed_mat.begin();
            idx_t  pos    = 0;

            while ( iter_C != condensed_mat.end() )
            {
                auto  M = *iter_M;
                auto  C = *iter_C;

                if ( M->rank() > 0 )
                {
                    auto  X_i    = blas::prod( value_t(1), M->U( op ), blas::adjoint( C ) );
                    auto  cols_i = blas::range( pos, pos + X_i.ncols() - 1 );
                    auto  X_sub  = blas::matrix< value_t >( Xt, blas::range::all, cols_i );

                    blas::copy( X_i, X_sub );

                    pos += X_i.ncols();
                    ++iter_C;
                }// if

                ++iter_M;
            }// while

            //
            // approximate basis up to given accuracy and update cluster basis
            //

            auto  [ Q, R ] = blas::factorise_ortho( Xt, acc );

            cb->set_basis( std::move( Q ) );
        }// if
    }// if

    //
    // recurse
    //

    if ( ct.nsons() > 0 )
    {
        cb->set_nsons( ct.nsons() );

        ::tbb::parallel_for( uint(0), ct.nsons(),
                             [&,adjoint] ( const uint  i )
                             {
                                 auto  cb_i = construct_basis< value_t >( *ct.son(i), mat_map, acc, adjoint );
            
                                 cb->set_son( i, cb_i.release() );
                             } );
    }// if

    return cb;
}

}// namespace detail

}}} // namespace hlr::tbb::matrix

#endif // __HLR_TBB_CLUSTER_BASIS_HH
