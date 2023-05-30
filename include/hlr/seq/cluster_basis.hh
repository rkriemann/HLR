#ifndef __HLR_SEQ_CLUSTER_BASIS_HH
#define __HLR_SEQ_CLUSTER_BASIS_HH
//
// Project     : HLR
// Module      : seq/cluster_basis
// Description : sequential functions for cluster bases
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <unordered_map>

#include <hpro/matrix/TRkMatrix.hh>

#include <hlr/utils/log.hh>
#include <hlr/utils/hash.hh>
#include <hlr/arith/blas.hh>
#include <hlr/matrix/shared_cluster_basis.hh>

namespace hlr { namespace seq { namespace matrix {

using namespace hlr::matrix;

//
// forward decl.
//
namespace detail
{

// map HLIBpro types to HLR 
using  indexset = Hpro::TIndexSet;

// mapping of clusters/indexsets to corresponding matrix blocks
template < typename value_t >
using  lrmatrix_map_t = std::unordered_map< indexset, std::list< const Hpro::TRkMatrix< value_t > * >, indexset_hash >;

template < typename value_t >
void
build_lrmatrix_map ( const cluster_tree &              ct,
                     const Hpro::TMatrix< value_t > &  M,
                     lrmatrix_map_t< value_t > &       mat_map,
                     const bool                        adjoint );

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
construct_from_H ( const cluster_tree &   rowct,
                   const cluster_tree &   colct,
                   const Hpro::TMatrix< value_t > &  M,
                   const accuracy &       acc )
{
    //
    // first construct mapping from individual clusters to
    // set of associated matrix blocks in H-matrix
    //

    detail::lrmatrix_map_t< value_t >  row_map, col_map;
    
    detail::build_lrmatrix_map( rowct, M, row_map, false );
    detail::build_lrmatrix_map( colct, M, col_map, true  );

    //
    // next, construct cluster basis for each cluster in cluster tree
    //

    auto  row_cb = detail::construct_basis< value_t >( rowct, row_map, acc, false );
    auto  col_cb = detail::construct_basis< value_t >( colct, col_map, acc, true  );

    return { std::move( row_cb ), std::move( col_cb ) };
}

namespace detail
{

//
// access factors U/V or low-rank matrices
//
template < typename value_t >
const blas::matrix< value_t > &
U ( const Hpro::TRkMatrix< value_t > *  M,
    const bool               adjoint )
{
    if ( adjoint ) return Hpro::blas_mat_B< value_t >( M );
    else           return Hpro::blas_mat_A< value_t >( M );
}

template < typename value_t >
const blas::matrix< value_t > &
V ( const Hpro::TRkMatrix< value_t > *  M,
    const bool               adjoint )
{
    if ( adjoint ) return Hpro::blas_mat_A< value_t >( M );
    else           return Hpro::blas_mat_B< value_t >( M );
}

//
// construct map from index sets to matrix blocks for row clusters
//
template < typename value_t >
void
build_lrmatrix_map ( const cluster_tree &              ct,
                     const Hpro::TMatrix< value_t > &  M,
                     lrmatrix_map_t< value_t > &       mat_map,
                     const bool                        adjoint )
{
    HLR_ASSERT( ct == M.row_is( adjoint ? Hpro::apply_transposed : Hpro::apply_normal ) );
    
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        if ( adjoint )
        {
            HLR_ASSERT( ct.nsons() == B->nblock_cols() );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  ct_j = ct.son( j );
            
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    auto  B_ij = B->block( i, j );
                
                    if ( ! is_null( B_ij ) )
                        build_lrmatrix_map( *ct_j, *B_ij, mat_map, adjoint );
                }// for
            }// for
        }// if
        else
        {
            HLR_ASSERT( ct.nsons() == B->nblock_rows() );

            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  ct_i = ct.son( i );
            
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  B_ij = B->block( i, j );
                
                    if ( ! is_null( B_ij ) )
                        build_lrmatrix_map( *ct_i, *B_ij, mat_map, adjoint );
                }// for
            }// for
        }// else
    }// if
    else if ( is_lowrank( M ) )
    {
        // std::cout << ct.to_string() << " : " << M.block_is().to_string() << std::endl;
        mat_map[ ct ].push_back( cptrcast( &M, Hpro::TRkMatrix< value_t > ) );
    }// if
    else if ( is_dense( M ) )
    {
        //
        // nearfield is ignored
        //
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// construct cluster basis for each cluster (block rows)
//
// For a cluster t let M = { M' : row_cluster( M ) = t } be the set of all
// low-rank matrices in the block row defined by t.
//
// Construct total cluster basis X_t:
//
//    X_t = [ X₁, X₂, … ] = [ U₁·C₁', U₂·C₂',  … ]
//
// with C_i from  Q_i C_i = qr( V_i )
//
// The cluster basis W_t is then defined as
//
//    W_t R_t = qr( X_t )
//
// Here, "qr" may be replaced by any orthonormal factorization.
//
template < typename value_t >
std::unique_ptr< shared_cluster_basis< value_t > >
construct_basis ( const cluster_tree &         ct,
                  lrmatrix_map_t< value_t > &  mat_map,
                  const accuracy &             acc,
                  const bool                   adjoint )
{
    auto  cb = std::make_unique< shared_cluster_basis< value_t > >( ct );

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
                auto  [ Q, C ] = blas::factorise_ortho( V< value_t >( M, adjoint ) );

                condensed_mat.push_back( std::move( C ) );
                rank_sum += M->rank();
            }// if
        }// for

        if ( rank_sum > 0 )
        {
            //
            // build X_t, the total cluster basis
            //
            //  X_t = [ U₀·C₀', U₁·C₁', … ]
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
                    auto  X_i   = blas::prod( U< value_t >( M, adjoint ), blas::adjoint( C ) );
                    auto  X_sub = blas::matrix< value_t >( Xt,
                                                           blas::range::all,
                                                           blas::range( pos, pos + X_i.ncols() - 1 ) );
                    
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

        for ( uint  i = 0; i < ct.nsons(); ++i )
        {
            auto  cb_i = construct_basis< value_t >( *ct.son(i), mat_map, acc, adjoint );
            
            cb->set_son( i, cb_i.release() );
        }// for
    }// if

    return cb;
}

}// namespace detail

}}} // namespace hlr::seq::matrix

#endif // __HLR_SEQ_CLUSTER_BASIS_HH
