//
// Project     : HLR
// Module      : cluster_basis
// Description : (non-nested) cluster basis
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <unordered_map>

#include <hlr/utils/log.hh>

#include <hlr/matrix/cluster_basis.hh>

namespace hlr { namespace matrix {

using namespace hpro;

namespace
{

// map HLIB types to HLR 
using  indexset = hpro::TIndexSet;
using  range    = blas::Range;

// hash function for index sets (for mapping below)
struct indexset_hash
{
    size_t operator () ( const indexset &  is ) const
    {
        return ( std::hash< idx_t >()( is.first() ) ^
                 std::hash< idx_t >()( is.last()  ) );
    }
};

// mapping of clusters/indexsets to corresponding matrix blocks
using  matrix_map_t = std::unordered_map< indexset, std::list< const TRkMatrix * >, indexset_hash >;

//
// construct map from index sets to matrix blocks for row clusters
//
void
build_row_matrix_map ( const clustertree &  ct,
                       const TMatrix &      M,
                       matrix_map_t &       mat_map )
{
    HLR_ASSERT( ct == M.row_is() );
    
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, TBlockMatrix );

        HLR_ASSERT( ct.nsons() == B->nblock_rows() );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  ct_i = ct.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                    build_row_matrix_map( *ct_i, *B_ij, mat_map );
            }// for
        }// for
    }// if
    else if ( is_lowrank( M ) )
    {
        mat_map[ ct ].push_back( cptrcast( &M, TRkMatrix ) );
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
// construct map from index sets to matrix blocks for column clusters
//
void
build_col_matrix_map ( const clustertree &  ct,
                       const TMatrix &      M,
                       matrix_map_t &       mat_map )
{
    HLR_ASSERT( ct == M.col_is() );
    
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, TBlockMatrix );

        HLR_ASSERT( ct.nsons() == B->nblock_cols() );

        for ( uint  j = 0; j < B->nblock_cols(); ++j )
        {
            auto  ct_j = ct.son( j );
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                    build_col_matrix_map( *ct_j, *B_ij, mat_map );
            }// for
        }// for
    }// if
    else if ( is_lowrank( M ) )
    {
        mat_map[ ct ].push_back( cptrcast( &M, TRkMatrix ) );
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
template < typename value_t >
std::unique_ptr< cluster_basis< value_t > >
construct_row_basis ( const clustertree &  ct,
                      matrix_map_t &       mat_map,
                      const accuracy &     acc )
{
    auto  cb = std::make_unique< cluster_basis< value_t > >( ct );

    //
    // compute row basis for all blocks
    //

    if ( ! mat_map[ ct ].empty() )
    {
        //
        // first, construct column basis for each block and store coefficients, e.g.,
        // for M, compute M = U·V^H = U·(P·C)^H with orthogonal P and store C
        //

        std::list< blas::Matrix< value_t > >  condensed_mat;
        uint                                  rank_sum = 0;
        
        for ( auto  M : mat_map[ ct ] )
        {
            if ( M->rank() > 0 )
            {
                auto  P = blas::copy( blas_mat_B< value_t >( M ) );
                auto  C = blas::Matrix< value_t >( M->rank(), M->rank() );
                
                blas::factorise_ortho( P, C );

                condensed_mat.push_back( std::move( C ) );
                rank_sum += M->rank();
            }// if
        }// for

        if ( rank_sum > 0 )
        {
            //
            // build X_t, the total cluster basis
            //
            //  X_t = [ U₀·C₀^H, U₁·C₁^H, ... ]
            //
            
            blas::Matrix< value_t >  Xt( ct.size(), rank_sum );

            auto   iter_M = mat_map[ ct ].begin();
            auto   iter_C = condensed_mat.begin();
            idx_t  pos    = 0;

            while ( iter_C != condensed_mat.end() )
            {
                auto  M = *iter_M;
                auto  C = *iter_C;

                if ( M->rank() > 0 )
                {
                    auto  X_i    = blas::prod( value_t(1), blas_mat_A< value_t >( M ), blas::adjoint( C ) );
                    auto  cols_i = range( pos, pos + X_i.ncols() - 1 );
                    auto  X_sub  = blas::Matrix< value_t >( Xt, range::all, cols_i );

                    blas::copy( X_i, X_sub );

                    pos += X_i.ncols();
                    ++iter_C;
                }// if

                ++iter_M;
            }// while

            //
            // approximate basis up to given accuracy and update cluster basis
            //

            blas::Matrix< value_t >  R;
            
            blas::factorise_ortho( Xt, R, acc );

            cb->set_basis( std::move( Xt ) );
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
            auto  cb_i = construct_row_basis< value_t >( *ct.son(i), mat_map, acc );
            
            cb->set_son( i, cb_i.release() );
        }// for
    }// if

    return cb;
}

//
// construct cluster basis for each cluster (block rows)
//
template < typename value_t >
std::unique_ptr< cluster_basis< value_t > >
construct_col_basis ( const clustertree &  ct,
                      matrix_map_t &       mat_map,
                      const accuracy &     acc )
{
    auto  cb = std::make_unique< cluster_basis< value_t > >( ct );

    //
    // compute row basis for all blocks
    //

    if ( ! mat_map[ ct ].empty() )
    {
        //
        // first, construct row basis for each block and store coefficients, e.g.,
        // for M, compute M = U·V^H = (P·C)·V^H with orthogonal P and store C
        //

        std::list< blas::Matrix< value_t > >  condensed_mat;
        uint                                  rank_sum = 0;
        
        for ( auto  M : mat_map[ ct ] )
        {
            if ( M->rank() > 0 )
            {
                auto  P = blas::copy( blas_mat_A< value_t >( M ) );
                auto  C = blas::Matrix< value_t >( M->rank(), M->rank() );
                
                blas::factorise_ortho( P, C );

                condensed_mat.push_back( std::move( C ) );
                rank_sum += M->rank();
            }// if
        }// for

        if ( rank_sum > 0 )
        {
            //
            // build X_t, the total cluster basis
            //
            //  X_t = [ V₀·C₀^H, V₁·C₁^H, ... ]
            //
            
            blas::Matrix< value_t >  Xt( ct.size(), rank_sum );

            auto   iter_M = mat_map[ ct ].begin();
            auto   iter_C = condensed_mat.begin();
            idx_t  pos    = 0;

            while ( iter_C != condensed_mat.end() )
            {
                auto  M = *iter_M;
                auto  C = *iter_C;

                if ( M->rank() > 0 )
                {
                    auto  X_i    = blas::prod( value_t(1), blas_mat_B< value_t >( M ), blas::adjoint( C ) );
                    auto  cols_i = range( pos, pos + X_i.ncols() - 1 );
                    auto  X_sub  = blas::Matrix< value_t >( Xt, range::all, cols_i );

                    blas::copy( X_i, X_sub );

                    pos += X_i.ncols();
                    ++iter_C;
                }// if

                ++iter_M;
            }// while

            //
            // approximate basis up to given accuracy and update cluster basis
            //

            blas::Matrix< value_t >  R;
            
            blas::factorise_ortho( Xt, R, acc );

            cb->set_basis( std::move( Xt ) );
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
            auto  cb_i = construct_col_basis< value_t >( *ct.son(i), mat_map, acc );
            
            cb->set_son( i, cb_i.release() );
        }// for
    }// if

    return cb;
}

}// namespace anonymous

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
                   const accuracy &       acc )
{
    //
    // first construct mapping from individual clusters to
    // set of associated matrix blocks in H-matrix
    //

    matrix_map_t  row_map, col_map;
    
    build_row_matrix_map( rowct, M, row_map );
    build_col_matrix_map( colct, M, col_map );

    //
    // construct cluster basis for each cluster in cluster tree
    //

    auto  row_cb = construct_row_basis< value_t >( rowct, row_map, acc );
    auto  col_cb = construct_col_basis< value_t >( colct, col_map, acc );

    return { std::move( row_cb ), std::move( col_cb ) };
}

//
// explicit instantiations
//

template
std::pair< std::unique_ptr< cluster_basis< hpro::real > >,
           std::unique_ptr< cluster_basis< hpro::real > > >
construct_from_H< hpro::real > ( const clustertree &    rowct,
                                 const clustertree &    colct,
                                 const hpro::TMatrix &  M,
                                 const accuracy &       acc );

template
std::pair< std::unique_ptr< cluster_basis< hpro::complex > >,
           std::unique_ptr< cluster_basis< hpro::complex > > >
construct_from_H< hpro::complex > ( const clustertree &    rowct,
                                    const clustertree &    colct,
                                    const hpro::TMatrix &  M,
                                    const accuracy &       acc );

}}// namespace hlr::matrix
