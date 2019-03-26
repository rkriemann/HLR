#ifndef __HLR_MPI_MATRIX_HH
#define __HLR_MPI_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <vector>
#include <list>
#include <unordered_map>
#include <type_traits>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <matrix/TGhostMatrix.hh>
#include <base/TTruncAcc.hh>

#include "utils/tools.hh"
#include "seq/matrix.hh"
#include "tbb/matrix.hh"
#include "mpi/mpi.hh"

namespace HLR
{

namespace Matrix
{
    
namespace MPI
{

//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< HLIB::TMatrix >
build ( const HLIB::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const HLIB::TTruncAcc &      acc )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                   typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );

    using  value_t = typename coeff_t::value_t;
    
    assert( bct != nullptr );

    //
    // decide upon cluster type, how to construct matrix
    //

    mpi::communicator           world;
    const auto                  pid    = world.rank();
    
    std::unique_ptr< TMatrix >  M;
    const auto                  rowis = bct->is().row_is();
    const auto                  colis = bct->is().col_is();

    // parallel handling too inefficient for small matrices
    if ( std::max( rowis.size(), colis.size() ) <= 0 )
        return Matrix::Seq::build( bct, coeff, lrapx, acc );

    if ( ! bct->procs().contains( pid ) )
    {
        M = std::make_unique< TGhostMatrix >( bct->is(), bct->procs(), value_type< value_t >::value );
    }// if
    else if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M.reset( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( rowis, colis );
        }// else
    }// if
    else
    {
        auto  B = std::make_unique< TBlockMatrix >( bct );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        tbb::blocked_range2d< uint >  r( 0, B->nblock_rows(),
                                         0, B->nblock_cols() );
        
        // tbb::parallel_for( r,
        //     [&,bct] ( const tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( bct->son( i, j ) != nullptr )
                        {
                            auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            }// );

        M = std::move( B );
    }// else

    // copy properties from the cluster
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// create communicator for each block row of the matrix
// - if processor sets of different rows are identical, so are the communicators
//
void
build_row_comms ( const TBlockMatrix *                             A,
                  std::vector< mpi::communicator > &               row_comms,  // communicator per row
                  std::vector< std::list< int > > &                row_procs,  // set of processors per row
                  std::vector< std::unordered_map< int, int > > &  row_maps )  // mapping of global rank to per communicator rank
{
    const auto         nbr = A->nblock_rows();
    const auto         nbc = A->nblock_cols();
    mpi::communicator  world;
    const auto         pid    = world.rank();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        std::list< int >  procs;
        
        for ( uint  j = 0; j < nbc; ++j )
            procs.push_back( A->block( i, j )->procs().master() );

        procs.sort();
        procs.unique();

        uint  pos = nbr;
            
        for ( uint  l = 0; l < i; ++l )
        {
            if ( procs == row_procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for

        if ( HLIB::verbose( 4 ) )
            std::cout << i << " : " << to_string( procs ) << " (" << ( pos == nbr ? i : pos ) << ")" << std::endl;
            
        // use previously created communicator or create new if none found
        if ( pos < nbr )
        {
            row_comms[i] = row_comms[pos];
            row_maps[i]  = row_maps[pos];
        }// if
        else
        {
            row_comms[i] = world.split( contains( procs, pid ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            
            for ( auto p : procs )
                row_maps[i][p] = comm_rank++;
        }// else
            
        row_procs[i] = std::move( procs );
    }// for
}

//
// create communicator for each block column of the matrix
// - if processor sets of different columns are identical, so are the communicators
//
void
build_col_comms ( const TBlockMatrix *                             A,
                  std::vector< mpi::communicator > &               col_comms,  // communicator per column                           
                  std::vector< std::list< int > > &                col_procs,  // set of processors per column                      
                  std::vector< std::unordered_map< int, int > > &  col_maps )  // mapping of global rank to per communicator rank
{
    const auto         nbr = A->nblock_rows();
    const auto         nbc = A->nblock_cols();
    mpi::communicator  world;
    const auto         pid    = world.rank();

    for ( uint  j = 0; j < nbc; ++j )
    {
        std::list< int >  procs;
            
        for ( uint  i = 0; i < nbr; ++i )
            procs.push_back( A->block( i, j )->procs().master() );

        procs.sort();
        procs.unique();

        uint  pos = nbc;
            
        for ( uint  l = 0; l < j; ++l )
        {
            if ( procs == col_procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for

        if ( HLIB::verbose( 4 ) )
            std::cout << j << " : " << to_string( procs ) << " (" << ( pos == nbc ? j : pos ) << ")" << std::endl;

        // use previously created communicator or create new if none found
        if ( pos < nbc )
        {
            col_comms[j] = col_comms[pos];
            col_maps[j]  = col_maps[pos];
        }// if
        else
        {
            col_comms[j] = world.split( contains( procs, pid ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            
            for ( auto p : procs )
                col_maps[j][p] = comm_rank++;
        }// else
            
        col_procs[j] = std::move( procs );
    }// for
}

}// namespace MPI

}// namespace Matrix

}// namespace HLR

#endif // __HLR_MPI_MATRIX_HH