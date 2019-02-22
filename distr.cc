//
// Project     : HLR-HPC
// File        : distr.cc
// Description : cluster tree distribution functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <cassert>

#include "utils/tensor.hh"

#include "distr.hh"

namespace distribution
{

using namespace HLIB;

//
// assigns 2d cyclic distribution
//
void
cyclic_2d ( const uint       nprocs,
            TBlockCluster *  bct )
{
    //
    // grid dimension in block layout
    //

    const uint  P = uint( std::sqrt(nprocs) );
    const uint  Q = uint( nprocs / P );

    assert( P*Q == nprocs );

    //
    // first set all to all processors (so the upper levels will stay on all)
    //

    bct->set_procs( ps( nprocs ), true );
    
    //
    // look for level in bct with sufficient nodes
    //

    std::list< TBlockCluster * >  nodes;

    nodes.push_back( bct );

    while ( nodes.size() < 2*nprocs )
    {
        std::list< TBlockCluster * >  sons;

        for ( auto node : nodes )
        {
            for ( uint  i = 0; i < node->nsons(); ++i )
                sons.push_back( node->son( i ) );
        }// for

        nodes = std::move( sons );
    }// while

    //
    // loop through nodes and count block rows/columns
    //

    std::list< TIndexSet >  block_rows, block_cols;

    for ( auto node : nodes )
    {
        const TIndexSet  row_is = *( node->rowcl() );
        const TIndexSet  col_is = *( node->colcl() );

        if ( find( block_rows.begin(), block_rows.end(), row_is ) == block_rows.end() )
            block_rows.push_back( row_is );

        if ( find( block_cols.begin(), block_cols.end(), col_is ) == block_cols.end() )
            block_cols.push_back( col_is );
    }// for

    block_rows.sort( [] ( const TIndexSet & is1, const TIndexSet & is2 ) { return is1.is_strictly_left_of( is2 ); } );
    block_cols.sort( [] ( const TIndexSet & is1, const TIndexSet & is2 ) { return is1.is_strictly_left_of( is2 ); } );

    const uint  nrows = block_rows.size();
    const uint  ncols = block_cols.size();
    
    //
    // set processor based on index in block grid
    //

    for ( auto node : nodes )
    {
        const TIndexSet  row_is = *( node->rowcl() );
        const TIndexSet  col_is = *( node->colcl() );

        uint  i = 0;
        uint  j = 0;

        for ( const auto & br : block_rows )
        {
            if ( br == row_is )
                break;
            ++i;
        }// for
        
        for ( const auto & bc : block_cols )
        {
            if ( bc == col_is )
                break;
            ++j;
        }// for

        const uint  i1   = i % P;
        const uint  j1   = j % Q;
        const uint  proc = i1*Q + j1;

        // set block (with subtree) to local processor only
        node->set_procs( ps_single( proc ), true );
    }// for
}

//
// assigns 2d cyclic distribution
//
void
shifted_cyclic_1d ( const uint       nprocs,
                    TBlockCluster *  bct )
{
    //
    // first set all to all processors (so the upper levels will stay on all)
    //

    bct->set_procs( ps( nprocs ), true );
    
    //
    // look for level in bct with sufficient nodes
    //

    std::list< TBlockCluster * >  nodes;

    nodes.push_back( bct );

    while ( nodes.size() < 2*nprocs )
    {
        std::list< TBlockCluster * >  sons;

        for ( auto node : nodes )
        {
            for ( uint  i = 0; i < node->nsons(); ++i )
                sons.push_back( node->son( i ) );
        }// for

        nodes = std::move( sons );
    }// while

    //
    // loop through nodes and count block rows/columns
    //

    std::list< TIndexSet >  block_rows, block_cols;

    for ( auto node : nodes )
    {
        const TIndexSet  row_is = *( node->rowcl() );
        const TIndexSet  col_is = *( node->colcl() );

        if ( find( block_rows.begin(), block_rows.end(), row_is ) == block_rows.end() )
            block_rows.push_back( row_is );

        if ( find( block_cols.begin(), block_cols.end(), col_is ) == block_cols.end() )
            block_cols.push_back( col_is );
    }// for

    block_rows.sort( [] ( const TIndexSet & is1, const TIndexSet & is2 ) { return is1.is_strictly_left_of( is2 ); } );
    block_cols.sort( [] ( const TIndexSet & is1, const TIndexSet & is2 ) { return is1.is_strictly_left_of( is2 ); } );

    const uint  nrows = block_rows.size();
    const uint  ncols = block_cols.size();
    
    //
    // set up 2D grid of nodes
    //

    tensor2< TBlockCluster * >  blocks2d( nrows, ncols );
    
    for ( auto node : nodes )
    {
        const TIndexSet  row_is = *( node->rowcl() );
        const TIndexSet  col_is = *( node->colcl() );

        uint  i = 0;
        uint  j = 0;

        for ( const auto & br : block_rows )
        {
            if ( br == row_is )
                break;
            ++i;
        }// for
        
        for ( const auto & bc : block_cols )
        {
            if ( bc == col_is )
                break;
            ++j;
        }// for

        blocks2d( i, j ) = node;
    }// for

    //
    // for each row, apply 1d cyclic with offset per row
    //

    for ( uint  i = 0; i < nrows; ++i )
        for ( uint  j = 0; j < ncols; ++j )
            blocks2d( i, j )->set_procs( ps_single( ( j + i ) % nprocs ), true );
}

}// namespace distribution
