//
// Project     : HLib
// File        : tileh-seq.cc
// Description : sequential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "tileh.hh"
#include "matrixbuild.hh"

namespace TileH
{

namespace SEQ
{

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
    auto  nbr = A->nblock_rows();
    auto  nbc = A->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        HLIB::LU::factorise_rec( A->block( i, i ), acc );

        for ( uint j = i+1; j < nbr; ++j )
        {
            solve_upper_right( A->block( j, i ),
                               A->block( i, i ), nullptr, acc,
                               solve_option_t( block_wise, general_diag, store_inverse ) );
        }// for

        for ( uint  l = i+1; l < nbc; ++l )
        {
            solve_lower_left( apply_normal, A->block( i, i ), nullptr,
                              A->block( i, l ), acc,
                              solve_option_t( block_wise, unit_diag, store_inverse ) );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                multiply( -1.0, A->block( j, i ), A->block( i, l ), 1.0, A->block( j, l ), acc );
            }// for
        }// for
    }// for
}

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    assert( is_blocked( A ) );

    lu< value_t >( ptrcast( A, TBlockMatrix ), acc );
}

}// namespace SEQ

}// namespace TILEH

//
// main function
//
void
mymain ( int argc, char ** argv )
{
    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = TileH::cluster( coord.get(), ntile, 4 );

    if ( verbose( 3 ) )
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if
    
    auto  A   = problem->build_matrix( bct.get(), fixed_rank( k ) );
    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "tileh_A" );
    }// if
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( Tile-H SEQ )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TileH::SEQ::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}
