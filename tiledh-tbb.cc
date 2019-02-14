//
// Project     : HLib
// File        : tiledh-tbb.cc
// Description : Tiled-H arithmetic with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include "common.inc"
#include "tiledh.hh"

namespace TiledH
{

namespace TBB
{

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );
    
    auto  nbr = A->nblock_rows();
    auto  nbc = A->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        HLIB::LU::factorise_rec( A->block( i, i ), acc );

        tbb::parallel_invoke(
            [A,i,nbr,&acc] ()
            {
                tbb::parallel_for( i+1, nbr,
                                   [A,i,&acc] ( uint  j )
                                   {
                                       solve_upper_right( A->block( j, i ),
                                                          A->block( i, i ), nullptr, acc,
                                                          solve_option_t( block_wise, general_diag, store_inverse ) );
                                   } );
            },
                
            [A,i,nbc,&acc] ()
            {
                tbb::parallel_for( i+1, nbc,
                                   [A,i,&acc] ( uint  l )
                                   {
                                       solve_lower_left( apply_normal, A->block( i, i ), nullptr,
                                                         A->block( i, l ), acc,
                                                         solve_option_t( block_wise, unit_diag, store_inverse ) );
                                   } );
            } );

        tbb::parallel_for( tbb::blocked_range2d< uint >( i+1, nbr,
                                                         i+1, nbc ),
                           [A,i,&acc] ( const tbb::blocked_range2d< uint > & r )
                           {
                               for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                               {
                                   for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                   {
                                       multiply( -1.0, A->block( j, i ), A->block( i, l ), 1.0, A->block( j, l ), acc );
                                   }// for
                               }// for
                           } );
    }// for
}

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( ! is_blocked( A ) )
        HERROR( ERR_ARG, "", "" );

    lu< value_t >( ptrcast( A, TBlockMatrix ), acc );
}

}// namespace TBB

}// namespace TILEDH

//
// main function
//
void
mymain ( int argc, char ** argv )
{
    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = TiledH::cluster( coord.get(), ntile, 4 );
    
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
        
        mvis.svd( false ).id( true ).print( A.get(), "tiledh_A" );
    }// if
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( Tiled-H TBB )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TiledH::TBB::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}
