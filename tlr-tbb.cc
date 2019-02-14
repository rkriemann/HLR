//
// Project     : HLib
// File        : tlr-tbb.cc
// Description : TLR arithmetic with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include "common.inc"
#include "tlr.hh"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace TBB
{

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  nbr = BA->nblock_rows();
        auto  nbc = BA->nblock_cols();

        for ( uint  i = 0; i < nbr; ++i )
        {
            auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
            TLR::TBB::lu< value_t >( A_ii, acc );

            tbb::parallel_for( i+1, nbc,
                               [A_ii,BA,i] ( uint  j )
                               {
                                   // L is unit diagonal !!!
                                   // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
                                   trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
                               } );

            tbb::parallel_for( tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                               [BA,i,&acc] ( const tbb::blocked_range2d< uint > & r )
                               {
                                   for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                   {
                                       for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                       {
                                           update< value_t >( BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
                                       }// for
                                   }// for
                               } );
        }// for
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( DA->blas_rmat() );
    }// else
}

}// namespace TBB

}// namespace TLR

//
// main function
//
void
mymain ( int argc, char ** argv )
{
    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = TLR::cluster( coord.get(), ntile );
    
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
        
        mvis.svd( false ).id( true ).print( A.get(), "hlrtest_A" );
    }// if
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR TBB )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::TBB::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}
