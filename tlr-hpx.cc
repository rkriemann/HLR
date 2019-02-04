//
// Project     : HLib
// File        : tlr-hpx.cc
// Description : TLR arithmetic with HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpx/hpx_init.hpp>
#include <hpx/parallel/task_block.hpp>
#include <hpx/include/iostreams.hpp>

#include "cmdline.inc"
#include "problem.inc"
#include "tlr.hh"
#include "tlr.inc"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace HPX
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
        auto  A_ii = ptrcast( A->block( i, i ), TDenseMatrix );

        BLAS::invert( blas_mat< value_t >( A_ii ) );
        
        hpx::parallel::v2::define_task_block(
            [i,nbc,A_ii,A] ( auto &  tb )
            {
                for ( uint  j = i+1; j < nbc; ++j )
                    tb.run( [A_ii,A,j,i] { TLR::trsmuh< value_t >( A_ii, A->block( j, i ) ); } );
            } );

        hpx::parallel::v2::define_task_block(
            [A,i,nbr,nbc,&acc] ( auto &  tb )
            {
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    auto  A_ji = A->block( j, i );
                                   
                    for ( uint  l = i+1; l < nbc; ++l )
                    {
                        auto  A_il = A->block( i, l );
                        auto  A_jl = A->block( j, l );
                                       
                        tb.run( [A_ji,A_il,A_jl,&acc] { TLR::update< value_t >( A_ji, A_il, A_jl, acc ); } );
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
    if ( is_blocked( A ) )
        lu< value_t >( ptrcast( A, TBlockMatrix ), acc );
    else
        B::invert( ptrcast( A, TDenseMatrix )->blas_rmat() );
}

}// namespace HPX

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
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR HPX )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::HPX::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}

int
hpx_main ( int argc, char ** argv )
{
    parse_cmdline( argc, argv );
    
    try
    {
        INIT();

        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        mymain( argc, argv );

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }
    
    return hpx::finalize();
}

int
main ( int argc, char ** argv )
{
    return hpx::init( argc, argv );
}
