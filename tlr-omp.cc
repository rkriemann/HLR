//
// Project     : HLib
// File        : tlr-omp.cc
// Description : Implements TLR arithmetic using OpenMP
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "tlr.hh"
#include "tlr.inc"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace OMP
{

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  nbr = BA->nblock_rows();
        auto  nbc = BA->nblock_cols();

        for ( uint  i = 0; i < nbr; ++i )
        {
            auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
            TLR::OMP::lu< value_t >( A_ii, acc );

            #pragma omp parallel for
            for ( uint  j = i+1; j < nbc; ++j )
            {
                // L is unit diagonal !!!
                // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
                trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
            }// for

            #pragma omp parallel for collapse(2)
            for ( uint  j = i+1; j < nbr; ++j )
            {
                for ( uint  l = i+1; l < nbc; ++l )
                {
                    update< value_t >( BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
                }// for
            }// for
        }// for
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( DA->blas_rmat() );
    }// else
}

}// namespace OMP

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
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR OMP )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::OMP::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;

        write_matrix( C.get(), "LU.hm" );
    }

}
