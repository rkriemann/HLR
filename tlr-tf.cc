//
// Project     : HLib
// File        : tlr-tbb.cc
// Description : TLR arithmetic with cpp-taskflow
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <taskflow/taskflow.hpp>

#include "tensor.hh"

#include "common.inc"
#include "tlr.hh"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace TF
{

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
    tf::Taskflow  tf;
    
    auto                 nbr = A->nblock_rows();
    auto                 nbc = A->nblock_cols();
    tensor2< tf::Task >  fs_tasks( nbr, nbc );
    tensor3< tf::Task >  u_tasks( nbr, nbr, nbc );
    tensor3< char >      has_u_task( nbr, nbr, nbc, false );

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( A->block( i, i ), TDenseMatrix );

        fs_tasks(i,i) = tf.silent_emplace( [A_ii] ()
                                           {
                                               TScopedLock  lock( *A_ii );
                                               
                                               BLAS::invert( blas_mat< value_t >( A_ii ) );
                                           } );
            
        for ( uint  l = 0; l < i; ++l )
            if ( has_u_task(l,i,i) )
                u_tasks(l,i,i).precede( fs_tasks(i,i) );
            
        for ( uint  j = i+1; j < nbc; ++j )
        {
            // L is identity; task only for ensuring correct execution order
            fs_tasks(i,j) = tf.silent_emplace( [A_ii,A,i,j] ()
                                               {
                                                   auto         A_ij = A->block(i,j);
                                                   TScopedLock  lock( *A_ij );
                                               } );
            fs_tasks(i,i).precede( fs_tasks(i,j) );

            for ( uint  l = 0; l < i; ++l )
                if ( has_u_task(l,i,j) )
                    u_tasks(l,i,j).precede( fs_tasks(i,j) );
            
            fs_tasks(j,i) = tf.silent_emplace( [A_ii,A,i,j] ()
                                               {
                                                   auto         A_ji = A->block(j,i);
                                                   TScopedLock  lock( *A_ji );
                                                   
                                                   TLR::trsmuh< value_t >( A_ii, A_ji );
                                               } );
            fs_tasks(i,i).precede( fs_tasks(j,i) );

            for ( uint  l = 0; l < i; ++l )
                if ( has_u_task(l,j,i) )
                    u_tasks(l,j,i).precede( fs_tasks(j,i) );
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  A_ji = A->block( j, i );
                
            for ( uint  l = i+1; l < nbc; ++l )
            {
                auto  A_il = A->block( i, l );
                auto  A_jl = A->block( j, l );

                u_tasks(i,j,l)    = tf.silent_emplace( [A_ji,A_il,A_jl,&acc] ()
                                                       {
                                                           TScopedLock  lock( *A_jl );
                                                           
                                                           TLR::update< value_t >( A_ji, A_il, A_jl, acc );
                                                       } );
                has_u_task(i,j,l) = true;
                
                // ensures non-simultanous writes
                // if ( i > 0 )
                //     u_tasks(i-1,j,l).precede( u_tasks(i,j,l) );
                
                fs_tasks(j,i).precede( u_tasks(i,j,l) );
                fs_tasks(i,l).precede( u_tasks(i,j,l) );
            }// for
        }// for
    }// for
    
    tf.wait_for_all();
}

}// namespace TF

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
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR TF )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::TF::lu< HLIB::real >( ptrcast( C.get(), TBlockMatrix ), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}
