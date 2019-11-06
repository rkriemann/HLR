//
// Project     : HLib
// File        : dag-hodlr.hh
// Description : main function for tiled HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "hlr/cluster/hodlr.hh"
#include "hlr/matrix/luinv_eval.hh"
#include "hlr/dag/lu.hh"
#include "hlr/dag/solve.hh"
#include "hlr/arith/lu.hh"

using Time::Wall::now;
using Time::Wall::since;

using hlr::matrix::tiled_lrmatrix;

//
// main function
//
template < typename problem_t >
void
mymain ( int, char ** )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic = now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< TMatrix >();

    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = cluster::hodlr::cluster( coord.get(), ntile );
        auto  bct     = cluster::hodlr::blockcluster( ct.get(), ct.get() );
    
        if ( verbose( 3 ) )
        {
            TPSBlockClusterVis   bc_vis;
        
            bc_vis.id( true ).print( bct->root(), "bct" );
            print_vtk( coord.get(), "coord" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = std::make_unique< TACAPlus< value_t > >( pcoeff.get() );

        A = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    }// if

    auto  toc    = since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    // convert to tiled format
    A = impl::matrix::copy_tiled< double >( *A, ntile );

    //////////////////////////////////////////////////////////////////////
    //
    // compute DAG
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "Tiled HODLR LU (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    hlr::dag::graph  dag;
    
    auto  C = ( onlydag ? std::shared_ptr( std::move( A ) ) : std::shared_ptr( A->copy() ) );

    //
    // set up DAG generation options optimised for different DAGs
    //

    if ( nosparsify )
    {
        hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
    }// if
    else
    {
        hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all_ext;
        hlr::dag::def_path_len  = 10;
    }// if

    if ( true )
    {
        auto  A01 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 0, 1 ), tiled_lrmatrix< real > );
        auto  A10 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 1, 0 ), tiled_lrmatrix< real > );
        auto & X   = A01->V();
        auto  T   = std::make_shared< BLAS::Matrix< real > >();
        auto & U   = A10->U();
        auto  Q   = std::make_shared< hlr::matrix::tile_storage< real > >();
        auto  R   = std::make_shared< BLAS::Matrix< real > >();

        *T = hlr::seq::tiled2::dot( A10->col_is(), A10->V(), A01->U(), ntile );

        if ( HLIB::CFG::Arith::use_dag )
        {
            auto  dag_tsqr = std::move( hlr::dag::gen_dag_tsqr2( A01->ncols(), X, T, U, Q, R, impl::dag::refine ) );
        
            dag_tsqr.print_dot( "tsqr.dot" );
            
            impl::dag::run( dag_tsqr, acc );
        }// if
        else
        {
            std::tie( *Q, *R ) = hlr::seq::tiled2::tsqr( A01->col_is(), 1.0, X, *T, U, 128 );
        }// else

        return;
    }
    
    // if ( false )
    // {
    //     auto  A01 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 0, 1 ), tiled_lrmatrix< real > );
    //     auto  A10 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 1, 0 ), tiled_lrmatrix< real > );
    //     auto  X   = A01->U();
    //     auto  T   = std::make_shared< BLAS::Matrix< real > >();
    //     auto  Y   = A01->V();

    //     *T = hlr::seq::tiled2::dot( A10->row_is(), real(1), A10->U(), A01->V(), ntile );

    //     if ( HLIB::CFG::Arith::use_dag )
    //     {
    //         auto  dag_trunc = std::move( hlr::dag::gen_dag_truncate2( X, T, Y, A10, impl::dag::refine ) );
        
    //         std::cout << "    #nodes = " << dag_trunc.nnodes() << std::endl;
    //         std::cout << "    #edges = " << dag_trunc.nedges() << std::endl;
    //         dag_trunc.print_dot( "trunc.dot" );

    //         tic = now();
            
    //         impl::dag::run( dag_trunc, acc );

    //         toc = since( tic );
    //     }// if
    //     else
    //     {
    //         tic = now();

    //         auto [ U, V ] = impl::tiled2::truncate( 1.0, X, *T, Y, A10->blas_rmat_A(), A10->blas_rmat_B(), acc, ntile );
            
    //         A10->set_lrmat( U, V );

    //         toc = since( tic );
    //     }// else

    //     std::cout << "  trunc in    " << format_time( toc ) << std::endl;

    //     return;
    // }

    // if ( false )
    // {
    //     auto  A01 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 0, 1 ), tiled_lrmatrix< real > );
    //     auto  A10 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 1, 0 ), tiled_lrmatrix< real > );
    //     auto  A11 = ptrcast( C.get(), TBlockMatrix )->block( 1, 1 );
    //     auto  X   = A10->blas_rmat_A();
    //     auto  T   = std::make_shared< BLAS::Matrix< real > >();
    //     auto  Y   = A01->blas_rmat_B();

    //     *T = BLAS::prod( real(1), BLAS::adjoint( A10->blas_rmat_B() ), A01->blas_rmat_A() );

    //     if ( A11->nrows() <= 4096 )
    //     {
    //         DBG::write( X,  "X.mat", "X" );
    //         DBG::write( *T, "T.mat", "T" );
    //         DBG::write( Y,  "Y.mat", "Y" );
    //         DBG::write( A11, "A.mat", "A" );
    //     }// if

    //     // T = std::shared_ptr< BLAS::Matrix< real > >();
        
    //     if ( HLIB::CFG::Arith::use_dag )
    //     {
    //         std::cout << "    mem    = " << mem_usage() << std::endl;
            
    //         auto  dag_addlr = std::move( hlr::dag::gen_dag_addlr2( X, T, Y, A11, impl::dag::refine ) );
        
    //         std::cout << "    #nodes = " << dag_addlr.nnodes() << std::endl;
    //         std::cout << "    #edges = " << dag_addlr.nedges() << std::endl;
    //         std::cout << "    mem    = " << Mem::to_string( dag_addlr.mem_size() ) << mem_usage() << std::endl;

    //         dag_addlr.print_dot( "addlr.dot" );

    //         tic = now();
            
    //         impl::dag::run( dag_addlr, acc );

    //         toc = since( tic );
    //         std::cout << "    mem    = " << mem_usage() << std::endl;
    //     }// if
    //     else
    //     {
    //         tic = now();

    //         impl::tiled2::hodlr::addlr( X, *T, Y, A11, acc, ntile );
            
    //         toc = since( tic );
    //     }// else

    //     std::cout << "  addlr in    " << format_time( toc ) << std::endl;
        
    //     if ( A11->nrows() <= 4096 )
    //         DBG::write( A11, "C.mat", "C" );

    //     return;
    // }
    
    //
    // benchmark DAG generation
    //
    
    std::vector< double >  runtime;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = now();

        dag = std::move( hlr::dag::gen_dag_lu_hodlr_tiled2( *C, ntile, impl::dag::refine ) );
        
        toc = since( tic );
        
        if ( verbose( 1 ) )
            std::cout << "  DAG in     " << format_time( toc ) << std::endl;
        
        runtime.push_back( toc.seconds() );
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for
        
    if ( verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
    }// if

    if ( verbose( 3 ) )
        dag.print_dot( "lu.dot" );
    
    if ( onlydag )
        return;
        
    //////////////////////////////////////////////////////////////////////
    //
    // factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    runtime.clear();
        
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = now();
        
        impl::dag::run( dag, acc );
        
        toc = since( tic );

        std::cout << "    done in " << format_time( toc ) << std::endl;

        runtime.push_back( toc.seconds() );

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *C );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;
        
    std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        
    TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
    // matrix::luinv_eval  A_inv( C, impl::dag::refine, impl::dag::run );
        
    std::cout << "    error  = " << format_error( inv_approx_2( A.get(), & A_inv ) ) << term::reset << std::endl;
}
