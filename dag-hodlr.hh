//
// Project     : HLib
// File        : dag-hodlr.hh
// Description : main function for tiled HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "hlr/cluster/hodlr.hh"
#include "hlr/matrix/luinv_eval.hh"
#include "hlr/dag/lu.hh"
#include "hlr/dag/solve.hh"
#include "hlr/arith/lu.hh"

using Time::Wall::now;
using Time::Wall::since;

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
    
    std::cout << "    done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "Tiled HODLR LU (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // compute DAG
    //
    //////////////////////////////////////////////////////////////////////
    
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

    if ( false )
    {
        auto  A01 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 1, 0 ), TRkMatrix );
        auto  X   = A01->blas_rmat_A();
        auto  T   = std::make_shared< BLAS::Matrix< real > >();
        auto  U   = A10->blas_rmat_A();
        auto  Q   = std::make_shared< BLAS::Matrix< real > >();
        auto  R   = std::make_shared< BLAS::Matrix< real > >();

        *T = BLAS::prod( real(1), BLAS::adjoint( A10->blas_rmat_B() ), A01->blas_rmat_B() );
        *Q = BLAS::Matrix< real >( X.nrows(), T->ncols() + U.ncols() );
        *R = BLAS::Matrix< real >( Q->ncols(), Q->ncols() );

        DBG::write( X, "X.mat", "X" );
        DBG::write( *T, "T.mat", "T" );
        DBG::write( U, "U.mat", "U" );

        std::cout << "T = " << T.get() << std::endl;
        std::cout << "Q = " << Q.get() << std::endl;
        std::cout << "R = " << R.get() << std::endl;
        
        T = std::shared_ptr< BLAS::Matrix< real > >();
        
        if ( HLIB::CFG::Arith::use_dag )
        {
            auto  dag_tsqr = std::move( hlr::dag::gen_dag_tsqr( X, T, U, Q, R, impl::dag::refine ) );
        
            dag_tsqr.print_dot( "tsqr.dot" );
            
            impl::dag::run( dag_tsqr, acc );
        }// if
        else
        {
            hlr::seq::tile::tsqr( 1.0, X, *T, U, *Q, *R, 128 );
        }// else

        DBG::write( *Q, "Q.mat", "Q" );
        DBG::write( *R, "R.mat", "R" );

        return;
    }
    
    if ( false )
    {
        auto  A01 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 1, 0 ), TRkMatrix );
        auto  X   = A01->blas_rmat_A();
        auto  T   = std::make_shared< BLAS::Matrix< real > >();
        auto  Y   = A01->blas_rmat_B();

        *T = BLAS::prod( real(1), BLAS::adjoint( A10->blas_rmat_B() ), A01->blas_rmat_B() );

        DBG::write( X,  "X.mat", "X" );
        DBG::write( *T, "T.mat", "T" );
        DBG::write( Y,  "Y.mat", "Y" );
        DBG::write( A10, "A.mat", "A" );

        // T = std::shared_ptr< BLAS::Matrix< real > >();
        
        if ( HLIB::CFG::Arith::use_dag )
        {
            auto  dag_trunc = std::move( hlr::dag::gen_dag_truncate( X, T, Y, A10, impl::dag::refine ) );
        
            std::cout << "    #nodes = " << dag_trunc.nnodes() << std::endl;
            std::cout << "    #edges = " << dag_trunc.nedges() << std::endl;
            dag_trunc.print_dot( "trunc.dot" );

            tic = now();
            
            impl::dag::run( dag_trunc, acc );

            toc = since( tic );
        }// if
        else
        {
            tic = now();

            auto [ U, V ] = impl::tile::truncate( 1.0, X, *T, Y, A10->blas_rmat_A(), A10->blas_rmat_B(), acc, ntile );
            
            A10->set_lrmat( U, V );

            toc = since( tic );
        }// else

        std::cout << "  trunc in    " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
        DBG::write( A10, "C.mat", "C" );

        return;
    }

    if ( true )
    {
        auto  A01 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( ptrcast( C.get(), TBlockMatrix )->block( 1, 0 ), TRkMatrix );
        auto  A11 = ptrcast( C.get(), TBlockMatrix )->block( 1, 1 );
        auto  X   = A10->blas_rmat_A();
        auto  T   = std::make_shared< BLAS::Matrix< real > >();
        auto  Y   = A01->blas_rmat_B();

        *T = BLAS::prod( real(1), BLAS::adjoint( A10->blas_rmat_B() ), A01->blas_rmat_A() );

        if ( A11->nrows() <= 4096 )
        {
            DBG::write( X,  "X.mat", "X" );
            DBG::write( *T, "T.mat", "T" );
            DBG::write( Y,  "Y.mat", "Y" );
            DBG::write( A11, "A.mat", "A" );
        }// if

        // T = std::shared_ptr< BLAS::Matrix< real > >();
        
        if ( HLIB::CFG::Arith::use_dag )
        {
            std::cout << "    mem    = " << mem_usage() << std::endl;
            
            auto  dag_addlr = std::move( hlr::dag::gen_dag_addlr( X, T, Y, A11, impl::dag::refine ) );
        
            std::cout << "    #nodes = " << dag_addlr.nnodes() << std::endl;
            std::cout << "    #edges = " << dag_addlr.nedges() << std::endl;
            std::cout << "    mem    = " << Mem::to_string( dag_addlr.mem_size() ) << mem_usage() << std::endl;

            dag_addlr.print_dot( "addlr.dot" );

            tic = now();
            
            impl::dag::run( dag_addlr, acc );

            toc = since( tic );
            std::cout << "    mem    = " << mem_usage() << std::endl;
        }// if
        else
        {
            tic = now();

            impl::tile::hodlr::addlr( X, *T, Y, A11, acc, ntile );
            
            toc = since( tic );
        }// else

        std::cout << "  addlr in    " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
        if ( A11->nrows() <= 4096 )
            DBG::write( A11, "C.mat", "C" );

        return;
    }
    
    //
    // benchmark DAG generation
    //
    
    std::vector< double >  runtime;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = now();

        dag = std::move( hlr::dag::gen_dag_lu_hodlr_tiled( *C, impl::dag::refine ) );
        
        toc = since( tic );
        
        if ( verbose( 1 ) )
            std::cout << "  DAG in     " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
        runtime.push_back( toc.seconds() );
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for
        
    if ( verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << Mem::to_string( dag.mem_size() ) << mem_usage() << std::endl;
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
    
    tmin = tmax = tsum = 0;
        
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = now();
        
        impl::dag::run( dag, acc );
        
        toc = since( tic );

        std::cout << "  LU in      " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *C );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                  << std::endl;
        
    std::cout << "    mem    = " << Mem::to_string( C->byte_size() ) << mem_usage() << std::endl;
        
    matrix::luinv_eval  A_inv( C, impl::dag::refine, impl::dag::run );
        
    std::cout << "    error  = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << term::reset << std::endl;
}
