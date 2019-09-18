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
#include "hlr/seq/dag.hh"
#include "hlr/seq/arith.hh"

namespace impl = hlr::seq;

//
// main function
//
template < typename problem_t >
void
mymain ( int, char ** )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic = Time::Wall::now();
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

    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    const size_t  ncoarse = ( coarse > 0 ? A->nrows() / coarse : A->nrows() / 50 );
    
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
        hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        hlr::dag::def_path_len  = 2;
    }// if

    //
    // benchmark DAG generation
    //
    
    double  tmin = 0;
    double  tmax = 0;
    double  tsum = 0;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = Time::Wall::now();

        dag = std::move( hlr::dag::gen_dag_lu_hodlr_tiled( *C, impl::dag::refine ) );
        
        toc = Time::Wall::since( tic );
        
        if ( verbose( 1 ) )
            std::cout << "  DAG in     " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for
        
    if ( verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
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
        tic = Time::Wall::now();
        
        impl::dag::run( dag, acc );
        
        toc = Time::Wall::since( tic );

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

int
main ( int argc, char ** argv )
{
    HLIB::CFG::set_nthreads( 1 );

    return hlrmain( argc, argv );
}
