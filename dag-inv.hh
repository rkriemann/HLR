//
// Project     : HLR
// File        : dag-inv.hh
// Description : DAG example for matrix inversion
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/dag/invert.hh"
#include "hlr/dag/lu.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    
    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< hpro::TMatrix >();

    if ( matrixfile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = cluster::h::cluster( *coord, ntile );
        auto  bct     = cluster::h::blockcluster( *ct, *ct );
    
        if ( hpro::verbose( 3 ) )
        {
            hpro::TPSBlockClusterVis   bc_vis;
        
            bc_vis.id( true ).print( bct->root(), "bct" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );

        A = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = hpro::read_matrix( matrixfile );
    }// else

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "Inversion (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    hlr::dag::graph  dag;

    if ( nosparsify )
        dag::sparsify_mode = dag::sparsify_none;
    else
        dag::sparsify_mode = dag::sparsify_sub_all;
    
    auto  A_inv = impl::matrix::copy( *A );

    std::vector< double >  runtime;
    
    std::cout << "    DAG" << std::endl;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();
        
        dag = std::move( hlr::dag::gen_dag_invert( *A_inv, nseq, impl::dag::refine ) );
        
        toc = timer::since( tic );
        
        if ( hpro::verbose( 1 ) )
            std::cout << "      done in  " << format_time( toc ) << std::endl;
        
        runtime.push_back( toc.seconds() );
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for
        
    if ( hpro::verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
    }// if
        
    if ( hpro::verbose( 3 ) )
        dag.print_dot( "inv.dot" );

    if ( onlydag )
        return;
        
    //////////////////////////////////////////////////////////////////////
    //
    // factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << "    Inversion" << std::endl;
    
    runtime.clear();
        
    for ( int  i = 0; i < 1; ++i ) // nbench
    {
        tic = timer::now();

        impl::dag::run( dag, acc );
        
        toc = timer::since( tic );

        std::cout << "      done in  " << format_time( toc ) << std::endl;

        runtime.push_back( toc.seconds() );

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *A_inv );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;
        
    std::cout << "      mem    = " << format_mem( A_inv->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( hpro::inv_approx_2( A.get(), A_inv.get() ) ) << std::endl;
}
