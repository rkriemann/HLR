//
// Project     : HLib
// File        : dag-inv.hh
// Description : DAG example for matrix inversion
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "hlr/cluster/h.hh"
#include "hlr/dag/invert.hh"
#include "hlr/dag/lu.hh"

//
// main function
//
template < typename problem_t >
void
mymain ( int, char ** )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< TMatrix >();

    if ( matrixfile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = cluster::h::cluster( coord.get(), ntile );
        auto  bct     = cluster::h::blockcluster( ct.get(), ct.get() );
    
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
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = read_matrix( matrixfile );
    }// else

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "Inversion (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    hlr::dag::graph  dag;

    if ( nosparsify )
        dag::sparsify_mode = dag::sparsify_none;
    else
        dag::sparsify_mode = dag::sparsify_sub_all;
    
    auto  A_inv = impl::matrix::copy( *A );

    double  tmin = 0;
    double  tmax = 0;
    double  tsum = 0;
    
    std::cout << "    DAG" << std::endl;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();
        
        dag = std::move( hlr::dag::gen_dag_invert( *A_inv, impl::dag::refine ) );
        
        toc = timer::since( tic );
        
        if ( verbose( 1 ) )
            std::cout << "      done in  " << format_time( toc ) << std::endl;
        
        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for
        
    if ( verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "      time =   "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
        std::cout << "      #nodes = " << dag.nnodes() << std::endl;
        std::cout << "      #edges = " << dag.nedges() << std::endl;
        std::cout << "      mem    = " << format_mem( dag.mem_size() ) << std::endl;
    }// if
        
    if ( verbose( 3 ) )
        dag.print_dot( "inv.dot" );

    if ( onlydag )
        return;
        
    //////////////////////////////////////////////////////////////////////
    //
    // factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << "    Inversion" << std::endl;
    
    tmin = tmax = tsum = 0;
        
    for ( int  i = 0; i < 1; ++i ) // nbench
    {
        tic = timer::now();

        impl::dag::run( dag, acc );
        
        toc = timer::since( tic );

        std::cout << "      done in  " << format_time( toc ) << std::endl;

        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *A_inv );
    }// for
        
    if ( nbench > 1 )
        std::cout << "      time =   " << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax << std::endl;
        
    std::cout << "      mem    = " << format_mem( A_inv->byte_size() ) << std::endl;
    std::cout << "      error  = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), A_inv.get() ) << term::reset << std::endl;
}
