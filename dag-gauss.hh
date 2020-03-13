//
// Project     : HLR
// File        : gauss.hh
// Description : DAG example for Gaussian elimination
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/dag/gauss_elim.hh"

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
        auto  ct      = gen_ct( *coord );
        auto  bct     = gen_bct( *ct, *ct );
    
        if ( verbose( 3 ) )
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

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// else

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "Gauss (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    hlr::dag::graph  dag;
    
    auto  T = A->copy();
    auto  C = A->copy();

    double  tmin = 0;
    double  tmax = 0;
    double  tsum = 0;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();
        
        dag = std::move( hlr::dag::gen_dag_gauss_elim( C.get(), T.get(), impl::dag::refine ) );
        
        toc = timer::since( tic );
        
        if ( verbose( 1 ) )
            std::cout << "  DAG in     " << format_time( toc ) << std::endl;
        
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
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
    }// if
        
    if ( verbose( 3 ) )
        dag.print_dot( "gauss.dot" );
        
    if ( onlydag )
        return;
        
    //////////////////////////////////////////////////////////////////////
    //
    // factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    tmin = tmax = tsum = 0;
        
    for ( int  i = 0; i < 1; ++i ) // nbench
    {
        tic = timer::now();

        if ( HLIB::CFG::Arith::use_dag )
            impl::dag::run( dag, acc );
        else
            impl::gauss_elim( C.get(), T.get(), acc );
        
        toc = timer::since( tic );

        std::cout << "  Gauss in   " << format_time( toc ) << std::endl;

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
        
    std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "    error  = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), C.get() ) << term::reset << std::endl;

    // DBG::write( C.get(), "C.mat", "C" );
}
