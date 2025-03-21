//
// Project     : HLR
// Module      : dag-waz.hh
// Description : main function for DAG examples
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

// #include <likwid.h>

#include <hpro/algebra/mat_fac.hh>
#include <hpro/matrix/TMatrixProduct.hh>
#include <hpro/matrix/TMatrixSum.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/dag/lu.hh"
#include "hlr/dag/invert.hh"
#include "hlr/seq/norm.hh"
#include "hlr/matrix/identity.hh"

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

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// else

    auto  toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    std::cout << "    norm   = " << format_error( hlr::norm::spectral( *A ) ) << std::endl;

    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "WAZ (DAG)" << term::reset
              << ", " << acc.to_string()
              << ", nseq = " << nseq
              << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // compute DAG
    //
    //////////////////////////////////////////////////////////////////////
    
    hlr::dag::graph  dag;
    
    auto  C = ( onlydag ? std::shared_ptr( std::move( A ) ) : std::shared_ptr( A->copy() ) );

    //
    // set up DAG generation options
    //

    if ( nosparsify )
    {
        hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
    }// if
    else
    {
        hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
        hlr::dag::def_path_len  = 10;
    }// if

    {
        //
        // single DAGs
        //

        auto  dag_lu     = hlr::dag::gen_dag_lu_ip(     *C,                     nseq, impl::dag::refine );
        auto  dag_inv_ll = hlr::dag::gen_dag_invert_ll( *C, hpro::unit_diag,    nseq, impl::dag::refine );
        auto  dag_inv_ur = hlr::dag::gen_dag_invert_ur( *C, hpro::general_diag, nseq, impl::dag::refine );

        dag_lu.print_dot( "lu.dot" );
        dag_inv_ll.print_dot( "invll.dot" );
        dag_inv_ur.print_dot( "invur.dot" );
    }
    
    //
    // benchmark DAG generation
    //
    
    std::vector< double >  runtime;
    
    // LIKWID_MARKER_INIT;
        
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();
        
        // LIKWID_MARKER_START( "dag" );

        dag = std::move( hlr::dag::gen_dag_waz( *C, nseq, impl::dag::refine ) );
        
        // LIKWID_MARKER_STOP( "dag" );
        
        toc = timer::since( tic );
        
        if ( hpro::verbose( 1 ) )
            std::cout << "  DAG in     " << format_time( toc ) << std::endl;
        
        runtime.push_back( toc.seconds() );
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for

    // LIKWID_MARKER_CLOSE;
        
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
        dag.print_dot( "waz.dot" );
    
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
        tic = timer::now();
        
        impl::dag::run( dag, acc );
        
        toc = timer::since( tic );

        std::cout << "  WAZ in     " << format_time( toc ) << std::endl;

        runtime.push_back( toc.seconds() );

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *C );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;
        
    std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;

    hpro::DBG::write( C.get(), "A.mat", "A" );

    auto  I     = hlr::matrix::identity( C->row_is() );
    auto  W     = impl::matrix::copy_ll( *C, hpro::unit_diag );
    auto  Z     = impl::matrix::copy_ur( *C, hpro::general_diag );
    auto  A_inv = hpro::matrix_product( Z.get(), W.get() );
    auto  I_apx = hpro::matrix_product( A.get(), A_inv.get() );
    auto  I_err = hpro::matrix_sum( 1.0, I.get(), -1.0, I_apx.get() );

    std::cout << "    error  = " << format_error( hlr::norm::spectral( *I_err ) ) << std::endl;
}
