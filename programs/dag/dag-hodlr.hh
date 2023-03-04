//
// Project     : HLR
// Module      : dag-hodlr.hh
// Description : main function for tiled HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/hodlr.hh"
#include "hlr/matrix/luinv_eval.hh"
#include "hlr/dag/lu.hh"
#include "hlr/dag/solve.hh"
#include "hlr/arith/lu.hh"

using namespace hlr;

using hlr::matrix::tiled_lrmatrix;
using hlr::matrix::to_dense;

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

    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = cluster::hodlr::cluster( *coord, ntile );
        auto  bct     = cluster::hodlr::blockcluster( *ct, *ct );
    
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

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
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

    //
    // benchmark DAG generation
    //
    
    std::vector< double >  runtime;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();

        dag = std::move( hlr::dag::gen_dag_lu_hodlr_tiled( *C, ntile, hlr::seq::dag::refine ) );
        
        toc = timer::since( tic );
        
        if ( hpro::verbose( 1 ) )
            std::cout << "  DAG in     " << format_time( toc ) << std::endl;
        
        runtime.push_back( toc.seconds() );
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for
        
    if ( hpro::verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
    }// if

    if ( hpro::verbose( 3 ) )
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
        tic = timer::now();
        
        impl::dag::run( dag, acc );
        
        toc = timer::since( tic );

        std::cout << "  LU in      " << format_time( toc ) << std::endl;

        runtime.push_back( toc.seconds() );

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *C );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;
        
    std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;

    // {
    //     auto  T1 = impl::matrix::copy_nontiled< double >( *C );
    //     auto  T2 = hpro::to_dense( T1.get() );
    //
    //     write_matrix( T2.get(), "B.mat", "B" );
    // }
    
    hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
    // matrix::luinv_eval  A_inv( C, impl::dag::refine, impl::dag::run );
        
    std::cout << "    error  = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << term::reset << std::endl;
}
