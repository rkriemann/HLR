//
// Project     : HLR
// File        : dag.hh
// Description : main function for DAG examples
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/cluster/hodlr.hh"
#include "hlr/cluster/tlr.hh"
#include "hlr/cluster/mblr.hh"
#include "hlr/cluster/tileh.hh"
#include "hlr/matrix/level_matrix.hh"
#include "hlr/matrix/luinv_eval.hh"
#include "hlr/dag/lu.hh"
#include "hlr/dag/solve.hh"
#include "hlr/arith/lu.hh"
#include "hlr/seq/dag.hh"
#include "hlr/seq/arith.hh"
#include "hlr/utils/likwid.hh"
#include "hlr/utils/io.hh"

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

    if ( matrixfile == "" && sparsefile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = gen_ct( *coord );
        auto  bct     = gen_bct( *ct, *ct );
    
        if ( hpro::verbose( 3 ) )
        {
            io::eps::print( *ct->root(), "ct" );
            io::eps::print( *bct->root(), "ct" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );

        A = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    }// if
    else if ( matrixfile != "" )
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = hpro::read_matrix( matrixfile );

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// if
    else if ( sparsefile != "" )
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    sparse matrix = " << sparsefile
                  << std::endl;

        auto  M = hpro::read_matrix( sparsefile );
        auto  S = ptrcast( M.get(), hpro::TSparseMatrix );

        // convert to H
        auto  part_strat    = hpro::TMongooseAlgPartStrat();
        auto  ct_builder    = hpro::TAlgCTBuilder( & part_strat, ntile );
        auto  nd_ct_builder = hpro::TAlgNDCTBuilder( & ct_builder, ntile );
        auto  cl            = nd_ct_builder.build( S );

        S->permute( *cl->perm_e2i(), *cl->perm_e2i() );

        if ( hpro::verbose( 3 ) )
            io::eps::print( *S, "S", "noid,pattern" );
        
        auto  adm_cond      = hpro::TWeakAlgAdmCond( S );
        auto  bct_builder   = hpro::TBCBuilder();
        auto  bcl           = bct_builder.build( cl.get(), cl.get(), & adm_cond );
        // auto  h_builder     = hpro::TSparseMatBuilder( S, cl->perm_i2e(), cl->perm_e2i() );

        if ( hpro::verbose( 3 ) )
        {
            io::eps::print( * cl->root(), "ct" );
            io::eps::print( * bcl->root(), "bct" );
        }// if
        
        // h_builder.set_use_zero_mat( true );
        // A = h_builder.build( bcl.get(), acc );

        approx::SVD< value_t >  apx;
            
        A = impl::matrix::build( bcl->root(), *S, acc, apx, nseq );
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

    const size_t  ncoarse = ( coarse > 0 ? A->nrows() / coarse : A->nrows() / 50 );
    
    std::cout << term::bullet << term::bold
              << ( levelwise ? "Level LU (DAG)" : ( coarse > 0 ? HLIB::to_string( "LU (Coarse-%d DAG)", ncoarse ) : "LU (DAG)" ) )
              << term::reset
              << ", " << acc.to_string()
              << ", nseq = " << nseq
              << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // compute DAG
    //
    //////////////////////////////////////////////////////////////////////
    
    hlr::dag::graph  dag;
    
    auto  C = ( onlydag ? std::shared_ptr( std::move( A ) ) : std::shared_ptr( impl::matrix::copy( *A ) ) );

    if ( levelwise )
        C->set_hierarchy_data();

    //
    // set up DAG generation options optimised for different DAGs
    //

    if ( nosparsify )
    {
        hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
    }// if
    else
    {
        if ( levelwise )
        {
            // different algorithm; no options evaluated
        }// if
        else if ( coarse > 0 )
        {
            hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
            hlr::dag::def_path_len  = 10;
        }// if
        else if ( oop_lu )
        {
            if ( hpro::CFG::Arith::use_accu )
            {
                hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
                hlr::dag::def_path_len  = 10;
            }// if
            else
            {
                hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
                hlr::dag::def_path_len  = 10;
            }// else
        }// if
        else
        {
            hlr::dag::sparsify_mode = hlr::dag::sparsify_node_succ;
            hlr::dag::def_path_len  = 2;
        }// if
    }// if

    //
    // benchmark DAG generation
    //

    auto  runtime = std::vector< double >();
    
    LIKWID_MARKER_INIT;
        
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();
        
        LIKWID_MARKER_START( "dag" );

        if ( levelwise )
            dag = std::move( hlr::dag::gen_dag_lu_lvl( *C, nseq ) );
        else if ( coarse > 0 )
            dag = std::move( hlr::dag::gen_dag_lu_oop_coarse( *C, ncoarse, impl::dag::refine, impl::dag::run ) );
        else if ( oop_lu )
        {
            if ( hpro::CFG::Arith::use_accu )
                if ( fused )
                    dag = std::move( hlr::dag::gen_dag_lu_oop_accu( *C, nseq, impl::dag::refine ) );
                else
                    dag = std::move( hlr::dag::gen_dag_lu_oop_accu_sep( *C, nseq, impl::dag::refine ) );
            else
                dag = std::move( hlr::dag::gen_dag_lu_oop_auto( *C, nseq, impl::dag::refine ) );
        }// if
        else 
            dag = std::move( hlr::dag::gen_dag_lu_ip( *C, nseq, impl::dag::refine ) );
        
        LIKWID_MARKER_STOP( "dag" );
        
        toc = timer::since( tic );
        
        if ( hpro::verbose( 1 ) )
        {
            std::cout << "  DAG in     " << format_time( toc ) << std::endl;
            
            // std::cout << "    #coll  = " << hlr::dag::collisions << std::endl;
        }// if
        
        runtime.push_back( toc.seconds() );
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for

    LIKWID_MARKER_CLOSE;
        
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
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;
        
    std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        
    // matrix::luinv_eval  A_inv( C, impl::dag::refine, impl::dag::run );
    hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
    std::cout << "    error  = " << format_error( inv_approx_2( A.get(), & A_inv ) ) << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // vector solves
    //
    //////////////////////////////////////////////////////////////////////

    // {
    //     TScalarVector *        x_ptr;
    //     hlr::dag::mutex_map_t  map_row_mtx;
        
    //     auto  dag_trsml = gen_dag_solve_lower( apply_normal, C.get(), & x_ptr, map_row_mtx, impl::dag::refine );

    //     dag_trsml.print_dot( "trsml.dot" );
    // }
    
    if ( false )
    {
        std::cout << term::bullet << term::bold << "Vector Solves" << term::reset << std::endl;
    
        hpro::CFG::Arith::vector_solve_method = 1;

        auto   mtx_map = std::map< idx_t, std::unique_ptr< std::mutex > >();
        idx_t  last    = -1;

        for ( auto  i : A->row_is() )
        {
            const idx_t  ci = i / hlr::dag::CHUNK_SIZE;
            
            if ( ci != last )
            {
                last = ci;
                mtx_map[ ci ] = std::make_unique< std::mutex >();
            }// if
        }// for
        
        {
            hpro::TScalarVector  x( A->col_is(), A->value_type() );

            x.fill_rand( 1 );

            const hpro::TScalarVector  xcopy( x );
            hpro::TScalarVector        xref( x );

            runtime.clear();
                
            for ( int  i = 0; i < nbench; ++i )
            {
                tic = timer::now();
        
                hlr::trsvu( hpro::apply_trans, *C, xref, hpro::general_diag );
                hlr::trsvl( hpro::apply_trans, *C, xref, hpro::unit_diag );
        
                toc = timer::since( tic );

                std::cout << "  trsv in    " << format_time( toc ) << std::endl;

                runtime.push_back( toc.seconds() );

                if ( i < (nbench-1) )
                    xref.assign( 1.0, & xcopy );
            }// for

            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            matrix::luinv_eval   A_inv2( *C ); // , impl::dag::refine, impl::dag::run );
            hpro::TScalarVector  v( x );
        
            runtime.clear();
            
            for ( int  i = 0; i < nbench; ++i )
            {
                tic = timer::now();

                A_inv2.apply( & x, & v, hpro::apply_trans );

                toc = timer::since( tic );

                std::cout << "  solve in   " << format_time( toc ) << std::endl;

                runtime.push_back( toc.seconds() );
            }// for

            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;
                
            // DBG::write( & v,    "x.mat", "x" );
            // DBG::write( & xref, "y.mat", "y" );

            v.axpy( -1, & xref );
            std::cout << "  error =    " << format_error( v.norm2() / xref.norm2() ) << std::endl;
        }
    }// if
}
