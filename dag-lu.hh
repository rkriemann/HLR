//
// Project     : HLib
// File        : dag.hh
// Description : main function for DAG examples
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

// #include <likwid.h>

#include "common.inc"
#include "hlr/cluster/h.hh"
#include "hlr/matrix/level_matrix.hh"
#include "hlr/matrix/luinv_eval.hh"
#include "hlr/dag/lu.hh"
#include "hlr/dag/solve.hh"
#include "hlr/arith/lu.hh"
#include "hlr/seq/dag.hh"
#include "hlr/seq/arith.hh"

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

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// else

    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << Mem::to_string( A->byte_size() ) << " / " << Mem::to_string( Mem::usage() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    const size_t  ncoarse = ( coarse > 0 ? A->nrows() / coarse : A->nrows() / 50 );
    
    std::cout << term::bullet << term::bold
              << ( levelwise ? "Level LU (DAG)" : ( coarse > 0 ? HLIB::to_string( "LU (Coarse-%d DAG)", ncoarse ) : "LU (DAG)" ) )
              << term::reset
              << ", " << acc.to_string()
              << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // compute DAG
    //
    //////////////////////////////////////////////////////////////////////
    
    hlr::dag::graph  dag;
    
    auto  C = ( onlydag ? std::shared_ptr( std::move( A ) ) : std::shared_ptr( A->copy() ) );
    auto  L = ( onlydag ? C : ( oop_lu ? std::shared_ptr( C->copy() ) : C ) );
    auto  U = ( onlydag ? C : ( oop_lu ? std::shared_ptr( C->copy() ) : C ) );

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
            hlr::dag::sparsify_mode = hlr::dag::sparsify_node_succ;
            hlr::dag::def_path_len  = 2;
        }// if
        else if ( oop_lu )
        {
            hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
            hlr::dag::def_path_len  = 10;
        }// if
        else if ( CFG::Arith::use_accu )
        {
            hlr::dag::sparsify_mode = hlr::dag::sparsify_node_succ;
            hlr::dag::def_path_len  = 2;
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
    
    double  tmin = 0;
    double  tmax = 0;
    double  tsum = 0;
    
    // LIKWID_MARKER_INIT;
        
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = Time::Wall::now();
        
        // LIKWID_MARKER_START( "dag" );

        if ( levelwise )
            dag = std::move( hlr::dag::gen_dag_lu_lvl( *C ) );
        else if ( coarse > 0 )
            dag = std::move( hlr::dag::gen_dag_coarselu( C.get(), impl::dag::refine, seq::dag::refine, impl::dag::run, ncoarse ) );
        else if ( oop_lu )
            dag = std::move( hlr::dag::gen_dag_lu_oop( *C, *L, *U, impl::dag::refine ) );
        else if ( CFG::Arith::use_accu )
            dag = std::move( hlr::dag::gen_dag_lu_accu( C.get(), impl::dag::refine ) );
        else 
            dag = std::move( hlr::dag::gen_dag_lu_rec( C.get(), impl::dag::refine ) );
        
        // LIKWID_MARKER_STOP( "dag" );
        
        toc = Time::Wall::since( tic );
        
        if ( verbose( 1 ) )
        {
            std::cout << "  DAG in     " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
            
            // std::cout << "    #coll  = " << hlr::dag::collisions << std::endl;
        }// if
        
        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();
        
        if ( i < nbench-1 )
            dag = std::move( hlr::dag::graph() );
    }// for

    // LIKWID_MARKER_CLOSE;
        
    if ( verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << Mem::to_string( dag.mem_size() ) << " / " << Mem::to_string( Mem::usage() ) << std::endl;
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
        
    for ( int  i = 0; i < 1; ++i ) // nbench
    {
        tic = Time::Wall::now();
        
        impl::dag::run( dag, acc );
        
        toc = Time::Wall::since( tic );

        std::cout << "  LU in      " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();

        // if ( i < (nbench-1) )
        // {
        //     impl::matrix::copy_to( *A, *C );
        //     dag.reset_dependencies();
        // }// if
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                  << std::endl;
        
    std::cout << "    mem    = " << Mem::to_string( C->byte_size() ) << " / " << Mem::to_string( Mem::usage() ) << std::endl;
        
    matrix::luinv_eval  A_inv( C, impl::dag::refine, impl::dag::run );
    // TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
    std::cout << "    error  = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << term::reset << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // vector solves
    //
    //////////////////////////////////////////////////////////////////////
        
    std::cout << term::bullet << term::bold << "Vector Solves" << term::reset << std::endl;
    
    HLIB::CFG::Arith::vector_solve_method = 1;

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
        TScalarVector  x( A->col_is() );

        x.fill_rand( 1 );

        const TScalarVector  xcopy( x );
        TScalarVector        xref( x );

        tmin = tmax = tsum = 0;
                
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();
        
            hlr::seq::trsvu( apply_trans, *C, xref, general_diag );
            hlr::seq::trsvl( apply_trans, *C, xref, unit_diag );
        
            toc = Time::Wall::since( tic );

            std::cout << "  trsv in    " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();

            if ( i < (nbench-1) )
                xref.assign( 1.0, & xcopy );
        }// for

        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;


        matrix::luinv_eval  A_inv2( C, impl::dag::refine, impl::dag::run );
        TScalarVector       v( x );
        
        tmin = tmax = tsum = 0;
        
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();

            A_inv2.apply( & x, & v, apply_trans );

            toc = Time::Wall::since( tic );

            std::cout << "  solve in   " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();
        }// for

        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
                
        // DBG::write( & v,    "x.mat", "x" );
        // DBG::write( & xref, "y.mat", "y" );

        v.axpy( -1, & xref );
        std::cout << "  error =    " << term::ltred << format( "%.3e s" ) % ( v.norm2() / xref.norm2() ) << term::reset << std::endl;
    }
}
