//
// Project     : HLib
// File        : dag-inv.hh
// Description : DAG example for matrix inversion
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "hlr/cluster/h.hh"
#include "hlr/dag/invert.hh"
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

extern std::atomic< size_t >  collisions;

} }// namespace hlr::dag

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
    }// else

    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "Inversion (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    if ( true )
    {
        hlr::dag::graph  dag;

        if ( nosparsify )
            hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        else
            hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
    
        auto  A_inv = impl::matrix::copy( *A );

        double  tmin = 0;
        double  tmax = 0;
        double  tsum = 0;
    
        std::cout << "    DAG" << std::endl;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();
        
            dag = std::move( hlr::dag::gen_dag_invert( A_inv.get(), impl::dag::refine ) );
        
            toc = Time::Wall::since( tic );
        
            if ( verbose( 1 ) )
                std::cout << "      done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
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
            std::cout << "      mem    = " << Mem::to_string( dag.mem_size() ) << mem_usage() << std::endl;
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
            tic = Time::Wall::now();

            impl::dag::run( dag, acc );
        
            toc = Time::Wall::since( tic );

            std::cout << "      done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *A_inv );
        }// for
        
        if ( nbench > 1 )
            std::cout << "      time =   "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
        
        std::cout << "      mem    = "
                  << Mem::to_string( A_inv->byte_size() )
                  << mem_usage() << std::endl;
        std::cout << "      error  = " << term::ltred
                  << format( "%.4e" ) % inv_approx_2( A.get(), A_inv.get() )
                  << term::reset << std::endl;
    }// if
    else if ( true )
    {
        hlr::dag::graph  dag_ll, dag_ur;

        if ( nosparsify )
            hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        else
            hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
    
        DBG::write( A.get(), "A.mat", "A" );
        
        auto  A_inv = impl::matrix::copy( *A );

        auto  dag_lu = std::move( hlr::dag::gen_dag_lu_rec( A_inv.get(), impl::dag::refine ) );

        impl::dag::run( dag_lu, acc );

        {
            auto  L = impl::matrix::copy_ll( *A_inv, unit_diag );
            auto  U = impl::matrix::copy_ur( *A_inv );

            DBG::write( L.get(), "L.mat", "L" );
            DBG::write( U.get(), "U.mat", "U" );
        }
        
        double  tmin = 0;
        double  tmax = 0;
        double  tsum = 0;
    
        std::cout << "    DAG" << std::endl;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();
        
            dag_ll = std::move( hlr::dag::gen_dag_invert_ll( A_inv.get(), unit_diag,    impl::dag::refine ) );
            dag_ur = std::move( hlr::dag::gen_dag_invert_ur( A_inv.get(), general_diag, impl::dag::refine ) );
        
            toc = Time::Wall::since( tic );
        
            if ( verbose( 1 ) )
                std::cout << "      done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();
        
            if ( i < nbench-1 )
                dag_ll = std::move( hlr::dag::graph() );
        }// for
        
        if ( verbose( 1 ) )
        {
            if ( nbench > 1 )
                std::cout << "      time =   "
                          << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                          << std::endl;
            std::cout << "      #nodes = " << dag_ll.nnodes() << " / " << dag_ur.nnodes() << std::endl;
            std::cout << "      #edges = " << dag_ll.nedges() << " / " << dag_ur.nedges() << std::endl;
            std::cout << "      mem    = " << Mem::to_string( dag_ll.mem_size() ) << mem_usage() << std::endl;
        }// if
        
        if ( verbose( 3 ) )
        {
            dag_ll.print_dot( "inv_ll.dot" );
            dag_ur.print_dot( "inv_ur.dot" );
        }// if
        
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
            tic = Time::Wall::now();

            impl::dag::run( dag_ll, acc );
            impl::dag::run( dag_ur, acc );
        
            toc = Time::Wall::since( tic );

            std::cout << "      done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *A_inv );
        }// for
        
        if ( nbench > 1 )
            std::cout << "      time =   "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
        
        auto  L     = impl::matrix::copy_ll( *A,     unit_diag );
        auto  L_inv = impl::matrix::copy_ll( *A_inv, unit_diag );
        auto  U     = impl::matrix::copy_ur( *A );
        auto  U_inv = impl::matrix::copy_ur( *A_inv );
        
        std::cout << "      mem    = "
                  << Mem::to_string( L_inv->byte_size() )
                  << " / "
                  << Mem::to_string( U_inv->byte_size() )
                  << mem_usage() << std::endl;
        std::cout << "      error  = " << term::ltred
                  << format( "%.4e" ) % inv_approx_2( L.get(), L_inv.get() )
                  << " / "
                  << format( "%.4e" ) % inv_approx_2( U.get(), U_inv.get() )
                  << term::reset << std::endl;

        DBG::write( L_inv.get(), "LI.mat", "LI" );
        DBG::write( U_inv.get(), "UI.mat", "UI" );
    }// else
    else
    {
        hlr::dag::graph  dag_ll, dag_ur;

        if ( nosparsify )
            hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        else
            hlr::dag::sparsify_mode = hlr::dag::sparsify_sub_all;
    
        auto  L     = impl::matrix::copy_ll( *A, unit_diag );
        auto  L_inv = impl::matrix::copy(    *L );
        auto  U     = impl::matrix::copy_ur( *A );
        auto  U_inv = impl::matrix::copy(    *U );

        DBG::write( A.get(),     "M.mat", "M" );
        DBG::write( L.get(),     "A.mat", "A" );
        DBG::write( L_inv.get(), "B.mat", "B" );
            
        double  tmin = 0;
        double  tmax = 0;
        double  tsum = 0;
    
        std::cout << "    DAG" << std::endl;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();
        
            dag_ll = std::move( hlr::dag::gen_dag_invert_ll( L_inv.get(), unit_diag, impl::dag::refine ) );
            dag_ur = std::move( hlr::dag::gen_dag_invert_ur( U_inv.get(), general_diag, impl::dag::refine ) );
        
            toc = Time::Wall::since( tic );
        
            if ( verbose( 1 ) )
                std::cout << "      done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();
        
            if ( i < nbench-1 )
                dag_ll = std::move( hlr::dag::graph() );
        }// for
        
        if ( verbose( 1 ) )
        {
            if ( nbench > 1 )
                std::cout << "      time =   "
                          << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                          << std::endl;
            std::cout << "      #nodes = " << dag_ll.nnodes() << " / " << dag_ur.nnodes() << std::endl;
            std::cout << "      #edges = " << dag_ll.nedges() << " / " << dag_ur.nedges() << std::endl;
            std::cout << "      mem    = " << Mem::to_string( dag_ll.mem_size() ) << mem_usage() << std::endl;
        }// if
        
        if ( verbose( 3 ) )
        {
            dag_ll.print_dot( "inv_ll.dot" );
            dag_ur.print_dot( "inv_ur.dot" );
        }// if
        
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
            tic = Time::Wall::now();

            impl::dag::run( dag_ll, acc );
            impl::dag::run( dag_ur, acc );
        
            toc = Time::Wall::since( tic );

            std::cout << "      done in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *L, *L_inv );
        }// for

        DBG::write( L_inv.get(), "C.mat", "C" );
        
        if ( nbench > 1 )
            std::cout << "      time =   "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
        
        std::cout << "      mem    = "
                  << Mem::to_string( L_inv->byte_size() )
                  << " / "
                  << Mem::to_string( U_inv->byte_size() )
                  << mem_usage() << std::endl;
        std::cout << "      error  = " << term::ltred
                  << format( "%.4e" ) % inv_approx_2( L.get(), L_inv.get() )
                  << " / "
                  << format( "%.4e" ) % inv_approx_2( U.get(), U_inv.get() )
                  << term::reset << std::endl;

        // DBG::write( C.get(), "C.mat", "C" );
    }// else
}
