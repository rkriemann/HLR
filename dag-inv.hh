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
    std::cout << "    mem    = " << Mem::to_string( A->byte_size() ) << " / " << Mem::to_string( Mem::usage() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    std::cout << term::bullet << term::bold << "Inversion (DAG)" << term::reset << ", " << acc.to_string() << std::endl;

    hlr::dag::graph  dag_ll, dag_ur;

    hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
    
    auto  L     = impl::matrix::copy_ll( *A );
    auto  L_inv = L->copy();
    auto  U     = impl::matrix::copy_ur( *A );
    auto  U_inv = U->copy();

    double  tmin = 0;
    double  tmax = 0;
    double  tsum = 0;
    
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = Time::Wall::now();
        
        dag_ll = std::move( hlr::dag::gen_dag_invert_ll( L_inv.get(), unit_diag, impl::dag::refine ) );
        
        toc = Time::Wall::since( tic );
        
        if ( verbose( 1 ) )
            std::cout << "  DAG in     " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
        
        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();
        
        if ( i < nbench-1 )
            dag_ll = std::move( hlr::dag::graph() );
    }// for
        
    if ( verbose( 1 ) )
    {
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                      << std::endl;
        std::cout << "    #nodes = " << dag_ll.nnodes() << std::endl;
        std::cout << "    #edges = " << dag_ll.nedges() << std::endl;
        std::cout << "    mem    = " << Mem::to_string( dag_ll.mem_size() ) << " / " << Mem::to_string( Mem::usage() ) << std::endl;
    }// if
        
    if ( verbose( 3 ) )
        dag_ll.print_dot( "inv_ll.dot" );
        
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

        impl::dag::run( dag_ll, acc );
        
        toc = Time::Wall::since( tic );

        std::cout << "  inversion in " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;

        tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
        tmax  = std::max( tmax, toc.seconds() );
        tsum += toc.seconds();

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *L, *L_inv );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % tmin % ( tsum / double(nbench) ) % tmax
                  << std::endl;
        
    std::cout << "    mem    = " << Mem::to_string( L_inv->byte_size() ) << " / " << Mem::to_string( Mem::usage() ) << std::endl;
    std::cout << "    error  = " << term::ltred << format( "%.4e" ) % inv_approx_2( L.get(), L_inv.get() ) << term::reset << std::endl;

    // DBG::write( C.get(), "C.mat", "C" );
}
