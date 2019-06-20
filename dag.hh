//
// Project     : HLib
// File        : dag.hh
// Description : main function for DAG examples
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "hlr/cluster/h.hh"
#include "hlr/matrix/level_matrix.hh"
#include "hlr/dag/lu.hh"
#include "hlr/arith/lu.hh"

namespace hlr { namespace dag {

extern std::atomic< size_t >  collisions;
extern bool                   lock_nodes;

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

        if ( A->nrows() != n )
            std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = read_matrix( matrixfile );

        std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::copy( *A );
    }// else

    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    if ( levelwise )
    {
        std::cout << term::bullet << term::bold << "Level LU (DAG)" << term::reset << std::endl;

        auto  C = ( onlydag ? std::move( A ) : A->copy() );

        C->set_hierarchy_data();

        hlr::dag::graph  dag;
        double           tmin = 0;
        double           tmax = 0;
        double           tsum = 0;
        
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();

            dag = std::move( hlr::dag::gen_dag_lu_lvl( *C ) );
            
            toc = Time::Wall::since( tic );
            
            if ( verbose( 2 ) )
            {
                std::cout << "  dag in     " << boost::format( "%.3e" ) % toc.seconds() << std::endl;

                if ( hlr::dag::lock_nodes )
                    std::cout << "    #coll  = " << hlr::dag::collisions << std::endl;
            }// if

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();

            if ( i < nbench-1 )
                dag = std::move( hlr::dag::graph() );
        }// for
        
        if ( verbose( 2 ) )
        {
            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << boost::format( "%.3e / %.3e / %.3e" ) % tmin % ( tsum / double(nbench) ) % tmax
                          << std::endl;
            std::cout << "    #nodes = " << dag.nnodes() << std::endl;
            std::cout << "    #edges = " << dag.nedges() << std::endl;
        }// if
        
        if ( verbose( 3 ) )
            dag.print_dot( "lvllu.dot" );
        
        if ( ! onlydag )
        {
            tic = Time::Wall::now();
        
            impl::dag::run( dag, acc );
        
            toc = Time::Wall::since( tic );
        
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
            std::cout << "  LU in      " << toc << std::endl;
            std::cout << "    error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
        }// if
    }
    else
    {
        std::cout << term::bullet << term::bold << "LU ( DAG " << impl_name
                  << ", " << acc.to_string() << " )" << term::reset << std::endl;
        
        auto  C = ( onlydag ? std::move( A ) : A->copy() );
        
        hlr::dag::graph  dag;
        double           tmin = 0;
        double           tmax = 0;
        double           tsum = 0;
        
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();

            dag = std::move( hlr::dag::gen_dag_lu_rec( C.get(), impl::dag::refine ) );
        
            toc = Time::Wall::since( tic );
            
            if ( verbose( 2 ) )
            {
                std::cout << "  dag in     " << boost::format( "%.3e" ) % toc.seconds() << std::endl;

                if ( hlr::dag::lock_nodes )
                    std::cout << "    #coll  = " << hlr::dag::collisions << std::endl;
            }// if

            tmin  = ( tmin == 0 ? toc.seconds() : std::min( tmin, toc.seconds() ) );
            tmax  = std::max( tmax, toc.seconds() );
            tsum += toc.seconds();

            if ( i < nbench-1 )
                dag = std::move( hlr::dag::graph() );
        }// for
        
        if ( verbose( 2 ) )
        {
            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << boost::format( "%.3e / %.3e / %.3e" ) % tmin % ( tsum / double(nbench) ) % tmax
                          << std::endl;
            std::cout << "    #nodes = " << dag.nnodes() << std::endl;
            std::cout << "    #edges = " << dag.nedges() << std::endl;
        }// if
        
        if ( verbose( 3 ) )
            dag.print_dot( "lu.dot" );

        if ( ! onlydag )
        {
            tic = Time::Wall::now();
        
            impl::dag::run( dag, acc );
        
            toc = Time::Wall::since( tic );
        
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            std::cout << "  LU in      " << toc << std::endl;
            std::cout << "    error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
        }// if
    }

}
