//
// Project     : HLib
// File        : dag-hpx.cc
// Description : DAG based H-LU using HPX
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpx/hpx_init.hpp>

#include "common.inc"
#include "cluster/H.hh"
#include "hpx/matrix.hh"
#include "hpx/dag.hh"
#include "dag/lu.hh"

using namespace HLR;

//
// main function
//
template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic     = Time::Wall::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< TMatrix >();

    if ( matrix == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = H::cluster( coord.get(), ntile );
        auto  bct     = H::blockcluster( ct.get(), ct.get() );
    
        if ( verbose( 3 ) )
        {
            TPSBlockClusterVis   bc_vis;
        
            bc_vis.id( true ).print( bct->root(), "bct" );
            print_vtk( coord.get(), "coord" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = std::make_unique< TACAPlus< value_t > >( pcoeff.get() );
        auto  acc    = gen_accuracy();

        A = Matrix::HPX::build( bct->root(), *pcoeff, *lrapx, acc );
    }// if
    else
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrix
                  << ( eps > 0 ? HLIB::to_string( ", ε = %.2e", eps ) : HLIB::to_string( ", k = %d", k ) )
                  << std::endl;

        A = read_matrix( matrix );
        A = Matrix::HPX::copy( *A ); // for spreading memory usage
    }// else

    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( DAG HPX )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();

        auto  dag = HLR::DAG::gen_LU_dag( C.get(), HLR::DAG::HPX::refine );
        
        toc = Time::Wall::since( tic );

        if ( verbose( 2 ) )
        {
            std::cout << "  dag in      " << toc << std::endl;
            std::cout << "    #nodes  = " << dag.nnodes() << std::endl;
            std::cout << "    #edges  = " << dag.nedges() << std::endl;
        }// if
        
        if ( verbose( 3 ) )
            dag.print_dot( "lu.dot" );

        tic = Time::Wall::now();
        
        HLR::DAG::HPX::run( dag, acc );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}

int
hpx_main ( int argc, char ** argv )
{
    hlrmain( argc, argv );
    
    return hpx::finalize();
}

int
main ( int argc, char ** argv )
{
    return hpx::init( argc, argv );
}
