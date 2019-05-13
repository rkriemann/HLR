//
// Project     : HLib
// File        : dag-seq.cc
// Description : sequential H-LU using DAG
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "cluster/H.hh"
#include "seq/matrix.hh"
#include "seq/dag.hh"
#include "dag/lu.hh"

//
// main function
//
template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic     = Time::Wall::now();
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
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( coeff.get() );
    auto  acc    = gen_accuracy();
    auto  A      = Matrix::Seq::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;

    DBG::write( A.get(), "A.hm", "A" );
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( DAG SEQ )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();

        auto  dag = HLR::DAG::gen_LU_dag( C.get() );
        
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
        
        HLR::DAG::Seq::run( dag, acc );
        
        toc = Time::Wall::since( tic );
        
        DBG::write( C.get(), "C.hm", "C" );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}

int
main ( int argc, char ** argv )
{
    return hlrmain( argc, argv );
}
