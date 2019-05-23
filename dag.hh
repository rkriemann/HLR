//
// Project     : HLib
// File        : dag.hh
// Description : main function for DAG examples
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "hlr/cluster/h.hh"
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

extern std::atomic< size_t >  collisions;

} }// namespace hlr::dag

//
// main function
//
template < typename problem_t >
void
dag_main ( int,
           char **,
           const std::string &  name )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic = Time::Wall::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< TMatrix >();

    if ( matrixfile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = h::cluster( coord.get(), ntile );
        auto  bct     = h::blockcluster( ct.get(), ct.get() );
    
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
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = read_matrix( matrixfile );

        std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;

        // for spreading memory usage
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
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( DAG " << name
                  << ", " << acc.to_string() << " )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();

        auto  dag = hlr::dag::gen_lu_dag( C.get(), impl::dag::refine );
        
        toc = Time::Wall::since( tic );

        if ( verbose( 2 ) )
        {
            std::cout << "  dag in      " << toc << std::endl;
            std::cout << "    #nodes  = " << dag.nnodes() << std::endl;
            std::cout << "    #edges  = " << dag.nedges() << std::endl;
            std::cout << "    #coll   = " << hlr::dag::collisions << std::endl;
        }// if
        
        if ( verbose( 3 ) )
            dag.print_dot( "lu.dot" );

        if ( onlydag )
            return;
        
        tic = Time::Wall::now();
        
        impl::dag::run( dag, acc );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << toc << std::endl;
        std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }

}
