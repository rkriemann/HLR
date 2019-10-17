//
// Project     : HLib
// File        : tileh-seq.cc
// Description : sequential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "hlr/cluster/tileh.hh"
#include "hlr/dag/lu.hh"

//
// main function
//
template < typename problem_t >
void
mymain ( int, char ** )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic     = Time::Wall::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tileh::cluster( coord.get(), ntile, 4 );
    auto  bct     = cluster::tileh::blockcluster( ct.get(), ct.get() );

    if ( verbose( 3 ) )
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if

    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset << std::endl;
    std::cout << "    mem   = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;

    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "tileh_A" );
    }// if
    
    {
        std::cout << term::bullet << term::bold << "LU ( Tile-H " << impl_name
                  << ", " << acc.to_string()
                  << " )" << term::reset << std::endl;

        auto  C = impl::matrix::copy( *A );
        
        if ( HLIB::CFG::Arith::use_dag )
        {
            // no sparsification
            hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        
            tic = Time::Wall::now();
        
            auto  dag = std::move( dag::gen_dag_lu_oop_auto( *C, impl::dag::refine ) );
            
            toc = Time::Wall::since( tic );
            
            std::cout << "    DAG in  " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset() << std::endl;
            std::cout << "    mem   = " << Mem::to_string( dag.mem_size() ) << mem_usage() << std::endl;
            
            tic = Time::Wall::now();
            
            impl::dag::run( dag, acc );

            toc = Time::Wall::since( tic );
        }// if
        else
        {
            tic = Time::Wall::now();
        
            impl::tileh::lu< HLIB::real >( C.get(), acc );
            
            toc = Time::Wall::since( tic );
        }// else
            
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    LU in   " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset() << std::endl;
        std::cout << "    mem   = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;
        std::cout << "    error = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << term::reset << std::endl;
    }

}
