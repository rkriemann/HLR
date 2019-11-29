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

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
mymain ( int, char ** )
{
    using value_t = typename problem_t::value_t;
    
    auto  tic     = timer::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tileh::cluster( coord.get(), ntile, 4 );
    auto  bct     = cluster::tileh::blockcluster( ct.get(), ct.get() );

    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if

    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;

    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
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
        
            tic = timer::now();
        
            auto  dag = std::move( dag::gen_dag_lu_oop_auto( *C, impl::dag::refine ) );
            
            toc = timer::since( tic );
            
            std::cout << "    DAG in  " << format_time( toc ) << std::endl;
            std::cout << "    mem   = " << format_mem( dag.mem_size() ) << std::endl;
            
            tic = timer::now();
            
            impl::dag::run( dag, acc );

            toc = timer::since( tic );
        }// if
        else
        {
            tic = timer::now();
        
            impl::tileh::lu< HLIB::real >( C.get(), acc );
            
            toc = timer::since( tic );
        }// else
            
        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    LU in   " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }

}
