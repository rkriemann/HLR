//
// Project     : HLR
// File        : tileh-seq.cc
// Description : sequential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrixSum.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/tileh.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/norm.hh"

using namespace hlr;

//
// main function
//
void
program_main ()
{
    using value_t = hpro::real; // typename problem_t::value_t;

    const auto  tileh_lvl = std::min< size_t >( std::log2( n ) - std::log2( ntile ), 4 );

    auto  tic     = timer::now();
    auto  problem = gen_problem();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tileh::cluster( coord.get(), ntile, tileh_lvl );
    auto  bct     = cluster::tileh::blockcluster( ct.get(), ct.get() );

    std::cout << "    tiling = " << ct->root()->nsons() << " × " << ct->root()->nsons() << std::endl;
    
    if ( verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if

    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    {
        auto  C1 = impl::matrix::copy( *A );
        auto  C2 = impl::matrix::copy( *A );

        tic = timer::now();
        
        impl::multiply< value_t >( value_t(1), hpro::apply_normal, *A, hpro::apply_normal, *A, *C1, acc );

        toc = timer::since( tic );
        std::cout << "    mult in " << format_time( toc ) << std::endl;
        
        tic = timer::now();

        hpro::multiply< value_t >( value_t(1), hpro::apply_normal, A.get(), hpro::apply_normal, A.get(), value_t(1), C2.get(), acc );

        toc = timer::since( tic );
        std::cout << "    mult in " << format_time( toc ) << std::endl;

        auto  diff = hpro::matrix_sum( 1.0, C1.get(), -1.0, C2.get() );

        std::cout << "    error = " << format_error( hlr::seq::norm::norm_2( *diff ) ) << std::endl;
    }
    
    {
        auto  C = impl::matrix::copy( *A );
        
        if ( HLIB::CFG::Arith::use_dag )
        {
            std::cout << term::bullet << term::bold << "LU ( Tile-H DAG " << impl_name
                      << ", " << acc.to_string()
                      << " )" << term::reset << std::endl;

            // no sparsification
            hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        
            tic = timer::now();
        
            // auto  dag = std::move( dag::gen_dag_lu_oop_auto( *C, nseq, impl::dag::refine ) );
            auto  dag = std::move( dag::gen_dag_lu_tileh( *C, nseq, impl::dag::refine, impl::dag::run ) );
            
            toc = timer::since( tic );
            
            std::cout << "    DAG in   " << format_time( toc ) << std::endl;
            std::cout << "    #nodes = " << dag.nnodes() << std::endl;
            std::cout << "    #edges = " << dag.nedges() << std::endl;
            std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
            
            if ( verbose( 3 ) )
                dag.print_dot( "lu.dot" );
            
            tic = timer::now();
            
            impl::dag::run( dag, acc );

            toc = timer::since( tic );
        }// if
        else
        {
            std::cout << term::bullet << term::bold << "LU ( Tile-H flat " << impl_name
                      << ", " << acc.to_string()
                      << " )" << term::reset << std::endl;

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
