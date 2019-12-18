//
// Project     : HLib
// File        : tlr.hh
// Description : TLR-LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/tlr.hh"

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
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tlr::cluster( coord.get(), ntile );
    auto  bct     = cluster::tlr::blockcluster( ct.get(), ct.get() );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if
    
    {
        std::cout << term::bullet << term::bold << "LU ( TLR " << impl_name << " )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = timer::now();
        
        impl::tlr::lu< HLIB::real >( C.get(), acc );
        
        toc = timer::since( tic );
        
        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( inv_approx_2( A.get(), & A_inv ) ) << std::endl;

        hpro::write_matrix( C.get(), "LU.hm" );
    }

}
