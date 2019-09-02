//
// Project     : HLib
// File        : tileh-seq.cc
// Description : sequential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.inc"
#include "hlr/cluster/tileh.hh"
#include "hlr/seq/norm.hh"

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
    std::cout << "    norm  = " << seq::norm::norm_F( *A ) << std::endl;
    std::cout << "    norm  = " << seq::norm::norm_F( 1.0, *A, -1.0, *A ) << std::endl;
    std::cout << "    norm  = " << HLIB::diff_norm_F( A.get(), A.get() ) << std::endl;

    DBG::write( A.get(), "A.mat", "A" );
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "tileh_A" );
    }// if
    
    {
        std::cout << term::bullet << term::bold << "LU ( Tile-H " << impl_name
                  << ", " << acc.to_string()
                  << " )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        impl::tileh::lu< HLIB::real >( C.get(), acc );
        
        toc = Time::Wall::since( tic );
        
        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    done in " << term::ltcyan << format( "%.3e s" ) % toc.seconds() << term::reset() << std::endl;
        std::cout << "    mem   = " << Mem::to_string( A->byte_size() ) << mem_usage() << std::endl;
        std::cout << "    error = " << term::ltred << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << term::reset << std::endl;
    }

}
