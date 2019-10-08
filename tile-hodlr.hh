//
// Project     : HLib
// File        : tile-hodlr.hh
// Description : geeric code for tile-based HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "common.hh"
#include "hlr/cluster/hodlr.hh"

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
    auto  ct      = cluster::hodlr::cluster( coord.get(), ntile );
    auto  bct     = cluster::hodlr::blockcluster( ct.get(), ct.get() );
    
    if ( verbose( 3 ) )
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( pcoeff.get() );
    auto  acc    = gen_accuracy();
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if
    
    {
        std::cout << term::bullet << term::bold << "LU ( HODLR " << impl_name << " )" << term::reset << std::endl;
        
        auto  C = impl::matrix::copy( *A );

        std::vector< double >  runtime;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = Time::Wall::now();
            
            impl::tile::hodlr::lu< HLIB::real >( C.get(), acc, ntile );
            
            toc = Time::Wall::since( tic );

            std::cout << "    done in " << format_time( toc ) << std::endl;
            
            runtime.push_back( toc.seconds() );

            if ( i < nbench-1 )
                impl::matrix::copy_to( *A, *C );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        std::cout << "    mem   = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error = " << format_error( inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }
}
