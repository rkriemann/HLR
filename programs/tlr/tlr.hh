//
// Project     : HLR
// File        : tlr.hh
// Description : TLR-LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <fstream>

#include "common.hh"
#include "common-main.hh"

#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/io/TClusterBasisVis.hh>
#include <hpro/algebra/mat_conv.hh>

#include "hlr/cluster/tlr.hh"
#include "hlr/cluster/mblr.hh"
#include "hlr/dag/lu.hh"
#include "hlr/bem/aca.hh"

using namespace hlr;

uint64_t
get_flops ( const std::string &  method );

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    
    auto  tic     = timer::now();
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::mblr::cluster( *coord, ntile, nlvl );
    auto  bct     = cluster::tlr::blockcluster( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).print( A.get(), "A" );
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "LU ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;
    
    if ( false )
    {
        std::cout << "  " << term::bullet << " recursive" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        auto  C = impl::matrix::copy( *A );
        
        for ( int i = 0; i < nbench; ++i )
        {
            blas::reset_flops();

            tic = timer::now();
        
            impl::tlr::lu< HLIB::real >( C.get(), acc );
        
            toc = timer::since( tic );
            std::cout << "  LU in      " << format_time( toc ) << std::endl;
            
            flops.push_back( get_flops( "lu" ) );
            runtime.push_back( toc.seconds() );

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *C );
        }// for
        
        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }

    if ( false )
    {
        std::cout << "  " << term::bullet << " DAG" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        auto  C = impl::matrix::copy( *A );

        C->set_hierarchy_data();

        tic = timer::now();
        
        auto  dag = std::move( dag::gen_dag_lu_lvl( *C, nseq ) );
            
        toc = timer::since( tic );
            
        std::cout << "    DAG in   " << format_time( toc ) << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
        
        if ( verbose( 3 ) )
            dag.print_dot( "lu.dot" );
        
        for ( int i = 0; i < nbench; ++i )
        {
            blas::reset_flops();

            tic = timer::now();
            
            impl::dag::run( dag, acc );

            toc = timer::since( tic );
            std::cout << "  LU in      " << format_time( toc ) << std::endl;
            
            flops.push_back( get_flops( "lu" ) );
            runtime.push_back( toc.seconds() );

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *C );
        }// for

        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    //
    // H² conversion
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "H² ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;

    {
        std::cout << "  " << term::bullet << " cluster basis" << std::endl;
        
        hpro::THClusterBasisBuilder< value_t >  bbuilder;

        tic = timer::now();

        auto  [ rowcb, colcb ] = bbuilder.build( A.get(), acc );

        toc = timer::since( tic );
        
        std::cout << "    time   = " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( rowcb->byte_size() ) << " / " << format_mem( colcb->byte_size() ) << std::endl;

        if ( verbose( 3 ) )
        {
            hpro::TPSClusterBasisVis< value_t >  cbvis;
            std::ofstream                        rowout( "cb.eps" );

            cbvis.visualise( rowcb.get(), rowout );
        }// if

        std::cout << "  " << term::bullet << " H → H²" << std::endl;
        
        tic = timer::now();

        auto  A2 = hpro::to_h2( A.get(), rowcb.get(), colcb.get() );

        toc = timer::since( tic );
        
        std::cout << "    time   = " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;
        std::cout << "    |A|    = " << format_norm( hpro::norm_F( A.get() ) ) << std::endl;
        std::cout << "    |A²|   = " << format_norm( hpro::norm_F( A2.get() ) ) << std::endl;
        // std::cout << "    |A-A'|₂= " << format_error( hpro::diff_norm_2( A.get(), A2.get() ) ) << std::endl;

        if ( verbose( 3 ) )
        {
            hpro::TPSMatrixVis  mvis;
        
            mvis.svd( false ).print( A2.get(), "A2" );
        }// if
    }
}

//
// return FLOPs for standard settings
//
uint64_t
get_flops ( const std::string &  method )
{
    #if HLIB_COUNT_FLOPS == 1

    return blas::get_flops();

    #else

    if ( ntile == 256 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 0;
            if ( gridfile == "sphere-6" ) return 0;
            if ( gridfile == "sphere-7" ) return 0;
            if ( gridfile == "sphere-8" ) return 0;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 0;
            if ( gridfile == "sphere-6" ) return 0;
            if ( gridfile == "sphere-7" ) return 0;
            if ( gridfile == "sphere-8" ) return 0;
        }// if
    }// if
    else if ( ntile == 128 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 0;
            if ( gridfile == "sphere-6" ) return 0;
            if ( gridfile == "sphere-7" ) return 0;
            if ( gridfile == "sphere-8" ) return 0;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 0;
            if ( gridfile == "sphere-6" ) return 0;
            if ( gridfile == "sphere-7" ) return 0;
            if ( gridfile == "sphere-8" ) return 0;
        }// if
    }// if
    else if ( ntile == 64 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 0;
            if ( gridfile == "sphere-6" ) return 0;
            if ( gridfile == "sphere-7" ) return 0;
            if ( gridfile == "sphere-8" ) return 0;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 0;
            if ( gridfile == "sphere-6" ) return 0;
            if ( gridfile == "sphere-7" ) return 0;
            if ( gridfile == "sphere-8" ) return 0;
        }// if
    }// if

    #endif

    return 0;
}
