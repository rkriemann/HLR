//
// Project     : HLR
// File        : tiled-hca.hh
// Description : generic code for tile-based HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TGeomAdmCond.hh>
#include <hpro/algebra/mul_vec.hh>
#include <hpro/bem/TLaplaceBF.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/bem/TBFCoeffFn.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/vector/tiled_scalarvector.hh"
#include "hlr/seq/norm.hh"
#include "hlr/seq/arith_tiled_v2.hh"
#include "hlr/bem/hca.hh"
#include "hlr/bem/tiled_hca.hh"
#include "hlr/arith/multiply.hh"

using namespace hlr;
using namespace hpro;

using function_space = hpro::TConstFnSpace;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using  value_t = hpro::real; // typename problem_t::value_t;
    using  real_t  = typename hpro::real_type< value_t >::type_t;
    
    std::vector< double >  runtime;
    auto                   tic = timer::now();
    auto                   toc = timer::since( tic );

    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    grid = " << gridfile
              << std::endl;
    
    auto  acc     = gen_accuracy();
    auto  grid    = make_grid( gridfile );
    auto  fnspace = std::make_unique< function_space >( grid.get() );
    auto  coord   = fnspace->build_coord();

    std::cout << "    dims = " << coord->ncoord() << " × " << coord->ncoord() << std::endl;
    
    auto  ct      = cluster::h::cluster( *coord, ntile );
    auto  bct     = cluster::h::blockcluster( *ct, *ct );
    
    if ( verbose( 3 ) )
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // Tiled HCA
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "H-matrix construction" << term::reset
              << ", " << acc.to_string()
              << ", ntile = " << ntile
              << std::endl;
    
    auto  tile_map = matrix::setup_tiling( * ct->root() );

    auto  bf     = new TLaplaceSLPBF( fnspace.get(), fnspace.get(), 4 );
    auto  coeff  = new TBFCoeffFn( bf );
    auto  pcoeff = new TPermCoeffFn( coeff, ct->perm_i2e(), ct->perm_i2e() );
    auto  genfn  = new TLaplaceSLPGenFn( fnspace.get(),
                                         fnspace.get(),
                                         ct->perm_i2e(),
                                         ct->perm_i2e(),
                                         4 ); // quad order
    auto  hca    = new impl::bem::hca( *pcoeff, *genfn, cmdline::eps / 100.0, 5 );
    auto  hcalr  = new bem::hca_lrapx( *hca );
    auto  thca   = new impl::bem::tiled_hca( *pcoeff, *genfn, cmdline::eps / 100.0, 5, tile_map, tile_map );
    auto  thcalr = new bem::hca_lrapx( *thca );
    auto  aca    = new bem::aca_lrapx( *pcoeff );
    auto  svd    = std::make_unique< TSVDLRApx< value_t > >( pcoeff );
    auto  dense  = std::make_unique< TDenseLRApx< value_t > >( pcoeff );

    //////////////////////////////////////////////////////////////////////
    
    const bool                        build_ref = (( ref != "" ) && ( ref != "none" ));
    std::unique_ptr< hpro::TMatrix >  M_ref;
    real_t                            norm_ref = 0;

    if ( build_ref )
    {
        std::cout << "  " << term::bullet << term::bold << "Reference" << term::reset
                  << " (" << ref << ")" << std::endl;
        
        tic = timer::now();
    
        if      ( ref == "svd" ) M_ref = impl::matrix::build( bct->root(), *pcoeff, *svd, acc, nseq );
        else if ( ref == "aca" ) M_ref = impl::matrix::build( bct->root(), *pcoeff, *aca, acc, nseq );
        else                     M_ref = impl::matrix::build( bct->root(), *pcoeff, *dense, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( M_ref->byte_size() ) << std::endl;

        norm_ref = hlr::seq::norm::norm_2( *M_ref );

        std::cout << "    norm   = " << format_norm( norm_ref ) << std::endl;
    }// if

    //////////////////////////////////////////////////////////////////////

    // {
    //     std::cout << "  " << term::bullet << term::bold << "ACA" << term::reset << std::endl;
        
    //     tic = timer::now();
    
    //     auto  M_aca = impl::matrix::build( bct->root(), *pcoeff, *aca, acc, nseq );

    //     toc = timer::since( tic );
    //     std::cout << "    done in  " << format_time( toc ) << std::endl;
    //     std::cout << "    mem    = " << format_mem( M_aca->byte_size() ) << std::endl;
    //     std::cout << "    norm   = " << format_norm( hlr::seq::norm::norm_2( *M_aca ) ) << std::endl;
    // }// if
    
    //////////////////////////////////////////////////////////////////////

    {
        std::cout << "  " << term::bullet << term::bold << "HCA" << term::reset << std::endl;

        auto  M_hca = std::unique_ptr< hpro::TMatrix >();

        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            M_hca = impl::matrix::build( bct->root(), *pcoeff, *hcalr, acc, nseq );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
        
            std::cout << "    done in  " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                M_hca.reset( nullptr );
        }// for
    
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        std::cout << "    mem    = " << format_mem( M_hca->byte_size() ) << std::endl;
        std::cout << "    norm   = " << format_norm( hlr::seq::norm::norm_2( *M_hca ) ) << std::endl;
    
        if ( build_ref )
        {
            auto  diff_hca  = hpro::matrix_sum( value_t(1), M_ref.get(), value_t(-1), M_hca.get() );
            auto  error_HCA = hlr::seq::norm::norm_2( *diff_hca );

            std::cout << "    error  = " << format_error( error_HCA )
                      << " / "
                      << format_error( error_HCA / norm_ref )
                      << std::endl;
        }// if    

        runtime.clear();
        
        //
        // mat-mul
        //

        std::cout << "    " << term::dash << "mat-mul" << std::endl;

        {
            auto  A   = M_hca.get();
            auto  C   = impl::matrix::copy( *M_hca );
            auto  REF = hpro::matrix_product( A, A );

            for ( int i = 0; i < nbench; ++i )
            {
                C->scale( real_t(0) );
                
                tic = timer::now();
        
                hlr::arith::multiply( value_t(1), *A, *A, *C, acc );
                // impl::multiply< value_t >( value_t(1), hpro::apply_normal, *A, hpro::apply_normal, *A, *C1, acc );

                toc = timer::since( tic );
                std::cout << "    mult in " << format_time( toc ) << std::endl;

                runtime.push_back( toc.seconds() );
            }// for
        
            if ( nbench > 1 )
                std::cout << "  runtime = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;
            runtime.clear();

            auto  diff = hpro::matrix_sum( 1.0, REF.get(), -1.0, C.get() );

            std::cout << "    error = " << format_error( hlr::seq::norm::norm_2( *diff ) ) << std::endl;
        }            

        //
        // mat-vec benchmark
        //

        std::cout << "    " << term::dash << "mat-vec" << std::endl;

        blas::Vector< value_t >  x( M_hca->ncols() );
        blas::Vector< value_t >  y( M_hca->nrows() );

        blas::fill( value_t(1), x );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                impl::mul_vec( 1.0, hpro::apply_normal, *M_hca, x, y );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
        
            std::cout << "    mvm in   " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                blas::fill( value_t(0), y );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        runtime.clear();
    }
    
    //////////////////////////////////////////////////////////////////////

    {
        std::cout << "  " << term::bullet << term::bold << "tHCA" << term::reset << std::endl;
    
        auto  M_thca = std::unique_ptr< hpro::TMatrix >();
    
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            M_thca = impl::matrix::build( bct->root(), *pcoeff, *thcalr, acc, nseq );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );

            std::cout << "    done in  " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                M_thca.reset( nullptr );
        }// for
    
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        std::cout << "    mem    = " << format_mem( M_thca->byte_size() ) << std::endl;
        std::cout << "    norm   = " << format_norm( hlr::seq::norm::norm_2( *M_thca ) ) << std::endl;

        if ( build_ref )
        {
            auto  diff_thca  = hpro::matrix_sum( value_t(1), M_ref.get(), value_t(-1), M_thca.get() );
            auto  error_tHCA = hlr::seq::norm::norm_2( *diff_thca );
    
            std::cout << "    error  = " << format_error( error_tHCA )
                      << " / "
                      << format_error( error_tHCA / norm_ref )
                      << std::endl;
        }// if

        if ( verbose( 3 ) )
        {
            TPSMatrixVis  mvis;
        
            mvis.svd( false ).id( true ).print( M_thca.get(), "A" );
        }// if

        //
        // mat-mul
        //

        std::cout << "    " << term::dash << "mat-mul" << std::endl;

        {
            auto  B = impl::matrix::copy_tiled< double >( *M_thca, ntile );
            auto  X = matrix::tile_storage< value_t >();

            // look for leafs in <ct> and initialize data
            std::list< hpro::TCluster * >  nodes{ ct->root() };

            while ( ! nodes.empty() )
            {
                auto  node = nodes.front();

                nodes.pop_front();
                
                if ( node->is_leaf() )
                {
                    auto  M = matrix::tile< value_t >( node->size(), 5 );

                    blas::fill( value_t(1), M );
                    
                    X[ *node ] = std::move( M );
                }// if
                else
                {
                    for ( uint  i = 0; i < node->nsons(); ++i )
                        nodes.push_front( node->son(i) );
                }// else
            }// while
            
            auto  Y = seq::tiled2::multiply( value_t(1), hpro::apply_normal, *B, X );
        }
    
        
        // auto  A = M_thca.get();
        // auto  B = impl::matrix::copy( *M_thca );
        // auto  C = impl::matrix::copy( *M_thca );

        // seq::arith::tiled2::multiply( value_t(1), *A, *B, value_t(1), *C, acc );
        
        //
        // mat-vec benchmark
        //

        std::cout << "    " << term::dash << "mat-vec" << std::endl;
        
        vector::tiled_scalarvector< value_t >  x( M_thca->col_is(), tile_map );
        vector::tiled_scalarvector< value_t >  y( M_thca->row_is(), tile_map );

        x.fill( real_t(1) );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                impl::tiled2::mul_vec( 1.0, hpro::apply_normal, *M_thca, x, y );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
        
            std::cout << "    mvm in   " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                y.fill( real_t(0) );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        runtime.clear();
    }
}
