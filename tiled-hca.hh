//
// Project     : HLR
// File        : tiled-hca.hh
// Description : generic code for tile-based HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/algebra/mul_vec.hh>
#include <hpro/bem/TLaplaceBF.hh>
#include <hpro/bem/TRefinableGrid.hh>
#include <hpro/bem/TBFCoeffFn.hh>
#include <hpro/matrix/TMatrixSum.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/vector/tiled_scalarvector.hh"
#include "hlr/seq/norm.hh"
#include "hlr/seq/arith_tiled_v2.hh"
#include "hlr/bem/hca.hh"
#include "hlr/bem/tiled_hca.hh"

using namespace hlr;
using namespace hpro;

using function_space = hpro::TConstFnSpace;

//
// main function
//
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
    
    auto  ct      = cluster::h::cluster( coord.get(), ntile );
    auto  bct     = cluster::h::blockcluster( ct.get(), ct.get() );
    
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
    auto  aca    = std::make_unique< TACAPlus< value_t > >( pcoeff );
    auto  svd    = std::make_unique< TSVDLRApx< value_t > >( pcoeff );
    auto  dense  = std::make_unique< TDenseLRApx< value_t > >( pcoeff );

    //////////////////////////////////////////////////////////////////////
    
    const bool                        build_ref = (( ref != "" ) && ( ref != "none" ));
    std::unique_ptr< hpro::TMatrix >  REF;
    real_t                            norm_REF = 0;

    if ( build_ref )
    {
        std::cout << "  " << term::bullet << term::bold << "Reference" << term::reset << std::endl;

        tic = timer::now();
    
        REF = impl::matrix::build( bct->root(), *pcoeff, *svd, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( REF->byte_size() ) << std::endl;

        norm_REF = hlr::seq::norm::norm_2( *REF );

        std::cout << "    norm   = " << format_norm( norm_REF ) << std::endl;
    }// if
    
    //////////////////////////////////////////////////////////////////////
    
    std::cout << "  " << term::bullet << term::bold << "HCA" << term::reset << std::endl;
    
    tic = timer::now();
    
    auto  B      = impl::matrix::build( bct->root(), *pcoeff, *hcalr, acc, nseq );

    toc = timer::since( tic );
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;
    std::cout << "    norm   = " << format_norm( hlr::seq::norm::norm_2( *B ) ) << std::endl;
    
    if ( build_ref )
    {
        auto  diff_hca  = hpro::matrix_sum( value_t(1), REF.get(), value_t(-1), B.get() );
        auto  error_HCA = hlr::seq::norm::norm_2( *diff_hca );

        std::cout << "    error  = " << format_error( error_HCA )
                  << " / "
                  << format_error( error_HCA / norm_REF )
                  << std::endl;
    }// if    

    //////////////////////////////////////////////////////////////////////
    
    std::cout << "  " << term::bullet << term::bold << "tHCA" << term::reset << std::endl;
    
    tic = timer::now();
    
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *thcalr, acc, nseq );

    toc = timer::since( tic );
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    std::cout << "    norm   = " << format_norm( hlr::seq::norm::norm_2( *A ) ) << std::endl;

    if ( build_ref )
    {
        auto  diff_thca  = hpro::matrix_sum( value_t(1), REF.get(), value_t(-1), A.get() );
        auto  error_tHCA = hlr::seq::norm::norm_2( *diff_thca );
    
        std::cout << "    error  = " << format_error( error_tHCA )
                  << " / "
                  << format_error( error_tHCA / norm_REF )
                  << std::endl;
    }// if

    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if
}
