//
// Project     : HLR
// Program     : approx-lu
// Description : testing approximation algorithms for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <fstream>

#include <hlr/utils/likwid.hh>

#include "hlr/arith/norm.hh"
#include "hlr/bem/aca.hh"
#include <hlr/matrix/print.hh>
#include <hlr/matrix/product.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/approx/lanczos.hh>
#include <hlr/approx/randlr.hh>
#include "hlr/dag/lu.hh"
#include "hlr/utils/io.hh"

#include "hlr/seq/arith.hh"
#include "hlr/seq/arith_accu.hh"
#include "hlr/seq/arith_lazy.hh"

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

uint64_t
get_flops ( const std::string &  method );

//
// standard LU
//
template < typename value_t,
           typename approx_t >
void
lu_std ( const Hpro::TMatrix< value_t > &  A,
         const Hpro::TTruncAcc &           acc,
         const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic    = timer::now();
    auto  toc    = timer::since( tic );
    auto  C      = impl::matrix::copy( A );
    auto  tstart = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hlustd" );
            
        impl::lu< value_t >( *C, acc, approx );

        LIKWID_MARKER_STOP( "hlustd" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  A_inv = matrix::luinv_eval( *C );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( norm::inv_error_2( A, A_inv ) ) << std::endl;
}

//
// standard LU
//
template < typename value_t,
           typename approx_t >
void
lu_std_dag ( const Hpro::TMatrix< value_t > &  A,
             const Hpro::TTruncAcc &           acc,
             const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    std::vector< double >  runtime, flops;

    auto  apx    = approx_t();
    auto  tic    = timer::now();
    auto  toc    = timer::since( tic );
    auto  C      = impl::matrix::copy( A );
    auto  lu_dag = hlr::dag::gen_dag_lu( *C, nseq, impl::dag::refine, apx );

    if ( Hpro::verbose( 3 ) )
        lu_dag.print_dot( "lu.dot" );
    
    auto  tstart = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );

        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hlustddag" );

        impl::dag::run( lu_dag, acc );

        LIKWID_MARKER_STOP( "hlustddag" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  A_inv = matrix::luinv_eval( *C );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( norm::inv_error_2( A, A_inv ) ) << std::endl;
}

//
// accumulator based LU
//
template < typename value_t,
           typename approx_t >
void
lu_accu ( const Hpro::TMatrix< value_t > &  A,
          const Hpro::TTruncAcc &           acc,
          const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic    = timer::now();
    auto  toc    = timer::since( tic );
    auto  C      = impl::matrix::copy( A );
    auto  tstart = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hluaccu" );
            
        impl::accu::lu< value_t >( *C, acc, approx );

        LIKWID_MARKER_STOP( "hluaccu" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  A_inv = matrix::luinv_eval( *C );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( norm::inv_error_2( A, A_inv ) ) << std::endl;
}

//
// accumulator based LU using DAG
//
template < typename value_t,
           typename approx_t >
void
lu_accu_dag ( const Hpro::TMatrix< value_t > &  A,
              const Hpro::TTruncAcc &           acc,
              const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    std::vector< double >  runtime, flops;

    auto  apx = approx_t();
    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  C   = impl::matrix::copy( A );

    // if ( Hpro::verbose( 3 ) )
    //     lu_dag.print_dot( "lu.dot" );
    
    auto  tstart = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );

        auto  [ lu_dag, accu_map, accu_mtx ] = hlr::dag::gen_dag_lu_accu_lazy( *C, nseq, impl::dag::refine, apx );

        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hluaccudag" );

        impl::dag::run( lu_dag, acc );

        LIKWID_MARKER_STOP( "hluaccudag" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for
        
    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  A_inv = matrix::luinv_eval( *C );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( norm::inv_error_2( A, A_inv ) ) << std::endl;
}

//
// H-LU with lazy updates
//
template < typename value_t,
           typename approx_t >
void
lu_lazy ( const Hpro::TMatrix< value_t > &  A,
          const Hpro::TTruncAcc &           acc,
          const std::string &               apx_name )
{
    std::cout << "    " << term::bullet << term::bold << apx_name << term::reset << std::endl;
    
    approx_t  approx;
    
    std::vector< double >  runtime, flops;

    auto  tic    = timer::now();
    auto  toc    = timer::since( tic );
    auto  C      = impl::matrix::copy( A );
    auto  tstart = timer::now();
        
    for ( int i = 0; i < nbench; ++i )
    {
        impl::matrix::copy_to( A, *C );
            
        blas::reset_flops();

        tic = timer::now();
        
        LIKWID_MARKER_START( "hluaccu" );
            
        impl::lazy::lu< value_t >( *C, acc, approx );

        LIKWID_MARKER_STOP( "hluaccu" );
            
        toc = timer::since( tic );
        std::cout << "      LU in    " << format_time( toc ) << std::endl;

        flops.push_back( get_flops( "lu" ) );
        runtime.push_back( toc.seconds() );

        if ( timer::since( tstart ) > tbench )
            break;
    }// for

    // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

    if ( runtime.size() > 1 )
        std::cout << "      runtime = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;

    auto  A_inv = matrix::luinv_eval( *C );
        
    std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
    std::cout << "      error  = " << format_error( norm::inv_error_2( A, A_inv ) ) << std::endl;
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    LIKWID_MARKER_INIT;

    using value_t = typename problem_t::value_t;

    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< Hpro::TMatrix< value_t > >();

    if ( matrixfile == "" && sparsefile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = gen_ct( *coord );
        auto  bct     = gen_bct( *ct, *ct );
    
        if ( Hpro::verbose( 3 ) )
        {
            io::eps::print( *ct->root(), "ct" );
            io::eps::print( *bct->root(), "ct" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = bem::aca_lrapx( pcoeff );

        A = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    }// if
    else if ( matrixfile != "" )
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = Hpro::read_matrix< value_t >( matrixfile );

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// if
    // else if ( sparsefile != "" )
    // {
    //     std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
    //               << "    sparse matrix = " << sparsefile
    //               << std::endl;

    //     auto  M = Hpro::read_matrix< value_t >( sparsefile );
    //     auto  S = ptrcast( M.get(), Hpro::TSparseMatrix< value_t > );

    //     // convert to H
    //     auto  part_strat    = Hpro::TMongooseAlgPartStrat();
    //     auto  ct_builder    = Hpro::TAlgCTBuilder( & part_strat, ntile );
    //     auto  nd_ct_builder = Hpro::TAlgNDCTBuilder( & ct_builder, ntile );
    //     auto  cl            = nd_ct_builder.build( S );
    //     auto  adm_cond      = Hpro::TWeakAlgAdmCond( S, cl->perm_i2e() );
    //     auto  bct_builder   = Hpro::TBCBuilder();
    //     auto  bcl           = bct_builder.build( cl.get(), cl.get(), & adm_cond );
    //     auto  h_builder     = Hpro::TSparseMatBuilder< value_t >( S, cl->perm_i2e(), cl->perm_e2i() );

    //     if ( Hpro::verbose( 3 ) )
    //     {
    //         io::eps::print( * cl->root(), "ct" );
    //         io::eps::print( * bcl->root(), "bct" );
    //     }// if

    //     h_builder.set_use_zero_mat( true );
        
    //     A = h_builder.build( bcl.get(), acc );
    // }// else

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *A, "A" );
    
    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "LU factorization ( " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;

    //
    // standard recursion with immediate updates
    //

    if ( cmdline::arith == "std" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "standard" << term::reset << std::endl;
    
        // if ( cmdline::approx == "hpro" || cmdline::approx == "all" )
        // {
        //     std::cout << "    " << term::bullet << term::bold << "Hpro" << term::reset << std::endl;

        //     std::vector< double >  runtime, flops;

        //     auto  C = impl::matrix::copy( *A );
        
        //     for ( int i = 0; i < nbench; ++i )
        //     {
        //         impl::matrix::copy_to( *A, *C );
            
        //         blas::reset_flops();

        //         tic = timer::now();
        
        //         LIKWID_MARKER_START( "hlustd" );
            
        //         Hpro::LU::factorise_rec( C.get(), acc );

        //         LIKWID_MARKER_STOP( "hlustd" );
            
        //         toc = timer::since( tic );
        //         std::cout << "      LU in    " << format_time( toc ) << std::endl;

        //         flops.push_back( get_flops( "lu" ) );
        //         runtime.push_back( toc.seconds() );
        //     }// for
        
        //     // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        //     if ( runtime.size() > 1 )
        //         std::cout << "      runtime = "
        //                   << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
        //                   << std::endl;

        //     auto  A_inv = matrix::luinv_eval( *C );
        
        //     std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
        //     std::cout << "      error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
        // }// if
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_std< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_std< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_std< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) lu_std< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_std< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_std< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
    
    //
    // DAG with immediate updates
    //

    if ( cmdline::arith == "dagstd" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "DAG standard" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_std_dag< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_std_dag< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_std_dag< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) lu_std_dag< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_std_dag< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_std_dag< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if

    //
    // using accumulators
    //

    if ( cmdline::arith == "accu" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulator" << term::reset << std::endl;
    
        // if ( cmdline::approx == "hpro" || cmdline::approx == "all" )
        // {
        //     std::cout << "    " << term::bullet << term::bold << "Hpro" << term::reset << std::endl;

        //     std::vector< double >  runtime, flops;
        //     auto                   old_config = Hpro::CFG::Arith::use_accu;

        //     Hpro::CFG::Arith::use_accu = true;
        
        //     auto  C = impl::matrix::copy( *A );
        
        //     for ( int i = 0; i < nbench; ++i )
        //     {
        //         impl::matrix::copy_to( *A, *C );
            
        //         blas::reset_flops();

        //         tic = timer::now();
        
        //         LIKWID_MARKER_START( "hluaccu" );
            
        //         Hpro::LU::factorise_rec( C.get(), acc );

        //         LIKWID_MARKER_STOP( "hluaccu" );
            
        //         toc = timer::since( tic );
        //         std::cout << "      LU in    " << format_time( toc ) << std::endl;

        //         flops.push_back( get_flops( "lu" ) );
        //         runtime.push_back( toc.seconds() );
        //     }// for

        //     Hpro::CFG::Arith::use_accu = old_config;
        
        //     // std::cout     << "      flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        //     if ( runtime.size() > 1 )
        //         std::cout << "      runtime = "
        //                   << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
        //                   << std::endl;

        //     auto  A_inv = matrix::luinv_eval( *C );
        
        //     std::cout << "      mem    = " << format_mem( C->byte_size() ) << std::endl;
        //     std::cout << "      error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
        // }// if
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_accu< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_accu< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_accu< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) lu_accu< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_accu< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_accu< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
    
    if ( cmdline::arith == "dagaccu" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "DAG accumulator" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_accu_dag< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_accu_dag< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_accu_dag< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) lu_accu_dag< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_accu_dag< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_accu_dag< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if

    //
    // lazy evaluation
    //

    if ( cmdline::arith == "lazy" || cmdline::arith == "all" )
    {
        std::cout << "  " << term::bullet << term::bold << "lazy" << term::reset << std::endl;
    
        if ( cmdline::approx == "svd"     || cmdline::approx == "all" ) lu_lazy< value_t, hlr::approx::SVD< value_t > >(     *A, acc, "SVD" );
        if ( cmdline::approx == "rrqr"    || cmdline::approx == "all" ) lu_lazy< value_t, hlr::approx::RRQR< value_t > >(    *A, acc, "RRQR" );
        if ( cmdline::approx == "randsvd" || cmdline::approx == "all" ) lu_lazy< value_t, hlr::approx::RandSVD< value_t > >( *A, acc, "RandSVD" );
        if ( cmdline::approx == "randlr"  || cmdline::approx == "all" ) lu_lazy< value_t, hlr::approx::RandLR< value_t > >(  *A, acc, "RandLR" );
        if ( cmdline::approx == "aca"     || cmdline::approx == "all" ) lu_lazy< value_t, hlr::approx::ACA< value_t > >(     *A, acc, "ACA" );
        if ( cmdline::approx == "lanczos" || cmdline::approx == "all" ) lu_lazy< value_t, hlr::approx::Lanczos< value_t > >( *A, acc, "Lanczos" );
    }// if
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

    if ( ntile == 128 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 455151893464;   // 515345354964;
            if ( gridfile == "sphere-6" ) return 2749530544148;  // 3622694502712;
            if ( gridfile == "sphere-7" ) return 12122134505132; // 21122045509696;
            if ( gridfile == "sphere-8" ) return 118075035109436;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 124087920212;  // 122140965488;
            if ( gridfile == "sphere-6" ) return 881254402164;  // 832636379560;
            if ( gridfile == "sphere-7" ) return 5442869949704; // 5113133279628;
            if ( gridfile == "sphere-8" ) return 30466486574184;
        }// if
    }// if
    else if ( ntile == 64 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 362295459228;  // 362301558484;
            if ( gridfile == "sphere-6" ) return 2254979752712; // 2364851019180;
            if ( gridfile == "sphere-7" ) return 9888495763740; // 10305554560228;
            if ( gridfile == "sphere-8" ) return 119869484219652;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 111349327848; // 111663294708;
            if ( gridfile == "sphere-6" ) return 912967909892; // 936010549040;
            if ( gridfile == "sphere-7" ) return 6025437614656; // 6205509061236;
            if ( gridfile == "sphere-8" ) return 33396933144996;
        }// if
    }// if

    #endif

    return 0;
}
