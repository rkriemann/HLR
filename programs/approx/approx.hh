//
// Project     : HLR
// Program     : approx
// Description : testing approximation algorithms for matrix construction
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <fstream>

#include <hlr/utils/likwid.hh>

#include <hlr/arith/norm.hh>

#include <hlr/bem/aca.hh>
#include <hlr/bem/hca.hh>
#include <hlr/bem/dense.hh>

#include <hlr/matrix/print.hh>
#include <hlr/matrix/product.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/info.hh>

#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/approx/lanczos.hh>
#include <hlr/approx/randlr.hh>

#include <hlr/utils/io.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    LIKWID_MARKER_INIT;

    using value_t = typename problem_t::value_t;

    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    auto  runtime = std::vector< double >();

    blas::reset_flops();
    
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< Hpro::TMatrix< value_t > >();

    if ( matrixfile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = gen_ct( *coord );
        auto  bct     = gen_bct( *ct, *ct );
        auto  coeff   = problem->coeff_func();
        auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );

        tic = timer::now();

        if ( cmdline::capprox == "hca" )
        {
            if constexpr ( problem_t::supports_hca )
            {
                std::cout << "    using HCA" << std::endl;
                
                auto  hcagen = problem->hca_gen_func( *ct );
                auto  hca    = bem::hca( pcoeff, *hcagen, cmdline::eps / 100.0, 6 );
                auto  hcalr  = bem::hca_lrapx( hca );
                
                A = impl::matrix::build( bct->root(), pcoeff, hcalr, acc, false, nseq );
            }// if
            else
                cmdline::capprox = "default";
        }// if
        else if (( cmdline::capprox == "aca" ) || ( cmdline::capprox == "default" ))
        {
            std::cout << "    using ACA" << std::endl;

            auto  acalr = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build( bct->root(), pcoeff, acalr, acc, false, nseq );
        }// else
        else if ( cmdline::capprox == "dense" )
        {
            std::cout << "    using dense" << std::endl;

            auto  dense = bem::dense_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );
        
            A = impl::matrix::build( bct->root(), pcoeff, dense, acc, false, nseq );
        }// else
        
        toc = timer::since( tic );
    }// if
    else
    {
        auto  ends_with = [] ( const std::string &  s,
                               const std::string &  end )
        {
            if ( end.size() > s.size() ) return false;
            
            return std::equal( end.rbegin(), end.rend(), s.rbegin() );
        };
        
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        auto  D = blas::matrix< value_t >();
        

        if ( ends_with( matrixfile, ".mat" ) || ends_with( matrixfile, ".m" ) )
            D = io::matlab::read< value_t >( matrixfile );
        else if ( ends_with( matrixfile, ".h5" ) )
            D = io::hdf5::read< blas::matrix< value_t > >( matrixfile );
        else
            HLR_ERROR( "unknown file format" );

        auto  apx = approx::SVD< value_t >();
        
        tic = timer::now();

        A = impl::matrix::compress( indexset( 0, D.nrows()-1 ),
                                    indexset( 0, D.ncols()-1 ),
                                    D,
                                    acc, apx, ntile, false );
                                    
        toc = timer::since( tic );

        
    }// else

    const auto  mem_A    = matrix::data_byte_size( *A );
    const auto  mem_A_d  = matrix::data_byte_size_dense( *A );
    const auto  mem_A_lr = matrix::data_byte_size_lowrank( *A );
    const auto  norm_A   = impl::norm::frobenius( *A );
        
    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A_lr, mem_A_d, mem_A ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|   = " << format_norm( norm_A ) << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // coarsen matrix
    //
    //////////////////////////////////////////////////////////////////////
    
    if ( cmdline::coarsen )
    {
        std::cout << term::bullet << term::bold << "coarsening" << term::reset << std::endl;
        
        auto  apx = approx::SVD< value_t >();

        tic = timer::now();
        
        auto  Ac = impl::matrix::coarsen( *A, acc, apx );
        
        toc = timer::since( tic );

        auto  mem_Ac    = matrix::data_byte_size( *Ac );
        auto  mem_Ac_d  = matrix::data_byte_size_dense( *Ac );
        auto  mem_Ac_lr = matrix::data_byte_size_lowrank( *Ac );
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( mem_Ac_lr, mem_Ac_d, mem_Ac ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Ac) / double(mem_A) ) << std::endl;
        
        if ( verbose( 3 ) )
            matrix::print_eps( *Ac, "Ac", "noid,nosize" );

        auto  diff   = matrix::sum( 1, *A, -1, *Ac );
        auto  norm_A = impl::norm::spectral( *A );
        auto  error  = impl::norm::spectral( *diff );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

        A = std::move( Ac );
    }// if
    
    if ( cmdline::tohodlr )
    {
        std::cout << term::bullet << term::bold << "converting to HODLR" << term::reset << std::endl;
        
        auto  apx = approx::SVD< value_t >();

        tic = timer::now();
        
        auto  Ac = impl::matrix::convert_to_hodlr( *A, acc, apx );
        
        toc = timer::since( tic );

        auto  mem_Ac    = matrix::data_byte_size( *Ac );
        auto  mem_Ac_d  = matrix::data_byte_size_dense( *Ac );
        auto  mem_Ac_lr = matrix::data_byte_size_lowrank( *Ac );
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( mem_Ac_lr, mem_Ac_d, mem_Ac ) << std::endl;
        std::cout << "      vs H  " << boost::format( "%.3f" ) % ( double(mem_Ac) / double(mem_A) ) << std::endl;
        
        if ( verbose( 3 ) )
            matrix::print_eps( *Ac, "Ac", "noid,nosize" );

        auto  diff   = matrix::sum( 1, *A, -1, *Ac );
        auto  norm_A = impl::norm::spectral( *A );
        auto  error  = impl::norm::spectral( *diff );

        std::cout << "    error = " << format_error( error, error / norm_A ) << std::endl;

        A = std::move( Ac );
    }// if
  
    if ( verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );
    
}
