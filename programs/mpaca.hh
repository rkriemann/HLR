//
// Project     : HLR
// Description : construction and MVM with compressed data blocks
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/arith/norm.hh>
#include <hlr/arith/mulvec.hh>
#include <hlr/bem/aca.hh>
#include <hlr/matrix/info.hh>

#include "common.hh"
#include "common-main.hh"

#include <hpro/io/TGridIO.hh>

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    auto  runtime = std::vector< double >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );

    
    //
    // mixedprec matrix
    //
    
    std::cout << term::bullet << term::bold
              << "mixedprec"
              << term::reset << std::endl;
    
    auto  mpacc  = fixed_prec( eps );
    auto  mpaca  = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff, true );
    
    tic = timer::now();
    
    auto  MP = impl::matrix::build( bct->root(), pcoeff, mpaca, mpacc, false );
        
    toc = timer::since( tic );
    std::cout << "    done in " << format_time( toc ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *MP, "mp", "noid" );

    //
    // reference matrix
    //
    
    std::cout << term::bullet << term::bold
              << "reference"
              << term::reset << std::endl;
    
    auto  refacc  = fixed_prec( 1e-12 );
    auto  refaca  = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff, false );
    
    tic = timer::now();
    
    auto  REF = impl::matrix::build( bct->root(), pcoeff, refaca, refacc, false );
        
    toc = timer::since( tic );
    std::cout << "    done in " << format_time( toc ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *REF, "ref", "noid" );

    const auto  norm_ref = impl::norm::frobenius( *REF );
    
    auto  mperr = impl::norm::frobenius( value_t(1), *REF, value_t(-1), *MP );

    std::cout << "    error MP/REF = " << format_error( mperr, mperr / norm_ref ) << std::endl;

    //
    // standard matrix
    //
    
    std::cout << term::bullet << term::bold
              << "standard"
              << term::reset << std::endl;
    
    auto  stdacc  = fixed_prec( eps );
    auto  stdaca  = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff, false );
    
    tic = timer::now();
    
    auto  STD = impl::matrix::build( bct->root(), pcoeff, stdaca, stdacc, false );
        
    toc = timer::since( tic );
    std::cout << "    done in " << format_time( toc ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *STD, "std", "noid" );

    {
        auto  stderr = impl::norm::frobenius( value_t(1), *REF, value_t(-1), *STD );

        std::cout << "    error STD/REF = " << format_error( stderr, stderr / norm_ref ) << std::endl;
    }

    {
        auto  stderr = impl::norm::frobenius( value_t(1), *STD, value_t(-1), *MP );

        std::cout << "    error MP/STD  = " << format_error( stderr, stderr / norm_ref ) << std::endl;
    }
}
    
