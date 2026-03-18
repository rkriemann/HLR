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

    // {
    //     //
    //     // benchmark Laplace SLP kernel
    //     //

    //     auto  slp_dbl = Hpro::TLaplaceSLPBF< double >( 
    // }
    
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    auto  runtime = std::vector< double >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );

    if constexpr ( std::is_same_v< problem_t, hlr::apps::laplace_slp > )
    {
        //
        // benchmark Laplace SLP kernel
        //

        auto  grid    = problem->grid();
        auto  fnspace = problem->fnspace();
        auto  bf      = reinterpret_cast< const Hpro::TLaplaceSLPBF< Hpro::TConstFnSpace< double >, Hpro::TConstFnSpace< double >, double > * >( problem->bf() );

        {
            uint  qorder  = 6;
            uint  tris[]  = { 1, 3, 2, 900 };
            uint  ntris   = sizeof(tris) / sizeof(uint);

            for ( uint  j = 0; j < ntris; j++ )
            {
                auto  tri0    = grid->triangle( 1 );
                auto  tri1    = grid->triangle( tris[j] );
                auto  ncommon = bf->reorder_common( tri0.vtx, tri1.vtx );

                std::cout << tris[j] << " / " << ncommon << std::endl;
                
                for ( uint  qorder = 1; qorder <= 10; ++qorder )
                {
                    auto  rule   = bf->quad_rule( ncommon, qorder );
                    auto  npts   = rule->npts + ( 4 - rule->npts % 4 );
                    auto  values = std::vector< double >( npts );
                
                    std::cout << "  "  << boost::format( "%2d" ) % qorder
                              << " / " << boost::format( "%5d" ) % npts
                              << " : ";
                
                    tic = timer::now();
                
                    for ( uint  i = 0; i < 10000; ++i )
                        bf->eval_kernel( 0, 0, tri0, tri1, rule, values, 0 );
                
                    toc = timer::since( tic );
                    std::cout << "    " << format_time( toc );

                    auto  t_dbl = toc.seconds();
                
                    tic = timer::now();
                
                    for ( uint  i = 0; i < 10000; ++i )
                        bf->eval_kernel( 0, 0, tri0, tri1, rule, values, 1e-6 );
                
                    toc = timer::since( tic );
                    std::cout << " , " << format_time( toc );

                    auto  t_sgl = toc.seconds();

                    std::cout << " , " << boost::format( "%.1f" ) % ( t_dbl / t_sgl );

                    std::cout << std::endl;
                }// for
            }// for
        }

        return;
    }// if
    
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

    {
        const auto  mem_d  = matrix::data_byte_size_dense( *MP );
        const auto  mem_lr = matrix::data_byte_size_lowrank( *MP );
        
        std::cout << "    mem   = " << format_mem( mem_lr, mem_d ) << std::endl;
        std::cout << "    |M|   = " << format_norm( impl::norm::frobenius( *MP ) ) << std::endl;
    }
    
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
    

    {
        const auto  mem_d  = matrix::data_byte_size_dense( *REF );
        const auto  mem_lr = matrix::data_byte_size_lowrank( *REF );
        
        std::cout << "    mem   = " << format_mem( mem_lr, mem_d ) << std::endl;
        std::cout << "    |M|   = " << format_norm( norm_ref ) << std::endl;
    }

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
        const auto  mem_d  = matrix::data_byte_size_dense( *STD );
        const auto  mem_lr = matrix::data_byte_size_lowrank( *STD );
        
        std::cout << "    mem   = " << format_mem( mem_lr, mem_d ) << std::endl;
        std::cout << "    |M|   = " << format_norm( impl::norm::frobenius( *STD ) ) << std::endl;
    }

    {
        auto  stderr = impl::norm::frobenius( value_t(1), *REF, value_t(-1), *STD );

        std::cout << "    error STD/REF = " << format_error( stderr, stderr / norm_ref ) << std::endl;
    }

    {
        auto  stderr = impl::norm::frobenius( value_t(1), *STD, value_t(-1), *MP );

        std::cout << "    error MP/STD  = " << format_error( stderr, stderr / norm_ref ) << std::endl;
    }
}
    
