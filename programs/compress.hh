//
// Project     : HLR
// Program     : combustion
// Description : compression of datasets from combustion simulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/utils/io.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/arith/norm.hh>
#include <hlr/utils/tensor.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

using indexset = hpro::TIndexSet;

struct local_accuracy : public hpro::TTruncAcc
{
    local_accuracy ( const double  abs_eps )
            : hpro::TTruncAcc( 0.0, abs_eps )
    {}
    
    virtual const TTruncAcc  acc ( const indexset &  rowis,
                                   const indexset &  colis ) const
    {
        return hpro::absolute_prec( abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) );
    }
};

template < typename value_t >
blas::matrix< value_t >
gen_matrix_log ( const size_t  n )
{
    double  h = 2 * math::pi< value_t >() / value_t(n+1);
    auto    M = blas::matrix< value_t >( n, n );

    for ( uint  i = 0; i < n; ++i )
    {
        const double  x1[2] = { sin(i*h), cos(i*h) };
        
        for ( uint  j = 0; j < n; ++j )
        {
            const double  x2[2] = { sin(j*h), cos(j*h) };
            const double  dist2 = math::square( x1[0] - x2[0] ) + math::square( x1[1] - x2[1] );

            if ( dist2 < 1e-12 )
                M(i,j) = 0;
            else
                M(i,j) = math::log( math::sqrt(dist2) );
        }// for
    }// for

    return M;
}

template < typename value_t >
blas::matrix< value_t >
gen_matrix_exp ( const size_t  n )
{
    double  h = 4 * math::pi< value_t >() / value_t(n+1);
    auto    M = blas::matrix< value_t >( n, n );

    for ( uint  i = 0; i < n; ++i )
    {
        const double  x = i*h - 2.0 * math::pi< value_t >();
        
        for ( uint  j = 0; j < n; ++j )
        {
            const double  y = j*h - 2.0 * math::pi< value_t >();
            
            // M(i,j) = std::real( math::exp( 3.0 * std::complex< value_t >( 0, 1 ) * math::sqrt( math::abs( x * x - y * y ) ) ) );
            M(i,j) = std::real( math::exp( 3.0 * std::complex< value_t >( 0, 1 ) * math::sqrt( math::abs( x*x - 4*y ) ) ) );
        }// for
    }// for

    return M;
}

template < typename approx_t,
           typename value_t >
void
do_compress ( blas::matrix< value_t > &  D,
              const double               delta,
              const double               norm_D,
              const size_t               mem_D )
{
    auto    T     = blas::copy( D );
    auto    acc   = local_accuracy( delta );
    auto    apx   = approx_t();
    size_t  csize = 0;
    auto    zconf = ( cmdline::zfp == 0 ? std::unique_ptr< zfp_config >() : std::make_unique< zfp_config >() );

    if ( cmdline::zfp > 0 )
    {
        if ( cmdline::zfp > 1 ) *zconf = zfp_config_rate( int( cmdline::zfp ), false );
        else                    *zconf = zfp_config_accuracy( cmdline::zfp );
    }// if
            
    // auto  zconf  = zfp_config_reversible();
    // auto  zconf  = zfp_config_rate( rate, false );
    // auto  zconf  = zfp_config_precision( rate );
    // auto  zconf  = zfp_config_accuracy( rate );
    auto    tic   = timer::now();

    impl::matrix::compress_replace( indexset( 0, D.nrows()-1 ),
                                    indexset( 0, D.ncols()-1 ),
                                    T, csize,
                                    acc, apx,
                                    cmdline::ntile,
                                    zconf.get() );
            
    auto    toc   = timer::since( tic );
        
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( csize ) << std::endl;
    std::cout << "     %full = " << format( "%.2f %%" ) % ( 100.0 * ( double( csize ) / double( mem_D ) )) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::matlab::write( T, "T" );
        
    blas::add( value_t(-1), D, T );

    const auto  error = blas::norm_F( T );
        
    std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;
}

template < typename approx_t,
           typename value_t >
void
do_H ( blas::matrix< value_t > &  D,
       const double               delta,
       const double               norm_D,
       const size_t               mem_D )
{
    auto  acc   = local_accuracy( delta );
    auto  apx   = approx_t();
    auto  zconf = ( cmdline::zfp == 0 ? std::unique_ptr< zfp_config >() : std::make_unique< zfp_config >() );

    if ( cmdline::zfp > 0 )
    {
        if ( cmdline::zfp > 1 ) *zconf = zfp_config_rate( int( cmdline::zfp ), false );
        else                    *zconf = zfp_config_accuracy( cmdline::zfp );
    }// if

    auto  tic   = timer::now();
    auto  A     = impl::matrix::compress( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, apx, cmdline::ntile, zconf.get() );
    auto  toc   = timer::since( tic );
        
    std::cout << "    done in  " << format_time( toc ) << std::endl;
        
    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid,nosize" );
        
    auto  DM      = hpro::TDenseMatrix( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D );
    auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), DM );
    auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );
    auto  mem_A   = A->byte_size();
        
    std::cout << "    mem    = " << format_mem( mem_A ) << std::endl;
    std::cout << "     %full = " << format( "%.2f %%" ) % ( 100.0 * ( double( mem_A ) / double( mem_D ) )) << std::endl;
    std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    //
    // read dataset
    //
    
    auto  D = blas::matrix< value_t >();

    if ( cmdline::matrixfile != "" )
    {
        std::cout << "  " << term::bullet << term::bold << "reading data (" << cmdline::matrixfile << ")" << term::reset << std::endl;
        
        D = io::matlab::read< value_t >( cmdline::matrixfile );
    }// if
    else
    {
        std::cout << "  " << term::bullet << term::bold << "generating matrix (" << cmdline::appl << ")" << term::reset << std::endl;

        if      ( cmdline::appl == "log" ) D = std::move( gen_matrix_log< value_t >( cmdline::n ) );
        else if ( cmdline::appl == "exp" ) D = std::move( gen_matrix_exp< value_t >( cmdline::n ) );
        else
            HLR_ERROR( "unknown matrix : " + cmdline::appl );
        
        if ( hpro::verbose( 3 ) )
            io::matlab::write( D, "D" );
    }// else
        
    auto  mem_D  = D.byte_size();
    auto  norm_D = blas::norm_F( D );

    std::cout << "    size   = " << D.nrows() << " × " << D.ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( mem_D ) << std::endl;
    std::cout << "    |D|    = " << format_norm( norm_D ) << std::endl;

    //
    // compress data and replace content
    //

    auto  delta = norm_D * hlr::cmdline::eps / D.nrows();
    
    std::cout << "  " << term::bullet << term::bold << "compression, "
              << cmdline::approx << " ε = " << delta << ", "
              << "zfp = " << cmdline::zfp
              << term::reset << std::endl;

    if ( true )
    {
        //
        // compress inplace (replace data)
        //
        
        if ( cmdline::approx == "svd" ||
             cmdline::approx == "default" ) do_compress< hlr::approx::SVD< value_t > >(     D, delta, norm_D, mem_D );
        if ( cmdline::approx == "rrqr"    ) do_compress< hlr::approx::RRQR< value_t > >(    D, delta, norm_D, mem_D );
        if ( cmdline::approx == "randsvd" ) do_compress< hlr::approx::RandSVD< value_t > >( D, delta, norm_D, mem_D );
        if ( cmdline::approx == "aca"     ) do_compress< hlr::approx::ACA< value_t > >(     D, delta, norm_D, mem_D );
    }// if
    else
    {
        //
        // compress to H-matrix
        //

        if ( cmdline::approx == "svd" ||
             cmdline::approx == "default" ) do_H< hlr::approx::SVD< value_t > >(     D, delta, norm_D, mem_D );
        if ( cmdline::approx == "rrqr"    ) do_H< hlr::approx::RRQR< value_t > >(    D, delta, norm_D, mem_D );
        if ( cmdline::approx == "randsvd" ) do_H< hlr::approx::RandSVD< value_t > >( D, delta, norm_D, mem_D );
        if ( cmdline::approx == "aca"     ) do_H< hlr::approx::ACA< value_t > >(     D, delta, norm_D, mem_D );
    }// else
}
    
