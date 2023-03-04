//
// Project     : HLR
// Program     : polykern
// Description : computation of kernel of matrix defined by polynomials
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <vector>
#include <random>
#include <limits>

#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/approx/lanczos.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

double
ipow ( const double  t,
       const uint    n )
{
    double  r = t;
    
    for ( uint  i = 1; i < n; ++ i )
        r *= t;

    return  r;
}

template < int N >
 double
 ipow ( const double  t )
{
    double  r = t;
    
    for ( int  i = 1; i < N; ++ i )
        r *= t;

    return  r;
}

template <> double ipow<0> ( const double    ) { return 1; }
template <> double ipow<1> ( const double  t ) { return t; }
template <> double ipow<2> ( const double  t ) { return t*t; }
template <> double ipow<3> ( const double  t ) { return t*t*t; }
template <> double ipow<4> ( const double  t ) { return t*t*t*t; }
template <> double ipow<5> ( const double  t ) { return t*t*t*t*t; }
template <> double ipow<6> ( const double  t ) { return t*t*t*t*t*t; }

//
// test polynomials
//
double
f1 ( const double  t )
{
    return ( (2.0/3.0) * ipow<5>(t) +
             (1.0/2.0) * ipow<4>(t) +
             (5.0/2.0) * ipow<3>(t) +
             (9.0/1.0) * ipow<2>(t) +
             (1.0/2.0) * ipow<1>(t) +
             (2.0/3.0) );
}

double
f2 ( const double  t )
{
    return ( (10.0/7.0)*ipow<3>(t) +
             ( 7.0/3.0)*ipow<2>(t) +
             (10.0/3.0)*ipow<1>(t) +
             ( 4.0/9.0) );
}

//
// compute low-rank approximation
//
template < typename approx_t >
void
lrapx ( const blas::matrix< double > &  M,
        const std::string &             apx_name )
{
    std::cout << term::bullet << term::bold << apx_name << term::reset << std::endl;
        
    auto  tol = double(10) * std::numeric_limits< double >::epsilon();
    auto  acc = hpro::fixed_prec( tol );
    
    auto  M2       = blas::copy( M );
    auto  M3       = blas::copy( M );
    auto  normM    = blas::norm_F( M );
    auto  apx      = approx_t();
    auto  [ U, V ] = apx( M2, acc );
    auto  S        = blas::sv( U, V );
    
    blas::prod( double(-1), U, blas::adjoint(V), double(1), M3 );

    std::cout << "    rank  = " << U.ncols() << std::endl
              << "    error = " << format_error( blas::norm_F( M3 ) / normM ) << std::endl
              << "    σ_i   = ";
    
    for ( uint  i = 0; i < S.length(); ++i )
        std::cout << S(i) << ", ";
    std::cout << std::endl;
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    //
    // set of monomials for combination of polynomials above
    // (x: f1, y: f2)
    //

    // {1, x, x^2, x^3, y, x*y, x^2*y, x^3*y, y^2, x*y^2, x^2*y^2, x^3*y^2, y^3, x*y^3, x^2*y^3, y^4, x*y^4, y^5}
    std::vector< std::pair< int, int > >  indices = { { 0, 0 }, // 1
                                                      { 1, 0 }, // x
                                                      { 2, 0 }, // x^2
                                                      { 3, 0 }, // x^3
                                                      { 0, 1 }, // y
                                                      { 1, 1 }, // x·y
                                                      { 2, 1 }, // x^2·y
                                                      { 3, 1 }, // x^3·y
                                                      { 0, 2 }, // y^2
                                                      { 1, 2 }, // x·y^2
                                                      { 2, 2 }, // x^2·y^2
                                                      { 3, 2 }, // x^3·y^2
                                                      { 0, 3 }, // y^3
                                                      { 1, 3 }, // x·y^3
                                                      { 2, 3 }, // x^2·y^3
                                                      { 0, 4 }, // y^4
                                                      { 0, 5 }  // y^5
                                                    };

    //
    // construct sample matrix
    //
    
    const size_t            nindices = indices.size();
    const size_t            nsamples = 50; // number of random sample points
    blas::matrix< double >  M( nsamples, nindices );

    const auto                                 seed = 1593694284; // time( nullptr );
    std::default_random_engine                 generator( seed );
    std::uniform_real_distribution< double >   uniform_distr( -1.0, 1.0 );
    auto                                       random      = [&] () { return uniform_distr( generator ); };
    
    for ( uint  i = 0; i < nsamples; ++i )
    {
        const double  t  = random();
        const auto    v1 = f1( t );
        const auto    v2 = f2( t );

        for ( uint  j = 0; j < nindices; ++j )
        {
            const auto  [ ix, iy ] = indices[j];
            const auto  v          = ipow( v1, ix ) * ipow( v2, iy );

            M(i,j) = v;
        }// for
    }// for

    hpro::DBG::write( M, "M.mat", "M" );
    
    //
    // approximate M
    //

    std::cout << "full rank = " << std::min( M.nrows(), M.ncols() ) << std::endl;

    lrapx< approx::SVD< double > >(     M, "SVD" );
    lrapx< approx::RRQR< double > >(    M, "RRQR" );
    lrapx< approx::RandSVD< double > >( M, "RandSVD" );
    lrapx< approx::ACA< double > >(     M, "ACA" );
    lrapx< approx::Lanczos< double > >( M, "Lanczos" );
}
