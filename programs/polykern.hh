//
// Project     : HLR
// Program     : polykern
// Description : computation of kernel of matrix defined by polynomials
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <boost/rational.hpp>
#include <gmpxx.h>
#include <universal/number/posit/posit.hpp>

using sw::universal::posit;

#include <vector>
#include <random>
#include <limits>

#include <hlr/arith/blas.hh>
#include <hlr/arith/blas_rational.hh>
//#include <hlr/approx/svd.hh>
//#include <hlr/approx/rrqr.hh>
//#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
//#include <hlr/approx/lanczos.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

// using value_t   = double;
// using value_t   = float;
// using value_t   = boost::rational< int64_t >;
using value_t   = mpq_class;
// using value_t   = posit< 64, 3 >;

template < typename T >
double
to_double ( const T  f )
{
    return double(f);
}

template < typename integer_t >
double
to_double ( const boost::rational< integer_t >  f )
{
    return boost::rational_cast< double >( f );
}

template <>
double
to_double ( const mpq_class  f )
{
    return f.get_d();
}

value_t
ipow ( const value_t  t,
       const uint    n )
{
    value_t  r = t;
    
    for ( uint  i = 1; i < n; ++ i )
        r *= t;

    return  r;
}

template < int N >
value_t
ipow ( const value_t  t )
{
    value_t  r = t;
    
    for ( int  i = 1; i < N; ++ i )
        r *= t;

    return  r;
}

template <> value_t ipow<0> ( const value_t    ) { return 1; }
template <> value_t ipow<1> ( const value_t  t ) { return t; }
template <> value_t ipow<2> ( const value_t  t ) { return t*t; }
template <> value_t ipow<3> ( const value_t  t ) { return t*t*t; }
template <> value_t ipow<4> ( const value_t  t ) { return t*t*t*t; }
template <> value_t ipow<5> ( const value_t  t ) { return t*t*t*t*t; }
template <> value_t ipow<6> ( const value_t  t ) { return t*t*t*t*t*t; }

//
// test polynomials
//
value_t
f1 ( const value_t  t )
{
    return ( (value_t(2)/value_t(3)) * ipow<5>(t) +
             (value_t(1)/value_t(2)) * ipow<4>(t) +
             (value_t(5)/value_t(2)) * ipow<3>(t) +
             (value_t(9)/value_t(1)) * ipow<2>(t) +
             (value_t(1)/value_t(2)) * ipow<1>(t) +
             (value_t(2)/value_t(3)) );
}

value_t
f2 ( const value_t  t )
{
    return ( (value_t(10)/value_t(7))*ipow<3>(t) +
             (value_t( 7)/value_t(3))*ipow<2>(t) +
             (value_t(10)/value_t(3))*ipow<1>(t) +
             (value_t( 4)/value_t(9)) );
}

//
// compute low-rank approximation
//
template < typename approx_t >
void
lrapx ( const blas::matrix< value_t > &  M,
        const std::string &              apx_name )
{
    std::cout << term::bullet << term::bold << apx_name << term::reset << std::endl;
        
    // auto  tol = value_t(10) * std::numeric_limits< value_t >::epsilon();
    auto  acc = hpro::fixed_prec( 1e-6 );
    
    auto  normM    = blas::sqnorm_F( M );
    auto  M2       = blas::copy( M );
    auto  apx      = approx_t();
    auto  [ U, V ] = apx( M2, acc );
    //auto  S        = blas::sv( U, V );

    blas::copy( M, M2 ); // was destroyed
    
    blas::prod( value_t(-1), U, blas::adjoint(V), value_t(1), M2 );

    std::cout << "    rank  = " << U.ncols() << std::endl
              << "    error = " << format_error( math::sqrt( to_double( value_t( blas::sqnorm_F( M2 ) / normM ) ) ) ) << std::endl;
    // << "    σ_i   = ";
    
    // for ( uint  i = 0; i < S.length(); ++i )
    //     std::cout << S(i) << ", ";
    // std::cout << std::endl;
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
    blas::matrix< value_t >  M( nsamples, nindices );

    const auto                                 seed = 1593694284; // time( nullptr );
    std::default_random_engine                 generator( seed );
    #if 0
    std::uniform_int_distribution< int64_t >   uniform_distr( 1, 1000000 );
    auto                                       random      = [&] ()
                                                             {
                                                                 return value_t( uniform_distr( generator ), 1000000 );
                                                             };
    #else
    std::uniform_real_distribution< double >   uniform_distr( -1.0, 1.0 );
    auto                                        random      = [&] () { return uniform_distr( generator ); };
    #endif

    for ( uint  i = 0; i < nsamples; ++i )
    {
        // const auto  v = value_t( uniform_distr( generator ),
        //                          uniform_distr( generator ) );
        // const auto     t = v / abs( v );
        const value_t  t  = random();
        const auto     v1 = f1( t );
        const auto     v2 = f2( t );

        for ( uint  j = 0; j < nindices; ++j )
        {
            const auto  [ ix, iy ] = indices[j];
            const auto  v          = ipow( v1, ix ) * ipow( v2, iy );

            M(i,j) = v;
        }// for
    }// for

    // hpro::DBG::write( M, "M.mat", "M" );
    
    //
    // approximate M
    //

    std::cout << "full rank = " << std::min( M.nrows(), M.ncols() ) << std::endl;

    // lrapx< approx::SVD< value_t > >(     M, "SVD" );
    // lrapx< approx::RRQR< value_t > >(    M, "RRQR" );
    // lrapx< approx::RandSVD< value_t > >( M, "RandSVD" );
    lrapx< approx::ACA< value_t > >(     M, "ACA" );
    // lrapx< approx::Lanczos< value_t > >( M, "Lanczos" );
}
