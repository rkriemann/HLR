//
// Project     : HLR
// Program     : polykern
// Description : computation of kernel of matrix defined by polynomials
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <vector>
#include <random>

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
    return (2.0/3.0) * ipow<5>(t) + (1.0/2.0) * ipow<4>(t) + (2/3);
}

double
f2 ( const double  t )
{
    return (10.0/7.0)*ipow<3>(t) + (7.0/3.0)*ipow<2>(t) + (4/9);
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
    std::uniform_real_distribution< double >   uniform_distr( 0.0, 1.0 );
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
}
