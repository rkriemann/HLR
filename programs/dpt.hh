//
// Project     : HLR
// Program     : dpt
// Description : testing DPT eigenvalue algorithmus
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <hlr/arith/blas_eigen.hh>

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
    using value_t = double;

    std::cout << term::bullet << term::bold << "dense DPT eigen iteration ( " << impl_name
              << " )" << term::reset << std::endl;

    blas::eigen_stat  stat;
        
    for ( size_t  n = 128; n <= 512; n += 128 )
    {
        std::mutex  mtx;
        uint        nsweeps_min = 0;
        uint        nsweeps_jac = 0;

        ::tbb::parallel_for( uint(0), uint(10),
                             [&,n] ( const uint )
                             // for ( uint  i = 0; i < 10; ++i )
                             {
                                 auto  R  = blas::random< value_t >( n, n );
                                 auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );
                                 auto  Mc = blas::copy( M );

                                 for ( uint nsweeps = 1; nsweeps < n; ++nsweeps )
                                 {
                                     auto  Ms         = blas::copy< float >( M );
                                     auto  [ Ej, Vj ] = blas::eigen_jac( Ms, nsweeps, 1e-7 );
                                         
                                     auto  Wj         = blas::copy< double >( Vj );
                                     auto  VM         = blas::prod( value_t(1), blas::adjoint( Wj ), M );
                                     auto  VMV        = blas::prod( value_t(1), VM, Wj );
                                     auto  [ Ed, Vd ] = blas::eigen_dpt( VMV, 0, 1e-8, "fro", 0, & stat );
                                         
                                     if ( stat.converged )
                                     {
                                         // converged
                                         std::scoped_lock  lock( mtx );
                                             
                                         nsweeps_min = std::max( nsweeps_min, nsweeps+1 );
                                         break;
                                     }// if
                                 }// for

                                 auto  [ E, V ] = blas::eigen_jac( Mc, 100, 1e-14, & stat );

                                 {
                                     std::scoped_lock  lock( mtx );
                                         
                                     nsweeps_jac = std::max( nsweeps_jac, stat.nsweeps );
                                 }
                             } );

        std::cout << "n = " << n << "   " << nsweeps_min << "    " << nsweeps_jac << std::endl;
    }// for
}
