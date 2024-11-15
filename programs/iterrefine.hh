//
// Project     : HLR
// Program     : iterrefine
// Description : testing iterative refinement (linear iteration in 2nd normal form)
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/bem/aca.hh>
#include "hlr/dag/lu.hh"
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/approx/svd.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

template < typename operatorA_t,
           typename operatorW_t >
requires ( is_linear_operator< operatorA_t > &&
           is_linear_operator< operatorW_t > )
void
linear_iteration ( const operatorA_t &  A,
                   const operatorW_t &  W )
{
    std::cout << "  " << term::bullet << term::bold << "solving" << term::reset << std::endl;
    
    auto  x   = A.col_vector();
    auto  sol = A.col_vector();
    auto  e   = A.col_vector();
    auto  b   = A.row_vector();

    // random solution, b=A·sol, x0=0
    sol->fill_rand( 1 );
    sol->scale( 1.0 / sol->norm2() );
    apply( A, *sol, *b );
    x->fill( 0 );
    
    auto  r     = b->copy();
    auto  c     = r->copy();
    auto  error = sol->norm2();
    uint  step  = 0;
    
    while (( error > 1e-8 ) && ( step < 100 ))
    {
        //
        // iteration step:  x_i+1 = x_i - W ( A x_i - b )
        //                        = x_i - (LU)^-1 r
        //

        apply( W, *r, *c );
        add( -1.0, *c, *x );

        // io::matlab::write( *r, Hpro::to_string( "r%03d", step ) );
        // io::matlab::write( *c, Hpro::to_string( "c%03d", step ) );
        // io::matlab::write( *x, Hpro::to_string( "x%03d", step ) );

        sol->copy_to( e.get() );
        add( -1.0, *x, *e );
        error = e->norm2();
            
        std::cout << "  "
                  << boost::format( "%3d" ) % step
                  << "  |r| = "
                  << boost::format( "%.4e" ) % r->norm2()
                  << "  |c| = "
                  << boost::format( "%.4e" ) % c->norm2()
                  << "  |e| = "
                  << boost::format( "%.4e" ) % error
                  << std::endl;
        
        //
        // update residual
        //
        
        apply( A, *x, *r );
        add( -1.0, *b, *r );

        ++step;
    }// while

    std::cout << "  "
              << boost::format( "%3d" ) % step
              << "  |e| = "
              << boost::format( "%.4e" ) % error
              << std::endl;
}

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

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    auto  acc     = gen_accuracy();
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx   = bem::aca_lrapx< hpro::TPermCoeffFn< value_t > >( pcoeff );

    tic = timer::now();
    
    auto  A       = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    
    toc = timer::since( tic );

    auto  mem_A  = A->byte_size();

    std::cout << "    dims  = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( mem_A ) << std::endl;
    std::cout << "      idx = " << format_mem( mem_A / A->nrows() ) << std::endl;
    std::cout << "    |A|   = " << format_norm( impl::norm::frobenius( *A ) ) << std::endl;

    {
        //
        // construct preconditioner vi a H-LU
        //

        std::cout << term::bullet << term::bold << "building preconditioner (FP64)" << term::reset << std::endl;

        auto  LU  = impl::matrix::copy( *A );
        auto  apx = approx::SVD< value_t >();
        auto  dag = hlr::dag::gen_dag_lu( *LU, nseq, impl::dag::refine, apx );

        tic = timer::now();
    
        impl::dag::run( dag, acc );

        toc = timer::since( tic );
    
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( LU->byte_size() ) << std::endl;

        auto  A_inv = matrix::luinv_eval( *LU );

        std::cout << "    error = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;

        linear_iteration( *A, A_inv );
    }

    {
        //
        // construct preconditioner vi a H-LU
        //

        std::cout << term::bullet << term::bold << "building preconditioner (FP32)" << term::reset << std::endl;

        using  value32_t = math::decrease_precision_t< value_t >;
        
        auto  LU  = impl::matrix::convert< value32_t >( *A );
        auto  apx = approx::SVD< value32_t >();
        auto  dag = hlr::dag::gen_dag_lu( *LU, nseq, impl::dag::refine, apx );

        tic = timer::now();
    
        impl::dag::run( dag, acc );

        toc = timer::since( tic );
    
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( LU->byte_size() ) << std::endl;

        {
            auto  LU2   = impl::matrix::convert< value_t >( *LU );
            auto  A_inv = matrix::luinv_eval( *LU2 );

            std::cout << "    error = " << format_error( norm::inv_error_2( impl::arithmetic, *A, A_inv ) ) << std::endl;
        }
    
        auto  A_inv = matrix::luinv_eval( *LU );

        linear_iteration( *A, A_inv );
    }
}
