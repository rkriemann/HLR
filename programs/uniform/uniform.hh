//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#define USE_H2

#if defined( USE_H2 )
#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/algebra/mat_conv.hh>
#endif

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/bem/aca.hh>
#include <hlr/approx/randsvd.hh>

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
    using value_t = typename problem_t::value_t;

    auto  runtime = std::vector< double >();
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *ct->root(), "ct" );
        io::eps::print( *bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = bem::aca_lrapx( pcoeff );

    auto  A = std::unique_ptr< hpro::TMatrix >();
    
    tic = timer::now();

    if ( cmdline::matrixfile != "" )
    {
        A = io::hpro::read( cmdline::matrixfile );
    }// if
    else
    {
        A = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );

        // io::hpro::write( *A, "A.hm" );
    }// else
    
    toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *A, "A", "noinnerid" );
        io::eps::print_lvl( *A, "L" );
    }// if

    const auto  normA = hlr::norm::spectral( *A, true, 1e-4 );

    //////////////////////////////////////////////////////////////////////
    //
    // directly build uniform matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto  apx = approx::SVD< value_t >();

    {
        std::cout << term::bullet << term::bold << "uniform H-matrix (lvl)" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_lvl( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;
        
        {
            auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
            auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }
    }

    {
        std::cout << term::bullet << term::bold << "uniform H-matrix (rec)" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb3, colcb3, A3 ] = impl::matrix::build_uniform_rec( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A3->byte_size(), rowcb3->byte_size(), colcb3->byte_size() ) << std::endl;
        
        {
            auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A3 );
            auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }

        if ( true )
        {
            std::cout << "    " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

            using single_t = math::decrease_precision_t< value_t >;

            auto  rowcb4 = matrix::copy< single_t >( *rowcb3 );
            auto  colcb4 = matrix::copy< single_t >( *colcb3 );
            
            auto  rowcb5 = matrix::copy< value_t >( *rowcb4 );
            auto  colcb5 = matrix::copy< value_t >( *colcb4 );

            std::cout << "      mem    = " << format_mem( A3->byte_size(), rowcb4->byte_size(), colcb4->byte_size() ) << std::endl;
            
            matrix::replace_cluster_basis( *A3, *rowcb5, *colcb5 );
            
            {
                auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A3 );
                auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                std::cout << "      error  = " << format_error( error / normA ) << std::endl;
            }
        }

        #if defined(HAS_HALF)
        if ( true )
        {
            std::cout << "    " << term::bullet << term::bold << "half precision" << term::reset << std::endl;

            using single_t = math::decrease_precision_t< value_t >;
            using half_t   = math::decrease_precision_t< single_t >;
            
            auto  rowcb4 = matrix::copy< half_t >( *rowcb3 );
            auto  colcb4 = matrix::copy< half_t >( *colcb3 );
            
            auto  rowcb5 = matrix::copy< single_t >( *rowcb4 );
            auto  colcb5 = matrix::copy< single_t >( *colcb4 );

            auto  rowcb6 = matrix::copy< value_t >( *rowcb5 );
            auto  colcb6 = matrix::copy< value_t >( *colcb5 );

            std::cout << "      mem    = " << format_mem( A3->byte_size(), rowcb4->byte_size(), colcb4->byte_size() ) << std::endl;
            
            matrix::replace_cluster_basis( *A3, *rowcb6, *colcb6 );
            
            {
                auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A3 );
                auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                std::cout << "      error  = " << format_error( error / normA ) << std::endl;
            }
        }
        #endif
    }

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb2, colcb2 ] = impl::matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *A, acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;


        std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

        tic = timer::now();
    
        auto  A3 = impl::matrix::copy_uniform< value_t >( *A, *rowcb2, *colcb2 );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A3->byte_size() ) << std::endl;

        auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A3 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;

        if ( hpro::verbose( 3 ) )
            io::eps::print( *A3, "A3", "noid" );
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    #if defined( USE_H2 )
    
    auto  rowcb3 = std::unique_ptr< hpro::TClusterBasis< value_t > >();
    auto  colcb3 = std::unique_ptr< hpro::TClusterBasis< value_t > >();
    auto  A3     = std::unique_ptr< hpro::TMatrix >();
    
    if ( true )
    {
        std::cout << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;

        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        hpro::THClusterBasisBuilder< value_t >  bbuilder;

        tic = timer::now();
    
        std::tie( rowcb3, colcb3 ) = bbuilder.build( ct->root(), ct->root(), A.get(), acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;

        if ( verbose( 3 ) )
            io::eps::print( *rowcb3, "rowcb" );

        std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

        tic = timer::now();
    
        A3 = std::move( to_h2( A.get(), rowcb3.get(), colcb3.get() ) );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A3->byte_size(), rowcb3->byte_size(), colcb3->byte_size() ) << std::endl;

        auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A3 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        
        if ( hpro::verbose( 3 ) )
            io::eps::print( *A3, "noinnerid" );
    }// if

    #endif
    
    #if 0
    
    //////////////////////////////////////////////////////////////////////
    //
    // H-matrix matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H-matrices" << term::reset << std::endl;
        
        blas::vector< value_t >  x( A->ncols() );
        blas::vector< value_t >  y( A->nrows() );

        blas::fill( x, value_t(1) );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                impl::mul_vec< value_t >( 1.0, hpro::apply_normal, *A, x, y );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
        
            std::cout << "    mvm in   " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                blas::fill( y, value_t(0) );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        runtime.clear();
    }// if

    //
    // set up reference vector for mat-vec error tests

    auto  x_ref = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
    auto  y_ref = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

    x_ref->fill( 1 );
    impl::mul_vec< value_t >( 1.0, hpro::apply_normal, *A, *x_ref, *y_ref );
    
    //////////////////////////////////////////////////////////////////////
    //
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "uniform H-matrix" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A2->row_is() );

            impl::uniform::mul_vec( value_t(1), hpro::apply_adjoint, *A2, *x_ref, *y, *rowcb, *colcb );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() ) << std::endl;
        }
            
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A2->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A2->row_is() );

        x->fill( 1 );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            for ( int j = 0; j < 50; ++j )
                impl::uniform::mul_vec( value_t(1), hpro::apply_normal, *A2, *x, *y, *rowcb, *colcb );
            
            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
            
            std::cout << "    mvm in   " << format_time( toc ) << std::endl;
            
            if ( i < nbench-1 )
                y->fill( 1 );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        runtime.clear();
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    #if defined( USE_H2 )
    
    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;
        
        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );
            
            A3->mul_vec( 1.0, x_ref.get(), 1.0, y.get(), hpro::apply_normal );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() ) << std::endl;
        }
            
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A3->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A3->row_is() );

        x->fill( 1 );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                A3->mul_vec( 1.0, x.get(), 1.0, y.get(), hpro::apply_normal );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
        
            std::cout << "    mvm in   " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                y->fill( 1 );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        runtime.clear();
    }// if

    #endif
    #endif
}
