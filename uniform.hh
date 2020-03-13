//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/io/TClusterBasisVis.hh>

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>

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
    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< hpro::TMatrix >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis  bc_vis;
        
        print_ps( ct->root(), "ct" );
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    
    A = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;
        
        blas::vector< value_t >  x( A->ncols() );
        blas::vector< value_t >  y( A->nrows() );

        blas::fill( value_t(1), x );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                impl::mul_vec< value_t >( 1.0, hpro::apply_normal, *A, x, y );

            toc = timer::since( tic );
            runtime.push_back( toc.seconds() );
        
            std::cout << "    mvm in   " << format_time( toc ) << std::endl;

            if ( i < nbench-1 )
                blas::fill( value_t(0), y );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime  = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;
        
        runtime.clear();
    }

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
        std::cout << term::bullet << term::bold << "uniform matrix" << term::reset << std::endl;

        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb ] = impl::matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *A, acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( rowcb->byte_size(), colcb->byte_size() ) << std::endl;


        std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

        tic = timer::now();
    
        auto  A2 = impl::matrix::copy_uniform< value_t >( *A, *rowcb, *colcb );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;

        auto  diff  = hpro::matrix_sum( value_t(1), A.get(), value_t(-1), A2.get() );
        auto  error = hlr::seq::norm::norm_2( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error ) << std::endl;

        if ( hpro::verbose( 3 ) )
            matrix::print_eps( *A2, "A1" );

        //
        // mat-vec benchmark
        //

        if ( true )
        {
            std::cout << "  " << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;
        
            {
                auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

                impl::mul_vec( value_t(1), hpro::apply_normal, *A2, *x_ref, *y );

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
                    impl::mul_vec< value_t >( 1.0, hpro::apply_normal, *A2, *x, *y );

                toc = timer::since( tic );
                runtime.push_back( toc.seconds() );
        
                std::cout << "    mvm in   " << format_time( toc ) << std::endl;

                if ( i < nbench-1 )
                    y->fill( 0 );
            }// for
        
            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;
        
            runtime.clear();
        }

        if ( true )
        {
            std::cout << "  " << term::bullet << term::bold << "mat-vec (uniform)" << term::reset << std::endl;

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
        }
    }

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    if ( true )
    {
        std::cout << term::bullet << term::bold << "H² matrix" << term::reset << std::endl;

        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        hpro::THClusterBasisBuilder< value_t >  bbuilder;

        tic = timer::now();
    
        auto  [ rowcb, colcb ] = bbuilder.build( ct->root(), ct->root(), A.get(), acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        if ( verbose( 3 ) )
        {
            hpro::TPSClusterBasisVis< value_t >  cbvis;

            cbvis.print( rowcb.get(), "rowcb.eps" );
        }// if

        std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

        tic = timer::now();
    
        auto  A2 = to_h2( A.get(), rowcb.get(), colcb.get() );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( hpro::diff_norm_2( A.get(), A2.get() ) ) << std::endl;

        if ( hpro::verbose( 3 ) )
        {
            hpro::TPSMatrixVis  mvis;
        
            mvis.svd( false ).print( A2.get(), "A2" );
        }// if

        //
        // mat-vec benchmark
        //

        std::cout << "  " << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;
        
        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

            A2->mul_vec( 1.0, x_ref.get(), 1.0, y.get(), hpro::apply_normal );

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
                A2->mul_vec( 1.0, x.get(), 1.0, y.get(), hpro::apply_normal );

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
    }
}
