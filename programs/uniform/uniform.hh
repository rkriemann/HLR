//
// Project     : HLR
// Module      : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/config.h>

#if defined(HPRO_USE_LIC_CHECK)
#define HLR_HAS_H2
#endif

#if defined(HLR_HAS_H2)
#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/matrix/convert.hh>
#endif

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/info.hh>
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

    if ( false )
    {
        auto  A = io::h2lib::read< value_t >( "V.cdf" );

        std::cout << "    dims   = " << term::bold << A->nrows() << " × " << A->ncols() << term::reset << std::endl;
        std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
        std::cout << "    |A|    = " << format_norm( norm::frobenius( *A ) ) << std::endl;
        
        auto  acc = gen_accuracy();
        auto  apx = approx::SVD< value_t >();
        
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_rec( *A, apx, acc, nseq );

        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        auto  [ row_min, row_avg, row_max ] = matrix::rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = matrix::rank_info( *colcb );

        std::cout << "    ranks  = "
                  << row_min << " … " << row_avg << " … " << row_max << " / "
                  << col_min << " … " << col_avg << " … " << col_max << std::endl;

        {
            const auto  normA = hlr::norm::spectral( impl::arithmetic, *A, 1e-4 );
            auto        diff  = matrix::sum( 1, *A, -1, *A2 );
            auto        error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }
    }
    
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
        io::vtk::print( *coord, "coord" );
        io::eps::print( *ct->root(), "ct" );
        io::eps::print( *bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = bem::aca_lrapx( pcoeff );

    auto  A = std::unique_ptr< hpro::TMatrix< value_t > >();
    
    tic = timer::now();

    if ( cmdline::matrixfile != "" )
    {
        A = io::hpro::read< value_t >( cmdline::matrixfile );
    }// if
    else
    {
        A = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );

        // io::hpro::write( *A, "A.hm" );
    }// else
    
    toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << term::bold << A->nrows() << " × " << A->ncols() << term::reset << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *A, "A", "noid" );
        io::eps::print_lvl( *A, "L" );
    }// if

    const auto  normA = hlr::norm::spectral( impl::arithmetic, *A, 1e-4 );

    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A ) ) << std::endl;
    
    {
        auto  [ kmin, kavg, kmax ] = matrix::rank_info( *A );
    
        std::cout << "    ranks  = " << kmin << " … " << kavg << " … " << kmax << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    //
    // directly build uniform matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto  rowcb_uni = std::unique_ptr< matrix::shared_cluster_basis< value_t > >();
    auto  colcb_uni = std::unique_ptr< matrix::shared_cluster_basis< value_t > >();
    auto  A_uni     = std::unique_ptr< hpro::TMatrix< value_t > >();
    auto  apx       = approx::SVD< value_t >();

    if ( false )
    {
        std::cout << term::bullet << term::bold << "uniform H-matrix (lvl)" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_lvl( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;
        
        auto  [ row_min, row_avg, row_max ] = matrix::rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = matrix::rank_info( *colcb );

        std::cout << "    ranks  = "
                  << row_min << " … " << row_avg << " … " << row_max << " / "
                  << col_min << " … " << col_avg << " … " << col_max << std::endl;
        
        {
            auto  diff  = matrix::sum( 1, *A, -1, *A2 );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }
    }

    if ( true )
    {
        std::cout << term::bullet << term::bold << "uniform H-matrix (rec)" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_rec( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        auto  [ row_min, row_avg, row_max ] = matrix::rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = matrix::rank_info( *colcb );

        std::cout << "    ranks  = "
                  << row_min << " … " << row_avg << " … " << row_max << " / "
                  << col_min << " … " << col_avg << " … " << col_max << std::endl;
        
        if ( hpro::verbose( 3 ) )
        {
            io::eps::print( *A2, "A2", "noid" );
            io::eps::print( *rowcb, "rowcb2" );
            io::eps::print( *colcb, "colcb2" );
        }// if
        
        {
            auto  diff  = matrix::sum( 1, *A, -1, *A2 );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }
        
        {
            auto  error = impl::norm::frobenius( 1, *A, -1, *A2 );

            std::cout << "    error = " << format_error( error, error / normA ) << std::endl;
        }

        //
        // preserve for MVM
        //

        A_uni     = std::move( A2 );
        rowcb_uni = std::move( rowcb );
        colcb_uni = std::move( colcb );
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
    
        auto  A2 = impl::matrix::copy_uniform< value_t >( *A, *rowcb2, *colcb2 );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;

        auto  diff  = matrix::sum( 1, *A, -1, *A2 );
        auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    auto  rowcb_h2 = std::unique_ptr< matrix::nested_cluster_basis< value_t > >();
    auto  colcb_h2 = std::unique_ptr< matrix::nested_cluster_basis< value_t > >();
    auto  A_h2     = std::unique_ptr< hpro::TMatrix< value_t > >();

    if ( true )
    {
        std::cout << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_h2_rec( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        auto  [ row_min, row_avg, row_max ] = matrix::rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = matrix::rank_info( *colcb );

        std::cout << "    ranks  = "
                  << row_min << " … " << row_avg << " … " << row_max << " / "
                  << col_min << " … " << col_avg << " … " << col_max << std::endl;
        
        if ( hpro::verbose( 3 ) )
        {
            io::eps::print( *A2, "H2", "noid" );
            io::eps::print( *rowcb, "rowcb_h2" );
            io::eps::print( *colcb, "colcb_h2" );
        }// if
        
        {
            auto  diff  = matrix::sum( 1, *A, -1, *A2 );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }
        
        {
            auto  error = impl::norm::frobenius( 1, *A, -1, *A2 );

            std::cout << "    error = " << format_error( error, error / normA ) << std::endl;
        }

        //
        // preserve for MVM
        //

        A_h2     = std::move( A2 );
        rowcb_h2 = std::move( rowcb );
        colcb_h2 = std::move( colcb );
    }

    #if defined(HLR_HAS_H2)
    
    auto  rowcb_h21 = std::unique_ptr< hpro::TClusterBasis< value_t > >();
    auto  colcb_h21 = std::unique_ptr< hpro::TClusterBasis< value_t > >();
    auto  A_h21     = std::unique_ptr< hpro::TMatrix< value_t > >();
    
    if ( true )
    {
        std::cout << term::bullet << term::bold << "H²-matrix (Hpro)" << term::reset << std::endl;

        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        hpro::THClusterBasisBuilder< value_t >  bbuilder;

        tic = timer::now();
    
        auto [ rowcb, colcb ] = bbuilder.build( ct->root(), ct->root(), A.get(), acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;

        auto  [ row_min, row_avg, row_max ] = matrix::rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = matrix::rank_info( *colcb );

        std::cout << "    ranks  = "
                  << row_min << " … " << row_avg << " … " << row_max << " / "
                  << col_min << " … " << col_avg << " … " << col_max << std::endl;
        
        if ( verbose( 3 ) )
        {
            io::eps::print( *rowcb, "rowcb_h2_2" );
            io::eps::print( *colcb, "colcb_h2_2" );
        }// if

        std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

        tic = timer::now();
    
        auto  A2 = to_h2( A.get(), rowcb.get(), colcb.get() );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        auto  diff  = matrix::sum( 1, *A, -1, *A2 );
        auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        
        if ( hpro::verbose( 3 ) )
            io::eps::print( *A2, "H2_2", "noid" );

        //
        // preserve for MVM
        //

        A_h21     = std::move( A2 );
        rowcb_h21 = std::move( rowcb );
        colcb_h21 = std::move( colcb );
    }// if

    #endif
    
    #if 1
    
    //////////////////////////////////////////////////////////////////////
    //
    // H-matrix matrix vector multiplication
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "mat-vec" << term::reset << std::endl;

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H-matrices" << term::reset << std::endl;
        
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

        x->fill( 1 );

        // blas::vector< value_t >  x( A->ncols() );
        // blas::vector< value_t >  y( A->nrows() );

        // blas::fill( x, value_t(1) );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *A, *x, *y );

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

    //
    // set up reference vector for mat-vec error tests

    auto  x_ref = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
    auto  y_ref = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

    x_ref->fill( 1 );
    impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *A, *x_ref, *y_ref );
    
    //////////////////////////////////////////////////////////////////////
    //
    // uniform-H
    //
    //////////////////////////////////////////////////////////////////////

    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "uniform H-matrix" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

            impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x_ref, *y, *rowcb_uni, *colcb_uni );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
        }
            
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

        x->fill( 1 );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            for ( int j = 0; j < 50; ++j )
                impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x, *y, *rowcb_uni, *colcb_uni );
            
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
    
    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "uniform H-matrix (v2)" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

            impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x_ref, *y, *rowcb_uni, *colcb_uni );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
        }
            
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

        x->fill( 1 );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            for ( int j = 0; j < 50; ++j )
                impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x, *y, *rowcb_uni, *colcb_uni );
            
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
    // H²
    //
    //////////////////////////////////////////////////////////////////////

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );
            
            impl::h2::mul_vec< value_t >( 2.0, apply_normal, *A_h2, *x_ref, *y, *rowcb_h2, *colcb_h2 );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
        }
            
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_h2->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_h2->row_is() );

        x->fill( 1 );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                // A_h2->mul_vec( 2.0, x.get(), 1.0, y.get(), hpro::apply_normal );
                impl::h2::mul_vec< value_t >( 2.0, apply_normal, *A_h2, *x, *y, *rowcb_h2, *colcb_h2 );

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

    #if defined(HLR_HAS_H2)
    
    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-matrix (Hpro)" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );
            
            impl::h2::mul_vec< value_t >( 2.0, apply_normal, *A_h21, *x_ref, *y, *rowcb_h21, *colcb_h21 );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
        }
            
        auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_h21->col_is() );
        auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_h21->row_is() );

        x->fill( 1 );
            
        for ( int i = 0; i < nbench; ++i )
        {
            tic = timer::now();
    
            for ( int j = 0; j < 50; ++j )
                // A_h2->mul_vec( 2.0, x.get(), 1.0, y.get(), hpro::apply_normal );
                impl::h2::mul_vec< value_t >( 2.0, apply_normal, *A_h21, *x, *y, *rowcb_h21, *colcb_h21 );

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
