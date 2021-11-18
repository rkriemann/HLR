//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlib-config.h>

#if defined(USE_LIC_CHECK)
#define HAS_H2
#endif

#if defined( HAS_H2 )
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
// return min/avg/max rank of given cluster basis
//
template < typename cluster_basis_t >
std::tuple< uint, uint, uint >
rank_info ( const cluster_basis_t &  cb );

std::tuple< uint, uint, uint >
rank_info ( const hpro::TMatrix &  M );

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
        io::vtk::print( *coord, "coord" );
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
        io::eps::print( *A, "A", "noid" );
        io::eps::print_lvl( *A, "L" );
    }// if

    const auto  normA = hlr::norm::spectral( *A, true, 1e-4 );

    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A ) ) << std::endl;
    
    {
        auto  [ kmin, kavg, kmax ] = rank_info( *A );
    
        std::cout << "    ranks  : " << kmin << " / " << kavg << " / " << kmax << std::endl;
    }
    
    //////////////////////////////////////////////////////////////////////
    //
    // directly build uniform matrix
    //
    //////////////////////////////////////////////////////////////////////

    auto  rowcb_uni = std::unique_ptr< matrix::cluster_basis< value_t > >();
    auto  colcb_uni = std::unique_ptr< matrix::cluster_basis< value_t > >();
    auto  A_uni     = std::unique_ptr< hpro::TMatrix >();
    auto  apx       = approx::SVD< value_t >();

    if ( true )
    {
        std::cout << term::bullet << term::bold << "uniform H-matrix (lvl)" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_lvl( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;
        
        auto  [ row_min, row_avg, row_max ] = rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = rank_info( *colcb );

        std::cout << "    ranks  " << std::endl
                  << "      row  : " << row_min << " / " << row_avg << " / " << row_max << std::endl
                  << "      col  : " << col_min << " / " << col_avg << " / " << col_max << std::endl;
        
        {
            auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
            auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }
    }

    {
        std::cout << term::bullet << term::bold << "uniform H-matrix (rec)" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb, A2 ] = impl::matrix::build_uniform_rec( *A, apx, acc, nseq );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        auto  [ row_min, row_avg, row_max ] = rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = rank_info( *colcb );

        std::cout << "    ranks  " << std::endl
                  << "      row  : " << row_min << " / " << row_avg << " / " << row_max << std::endl
                  << "      col  : " << col_min << " / " << col_avg << " / " << col_max << std::endl;
        
        if ( hpro::verbose( 3 ) )
        {
            io::eps::print( *A2, "A2", "noid" );
            io::eps::print( *rowcb, "rowcb2" );
            io::eps::print( *colcb, "colcb2" );
        }// if
        
        {
            auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
            auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        }

        if ( false )
        {
            std::cout << "    " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

            using single_t = math::decrease_precision_t< value_t >;

            auto  rowcbs = matrix::copy< single_t >( *rowcb );
            auto  colcbs = matrix::copy< single_t >( *colcb );
            
            auto  rowcbv = matrix::copy< value_t >( *rowcbs );
            auto  colcbv = matrix::copy< value_t >( *colcbs );

            std::cout << "      mem    = " << format_mem( A2->byte_size(), rowcbs->byte_size(), colcbs->byte_size() ) << std::endl;
            
            matrix::replace_cluster_basis( *A2, *rowcbv, *colcbv );
            
            {
                auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
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
            
            auto  rowcbh = matrix::copy< half_t >( *rowcb );
            auto  colcbh = matrix::copy< half_t >( *colcb );
            
            auto  rowcbs = matrix::copy< single_t >( *rowcbh );
            auto  colcbs = matrix::copy< single_t >( *colcbh );

            auto  rowcbv = matrix::copy< value_t >( *rowcbs );
            auto  colcbv = matrix::copy< value_t >( *colcbs );

            std::cout << "      mem    = " << format_mem( A2->byte_size(), rowcbh->byte_size(), colcbh->byte_size() ) << std::endl;
            
            matrix::replace_cluster_basis( *A2, *rowcbv, *colcbv );
            
            {
                auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
                auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
                std::cout << "      error  = " << format_error( error / normA ) << std::endl;
            }
        }
        #endif

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

        auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;
    }// if
    
    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    #if defined( HAS_H2 )
    
    auto  rowcb_h2 = std::unique_ptr< hpro::TClusterBasis< value_t > >();
    auto  colcb_h2 = std::unique_ptr< hpro::TClusterBasis< value_t > >();
    auto  A_h2     = std::unique_ptr< hpro::TMatrix >();
    
    if ( true )
    {
        std::cout << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;

        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        hpro::THClusterBasisBuilder< value_t >  bbuilder;

        tic = timer::now();
    
        auto [ rowcb, colcb ] = bbuilder.build( ct->root(), ct->root(), A.get(), acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;

        auto  [ row_min, row_avg, row_max ] = rank_info( *rowcb );
        auto  [ col_min, col_avg, col_max ] = rank_info( *colcb );

        std::cout << "    ranks  " << std::endl
                  << "      row  : " << row_min << " / " << row_avg << " / " << row_max << std::endl
                  << "      col  : " << col_min << " / " << col_avg << " / " << col_max << std::endl;
        
        if ( verbose( 3 ) )
        {
            io::eps::print( *rowcb, "rowcb" );
            io::eps::print( *colcb, "colcb" );
        }// if

        std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

        tic = timer::now();
    
        auto  A2 = std::move( to_h2( A.get(), rowcb.get(), colcb.get() ) );
    
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A2->byte_size(), rowcb->byte_size(), colcb->byte_size() ) << std::endl;

        auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "    error  = " << format_error( error / normA ) << std::endl;
        
        if ( hpro::verbose( 3 ) )
            io::eps::print( *A2, "A2", "noid" );

        //
        // preserve for MVM
        //

        A_h2     = std::move( A2 );
        rowcb_h2 = std::move( rowcb );
        colcb_h2 = std::move( colcb );
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
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "uniform H-matrix" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

            impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x_ref, *y, *rowcb_uni, *colcb_uni );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() ) << std::endl;
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
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    #if defined( HAS_H2 )
    
    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-matrix" << term::reset << std::endl;

        {
            auto  y = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );
            
            impl::h2::mul_vec< value_t >( 2.0, apply_normal, *A_h2, *x_ref, *y, *rowcb_h2, *colcb_h2 );
            
            y->axpy( -1.0, y_ref.get() );
            std::cout << "    error  = " << format_error( y->norm2() ) << std::endl;
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

    #endif
    #endif
}

//
// return min/avg/max rank of given cluster basis
//
template < typename cluster_basis_t >
std::tuple< uint, size_t, uint, size_t >
rank_info_helper ( const cluster_basis_t &  cb )
{
    uint    min_rank = cb.rank();
    uint    max_rank = cb.rank();
    size_t  sum_rank = cb.rank();
    size_t  nnodes   = cb.rank() > 0 ? 1 : 0;

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            auto [ min_i, sum_i, max_i, n_i ] = rank_info_helper( *cb.son(i) );

            if      ( min_rank == 0 ) min_rank = min_i;
            else if ( min_i    != 0 ) min_rank = std::min( min_rank, min_i );
            
            max_rank  = std::max( max_rank, max_i );
            sum_rank += sum_i;
            nnodes   += n_i;
        }// for
    }// if

    return { min_rank, sum_rank, max_rank, nnodes };
}

template < typename cluster_basis_t >
std::tuple< uint, uint, uint >
rank_info ( const cluster_basis_t &  cb )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = rank_info_helper( cb );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

//
// return min/avg/max rank of given matrix
//
std::tuple< uint, size_t, uint, size_t >
rank_info_helper ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto    B        = cptrcast( &M, hpro::TBlockMatrix );
        uint    min_rank = 0;
        uint    max_rank = 0;
        size_t  sum_rank = 0;
        size_t  nnodes   = 0;

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto [ min_ij, sum_ij, max_ij, n_ij ] = rank_info_helper( *B->block( i, j ) );
                
                if      ( min_rank == 0 ) min_rank = min_ij;
                else if ( min_ij   != 0 ) min_rank = std::min( min_rank, min_ij );
                
                max_rank  = std::max( max_rank, max_ij );
                sum_rank += sum_ij;
                nnodes   += n_ij;
            }// for
        }// for

        return { min_rank, sum_rank, max_rank, nnodes };
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, hpro::TRkMatrix );

        return { R->rank(), R->rank(), R->rank(), R->rank() > 1 ? 1 : 0 };
    }// if

    return { 0, 0, 0, 0 };
}

std::tuple< uint, uint, uint >
rank_info ( const hpro::TMatrix & M )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = rank_info_helper( M );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}
