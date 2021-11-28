//
// Project     : HLR
// Program     : combustion
// Description : compression of datasets from combustion simulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hlr/utils/io.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/compress.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/arith/norm.hh>
#include <hlr/tbb/matrix.hh>
#include <hlr/tbb/convert.hh>
#include <hlr/utils/tensor.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

using indexset = hpro::TIndexSet;

template < typename value_t, typename approx_t >
std::unique_ptr< hpro::TMatrix >
build ( const indexset &                 rowis,
        const indexset &                 colis,
        const blas::matrix< value_t > &  D,
        const hpro::TTruncAcc &          acc,
        const approx_t &                 approx,
        const size_t                     ntile,
        const int                        zfp_rate )
{
    using namespace hlr::matrix;
    
    if ( std::min( D.nrows(), D.ncols() ) <= ntile )
    {
        //
        // build leaf
        //
        // Apply low-rank approximation and compare memory consumption
        // with dense representation. If low-rank format uses less memory
        // the leaf is represented as low-rank (considered admissible).
        // Otherwise a dense representation is used.
        //

        auto  Dc       = blas::copy( D );  // do not modify D (!)

        // DEBUG {
        // io::matlab::write( Dc, "D" );
        // DEBUG }
        
        auto  [ U, V ] = approx( Dc, acc );

        // DEBUG {
        // io::matlab::write( U, "U" );
        // io::matlab::write( V, "V" );
        
        // auto  M = blas::prod( U, blas::adjoint( V ) );

        // blas::add( value_t(-1), D, M );

        // std::cout << rowis.to_string() << " × " << colis.to_string() << " : " << format_error( blas::norm_F( M ) ) << std::endl;
        // std::cout << rowis.to_string() << " × " << colis.to_string() << std::endl;
            
        // DEBUG }
        
        if ( U.byte_size() + V.byte_size() < Dc.byte_size() )
        {
            auto  M = std::make_unique< lrmatrix >( rowis, colis, std::move( U ), std::move( V ) );

            if ( zfp_rate > 0 )
                M->compress( zfp_config_rate( zfp_rate, false ) );
            
            return M;
        }// if
        else
        {
            auto  M = std::make_unique< dense_matrix >( rowis, colis, blas::copy( D ) );

            if ( zfp_rate > 0 )
                M->compress( zfp_config_rate( zfp_rate, false ) );

            return M;
        }// else
    }// if
    else
    {
        //
        // Recursion
        //
        // If all sub blocks are low-rank, an agglomorated low-rank matrix of all sub-blocks
        // is constructed. If the memory of this low-rank matrix is smaller compared to the
        // combined memory of the sub-block, it is kept. Otherwise a block matrix with the
        // already constructed sub-blocks is created.
        //

        const auto  mid_row = ( rowis.first() + rowis.last() + 1 ) / 2;
        const auto  mid_col = ( colis.first() + colis.last() + 1 ) / 2;

        indexset    sub_rowis[2] = { indexset( rowis.first(), mid_row-1 ), indexset( mid_row, rowis.last() ) };
        indexset    sub_colis[2] = { indexset( colis.first(), mid_col-1 ), indexset( mid_col, colis.last() ) };
        auto        sub_D        = tensor2< std::unique_ptr< hpro::TMatrix > >( 2, 2 );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 ),
            [&,ntile] ( const auto &  r )
            // auto  r = ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 );
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                               sub_colis[j] - colis.first() );
                
                        sub_D(i,j) = build( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zfp_rate );

                        HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );

                        // ensure value type equals given value type for lowrank blocks
                        if ( is_generic_lowrank( *sub_D(i,j) ) )
                            HLR_ASSERT( blas::value_type_v< value_t > == ptrcast( sub_D(i,j).get(), lrmatrix )->value_type() );
                    }// for
            } );

        bool  all_lowrank = true;

        for ( uint  i = 0; i < 2; ++i )
            for ( uint  j = 0; j < 2; ++j )
                if ( ! is_generic_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;
        
        if ( all_lowrank )
        {
            //
            // construct larger lowrank matrix out of smaller sub blocks
            //

            // compute initial total rank
            uint  rank_sum = 0;

            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    rank_sum += ptrcast( sub_D(i,j).get(), lrmatrix )->U< value_t >().ncols();

            // copy sub block data into global structure
            auto    U    = blas::matrix< value_t >( rowis.size(), rank_sum );
            auto    V    = blas::matrix< value_t >( colis.size(), rank_sum );
            auto    pos  = 0; // pointer to next free space in U/V
            size_t  smem = 0; // holds memory of sub blocks
            
            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                {
                    auto  Rij   = ptrcast( sub_D(i,j).get(), lrmatrix );
                    auto  Uij   = Rij->U< value_t >();
                    auto  Vij   = Rij->V< value_t >();
                    auto  U_sub = U( sub_rowis[i] - rowis.first(), blas::range( pos, pos + Uij.ncols() ) );
                    auto  V_sub = V( sub_colis[j] - colis.first(), blas::range( pos, pos + Uij.ncols() ) );

                    blas::copy( Uij, U_sub );
                    blas::copy( Vij, V_sub );

                    pos  += Uij.ncols();
                    smem += Uij.byte_size() + Vij.byte_size();
                }// for
            
            auto  [ W, X ] = approx( U, V, acc );

            if ( W.byte_size() + X.byte_size() < smem )
            {
                //
                // large low-rank more memory efficient: keep it
                //
                
                return std::make_unique< lrmatrix >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // not all low-rank or memory gets larger: construct block matrix
        //

        auto  B = std::make_unique< hpro::TBlockMatrix >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
            for ( uint  j = 0; j < 2; ++j )
                B->set_block( i, j, sub_D(i,j).release() );

        return B;
    }// else
}

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

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    // {
    //     auto  t = tensor3< value_t >( 2, 2, 2 );

    //     for ( uint k = 0; k < 2; ++k )
    //         for ( uint i = 0; i < 2; ++i )
    //             for ( uint j = 0; j < 2; ++j )
    //                 t(i,j,k) = k*2*2 + i*2 + j;

    //     for ( uint k = 0; k < 2; ++k )
    //     {
    //         for ( uint i = 0; i < 2; ++i )
    //         {
    //             for ( uint j = 0; j < 2; ++j )
    //                 std::cout << t(i,j,k) << " ";
    //             std::cout << std::endl;
    //         }// for
    //         std::cout << std::endl;
    //     }// for

    //     return;
    // }
    
    //
    // read dataset
    //
    
    std::cout << "  " << term::bullet << term::bold << "reading data" << term::reset << std::endl;
    
    auto  D     = io::matlab::read< value_t >( cmdline::matrixfile );
    auto  mem_D = D.byte_size();

    std::cout << "    size   = " << D.nrows() << " × " << D.ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( mem_D ) << std::endl;
    
    auto  norm_D  = blas::norm_F( D );

    //
    // compress data and replace content
    //

    {
        auto  delta = norm_D * hlr::cmdline::eps / D.nrows();
    
        std::cout << "  " << term::bullet << term::bold << "compression, ε = " << delta << term::reset << std::endl;
        
        auto    T     = blas::copy( D );
        auto    acc   = local_accuracy( delta );
        auto    apx   = approx::SVD< value_t >();
        size_t  csize = 0;

        matrix::compress_replace( indexset( 0, D.nrows()-1 ),
                                  indexset( 0, D.ncols()-1 ),
                                  T, acc, apx,
                                  cmdline::ntile,
                                  cmdline::zfp_rate,
                                  csize );
            
        std::cout << "    mem    = " << format_mem( csize ) << std::endl;
        std::cout << "     %full = " << format( "%.2f %%" ) % ( 100.0 * ( double( csize ) / double( mem_D ) )) << std::endl;

        io::matlab::write( T, "T" );
        
        blas::add( value_t(-1), D, T );

        const auto  error = blas::norm_F( T );
        
        std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;
    }
    
    //
    // convert matrix to H-matrix
    //

    auto  delta = 100.0 * norm_D * hlr::cmdline::eps / D.nrows();
    
    std::cout << "  " << term::bullet << term::bold << "H-matrix, ε = " << delta << term::reset << std::endl;

    auto  acc   = local_accuracy( delta );
    auto  apx   = approx::SVD< value_t >();
    auto  tic   = timer::now();
    auto  A     = build( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, apx, cmdline::ntile, cmdline::zfp_rate );
    // auto  A     = hlr::tbb::matrix::build( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, apx, cmdline::ntile );
    auto  toc   = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    // auto  DT      = hlr::tbb::matrix::convert_to_dense< value_t >( *A );

    // io::matlab::write( blas::mat< value_t >( *DT ), "T" );
    
    auto  DM      = hpro::TDenseMatrix( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D );
    auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), DM );
    auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );
    auto  mem_A   = A->byte_size();
   
    std::cout << "    mem    = " << format_mem( mem_A ) << std::endl;
    std::cout << "     %full = " << format( "%.2f %%" ) % ( 100.0 * ( double( mem_A ) / double( mem_D ) )) << std::endl;
    std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;
}

#include <tbb/global_control.h>

template < typename problem_t >
void
framework_main ()
{
    auto                   param = ::tbb::global_control::max_allowed_parallelism;
    ::tbb::global_control  tbb_control( param, ( hlr::cmdline::nthreads > 0 ? hlr::cmdline::nthreads : ::tbb::global_control::active_value( param ) ) );
    
    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
