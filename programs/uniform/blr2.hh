//
// Project     : HLR
// Program     : blr2
// Description : program for testing uniform BLR matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/config.h>

#include <hlr/arith/mulvec.hh>
#include <hlr/arith/uniform.hh>
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

    auto  runtime = std::vector< double >();
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );
    
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();

    // adjust ntile to √n
    if ( cmdline::ntile == 0 )
    {
        auto  sqn = std::sqrt( double(coord->ncoord()) );
        auto  lg2 = std::log2( sqn );
        auto  up  = 1 << uint( std::ceil( lg2 ) );
        auto  lo  = 1 << uint( std::floor( lg2 ) );

        // round to next larger power of 2
        ntile = up;
        
        // round to nearest power of 2
        // if (( up - sqn ) < ( sqn - lo )) ntile = up;
        // else                             ntile = lo;

        std::cout << "    ntile  = " << ntile << std::endl;
    }
    
    auto  ct  = cluster::sfc::cluster( hlr::cluster::sfc::blr, *coord, ntile );

    if ( false )
    {
        auto  root  = ct->root();
        auto  label = std::vector< uint >( root->size() );

        for ( uint  i = 0; i < root->nsons(); ++i )
        {
            auto  son = root->son( i );
            
            for ( idx_t  ii = son->first(); ii <= son->last(); ++ii )
            {
                auto  pii = ct->perm_i2e()->permute( ii );

                label[ pii ] = i;
            }// for
        }// for

        io::vtk::print( *coord, label, "coord_l1" );
    }
    
    auto  bct = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        io::vtk::print( *coord, "coord" );
        io::eps::print( *ct->root(), "ct" );
        io::vtk::print( *ct->root(), 1, "ct" );
        io::eps::print( *bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = bem::aca_lrapx( pcoeff );
    auto  cbapx  = approx::SVD< value_t >();

    tic = timer::now();

    auto  [ rowcb, colcb, A ] = impl::matrix::tlr::build_uniform( bct->root(), pcoeff, lrapx, cbapx, acc );
    
    toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << term::bold << A->nrows() << " × " << A->ncols() << term::reset << std::endl;
    std::cout << "    mem    = " << format_mem( rowcb->byte_size(), colcb->byte_size(), A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    // seq::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid" );

    const auto  normA = hlr::norm::spectral( impl::arithmetic, *A, 1e-4 );

    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A ) ) << std::endl;
    
    {
        auto  [ kmin, kavg, kmax ] = matrix::rank_info( *A );
    
        std::cout << "    ranks  = " << kmin << " … " << kavg << " … " << kmax << std::endl;
    }

    //
    // build with sep. bases
    //
    
    std::cout << term::bullet << term::bold
              << "BLR² with sep. couplings"
              << term::reset << std::endl;
    
    tic = timer::now();
    
    auto  [ rowcb2, colcb2, A2 ] = impl::matrix::tlr::build_uniform_sep( bct->root(), pcoeff, lrapx, cbapx, acc, cmdline::compress );
    
    toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( rowcb2->byte_size(), colcb2->byte_size(), A2->byte_size() ) << std::endl;
    std::cout << "    |A|    = " << format_norm( norm::frobenius( *A2 ) ) << std::endl;

    //
    // compare with reference matrix
    //
 
    if (( cmdline::ref == "H" ) || ( cmdline::ref == "BLR" ))
    {
        std::cout << term::bullet << term::bold
                  << "BLR reference"
                  << term::reset << std::endl;
        
        tic = timer::now();

        auto  H = impl::matrix::build( bct->root(), pcoeff, lrapx, acc );
        
        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

        auto  [ kmin, kavg, kmax ] = matrix::rank_info( *H );
    
        std::cout << "    ranks  = " << kmin << " … " << kavg << " … " << kmax << std::endl;
        
        auto  normH = hlr::norm::spectral( impl::arithmetic, *H, 1e-4 );

        {
            auto  diff  = matrix::sum( 1, *H, -1, *A );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normH ) << std::endl;
        }

        {
            auto  diff  = matrix::sum( 1, *H, -1, *A2 );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normH ) << std::endl;
        }

        std::cout << "  " << term::bullet << term::bold << "BLR² matrix" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb3, colcb3, A3 ] = impl::matrix::tlr::build_uniform( *H, cbapx, acc );

        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;

        const auto  mem_uni = A3->byte_size();
        const auto  mem_rcb = rowcb3->byte_size();
        const auto  mem_ccb = colcb3->byte_size();
    
        std::cout << "    mem    = " << format_mem( mem_rcb, mem_ccb, mem_uni, mem_rcb + mem_ccb + mem_uni ) << std::endl;

        {
            auto  [ kmin, kavg, kmax ] = matrix::rank_info( *A3 );
    
            std::cout << "    ranks  = " << kmin << " … " << kavg << " … " << kmax << std::endl;
        }
    
        {
            auto  diff  = matrix::sum( 1, *H, -1, *A3 );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error / normH ) << std::endl;
        }
    }// if
    
    if ( cmdline::compress )
    {
        std::cout << "  "
                  << term::bullet << term::bold
                  << "compression ("
                  << "ε = " << boost::format( "%.2e" ) % cmdline::eps
                  << ", "
                  << hlr::compress::provider << " + " << hlr::compress::valr::provider << ')'
                  << term::reset << std::endl;

        auto  lacc  = absolute_prec( cmdline::eps );
        
        impl::matrix::compress( *A, lacc );

        const auto  zmem = A->byte_size();
        
        std::cout << "    mem    = " << format_mem( zmem ) << std::endl;
    }// if

    return;
    
    // //////////////////////////////////////////////////////////////////////
    // //
    // // directly build uniform matrix
    // //
    // //////////////////////////////////////////////////////////////////////

    // auto  apx = approx::SVD< value_t >();

    // std::cout << term::bullet << term::bold << "BLR² matrix" << term::reset << std::endl;
    
    // tic = timer::now();
    
    // auto  [ rowcb, colcb, A_uni ] = impl::matrix::tlr::build_uniform( *A, apx, acc );

    // toc = timer::since( tic );
    // std::cout << "    done in  " << format_time( toc ) << std::endl;

    // const auto  mem_uni = A_uni->byte_size();
    // const auto  mem_rcb = rowcb->byte_size();
    // const auto  mem_ccb = colcb->byte_size();
    
    // std::cout << "    mem    = " << format_mem( mem_rcb, mem_ccb, mem_uni, mem_rcb + mem_ccb + mem_uni ) << std::endl;

    // {
    //     auto  [ row_min, row_avg, row_max ] = matrix::rank_info( *rowcb );
    //     auto  [ col_min, col_avg, col_max ] = matrix::rank_info( *colcb );

    //     std::cout << "    ranks  = "
    //               << row_min << " … " << row_avg << " … " << row_max << " / "
    //               << col_min << " … " << col_avg << " … " << col_max << std::endl;
    // }
    
    // if ( hpro::verbose( 3 ) )
    // {
    //     io::eps::print( *A_uni, "A_uni", "noid" );
    //     io::eps::print( *rowcb, "rowcb2" );
    //     io::eps::print( *colcb, "colcb2" );
    // }// if
        
    // {
    //     auto  diff  = matrix::sum( 1, *A, -1, *A_uni );
    //     auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
    //     std::cout << "    error  = " << format_error( error / normA ) << std::endl;
    // }
    
    // {
    //     auto  error = impl::norm::frobenius( 1, *A, -1, *A_uni );
        
    //     std::cout << "    error  = " << format_error( error, error / normA ) << std::endl;
    // }

    // if ( cmdline::compress )
    // {
    //     std::cout << "  "
    //               << term::bullet << term::bold
    //               << "compression ("
    //               << "ε = " << boost::format( "%.2e" ) % cmdline::eps
    //               << ", "
    //               << hlr::compress::provider << " + " << hlr::compress::valr::provider << ')'
    //               << term::reset << std::endl;

    //     auto  lacc  = absolute_prec( cmdline::eps );
        
    //     impl::matrix::compress< matrix::shared_cluster_basis< value_t > >( *rowcb, lacc );
    //     impl::matrix::compress< matrix::shared_cluster_basis< value_t > >( *colcb, lacc );
    //     impl::matrix::compress( *A_uni, lacc );

    //     const auto  zmem_uni = A_uni->byte_size();
    //     const auto  zmem_rcb = rowcb->byte_size();
    //     const auto  zmem_ccb = colcb->byte_size();
        
    //     std::cout << "    mem    = " << format_mem( zmem_rcb, zmem_ccb, zmem_uni, zmem_rcb + zmem_ccb + zmem_uni ) << std::endl;
    // }// if
    
    // //////////////////////////////////////////////////////////////////////
    // //
    // // H-matrix matrix vector multiplication
    // //
    // //////////////////////////////////////////////////////////////////////
    
    // if ( nbench > 0 )
    // {
    //     std::cout << term::bullet << term::bold
    //               << "mat-vec"
    //               << term::reset << std::endl;

    //     const uint  nmvm      = 50;
    
    //     const auto  flops_h   = nmvm * hlr::mul_vec_flops( apply_normal, *A );
    //     const auto  flops_uni = nmvm * hlr::uniform::mul_vec_flops( apply_normal, *A_uni, *rowcb, *colcb );

    //     const auto  bytes_h   = nmvm * hlr::mul_vec_datasize( apply_normal, *A );
    //     const auto  bytes_uni = nmvm * hlr::uniform::mul_vec_datasize( apply_normal, *A_uni, *rowcb, *colcb );

    //     std::cout << "  " << term::bullet << term::bold << "FLOPs/byte " << term::reset << std::endl;
    //     std::cout << "    H    = " << format_flops( flops_h   ) << ", " << flops_h   / bytes_h   << std::endl;
    //     std::cout << "    UniH = " << format_flops( flops_uni ) << ", " << flops_uni / bytes_uni << std::endl;
    
    //     //
    //     // set up reference vector for mat-vec error tests
    //     //
        
    //     auto  x_ref = std::make_unique< vector::scalar_vector< value_t > >( A->col_is() );
    //     auto  y_ref = std::make_unique< vector::scalar_vector< value_t > >( A->row_is() );

    //     x_ref->fill( 1 );
    //     impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *A, *x_ref, *y_ref );
    
    //     if ( true )
    //     {
    //         std::cout << "  " << term::bullet << term::bold << "H-matrices" << term::reset << std::endl;
        
    //         auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
    //         auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //         x->fill( 1 );

    //         for ( int i = 0; i < nbench; ++i )
    //         {
    //             tic = timer::now();
    
    //             for ( int j = 0; j < nmvm; ++j )
    //                 impl::mul_vec< value_t >( 2.0, hpro::apply_normal, *A, *x, *y );

    //             toc = timer::since( tic );
    //             runtime.push_back( toc.seconds() );
        
    //             std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

    //             if ( i < nbench-1 )
    //                 y->fill( 1 );
    //         }// for
        
    //         if ( nbench > 1 )
    //             std::cout << term::rollback << term::clearline << "      runtime = "
    //                       << format_time( min( runtime ), median( runtime ), max( runtime ) );
    //         std::cout << std::endl;
        
    //         std::cout << "    flops  = " << format_flops( flops_h, min( runtime ) ) << std::endl;
        
    //         runtime.clear();
    //     }// if

    //     if ( true )
    //     {
    //         std::cout << "  " << term::bullet << term::bold << "BLR matrix" << term::reset << std::endl;
        
    //         {
    //             auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //             impl::tlr::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x_ref, *y );
            
    //             y->axpy( -1.0, y_ref.get() );
    //             std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
    //         }
            
    //         auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
    //         auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //         x->fill( 1 );

    //         for ( int i = 0; i < nbench; ++i )
    //         {
    //             tic = timer::now();
    
    //             for ( int j = 0; j < nmvm; ++j )
    //                 impl::tlr::mul_vec< value_t >( 2.0, hpro::apply_normal, *A, *x, *y );

    //             toc = timer::since( tic );
    //             runtime.push_back( toc.seconds() );
        
    //             std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;

    //             if ( i < nbench-1 )
    //                 y->fill( 1 );
    //         }// for
        
    //         if ( nbench > 1 )
    //             std::cout << term::rollback << term::clearline << "      runtime = "
    //                       << format_time( min( runtime ), median( runtime ), max( runtime ) );
    //         std::cout << std::endl;
        
    //         std::cout << "    flops  = " << format_flops( flops_h, min( runtime ) ) << std::endl;
        
    //         runtime.clear();
    //     }// if

    //     //////////////////////////////////////////////////////////////////////
    //     //
    //     // uniform-H
    //     //
    //     //////////////////////////////////////////////////////////////////////

    //     if ( true )
    //     {
    //         std::cout << "  " << term::bullet << term::bold << "BLR² matrix (general)" << term::reset << std::endl;

    //         {
    //             auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //             impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x_ref, *y, *rowcb, *colcb );
            
    //             y->axpy( -1.0, y_ref.get() );
    //             std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
    //         }
            
    //         auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
    //         auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //         x->fill( 1 );
            
    //         for ( int i = 0; i < nbench; ++i )
    //         {
    //             tic = timer::now();
            
    //             for ( int j = 0; j < nmvm; ++j )
    //                 impl::uniform::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x, *y, *rowcb, *colcb );
            
    //             toc = timer::since( tic );
    //             runtime.push_back( toc.seconds() );
            
    //             std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;
            
    //             if ( i < nbench-1 )
    //                 y->fill( 1 );
    //         }// for
        
    //         if ( nbench > 1 )
    //             std::cout << term::rollback << term::clearline << "      runtime = "
    //                       << format_time( min( runtime ), median( runtime ), max( runtime ) );
    //         std::cout << std::endl;

    //         std::cout << "    flops  = " << format_flops( flops_uni, min( runtime ) ) << std::endl;
        
    //         runtime.clear();
    //     }// if

    //     if ( true )
    //     {
    //         std::cout << "  " << term::bullet << term::bold << "BLR² matrix (tlr)" << term::reset << std::endl;

    //         {
    //             auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //             impl::uniform::tlr::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x_ref, *y, *rowcb, *colcb );
            
    //             y->axpy( -1.0, y_ref.get() );
    //             std::cout << "    error  = " << format_error( y->norm2() / y_ref->norm2() ) << std::endl;
    //         }
            
    //         auto  x = std::make_unique< vector::scalar_vector< value_t > >( A_uni->col_is() );
    //         auto  y = std::make_unique< vector::scalar_vector< value_t > >( A_uni->row_is() );

    //         x->fill( 1 );
            
    //         for ( int i = 0; i < nbench; ++i )
    //         {
    //             tic = timer::now();
            
    //             for ( int j = 0; j < nmvm; ++j )
    //                 impl::uniform::tlr::mul_vec( value_t(2), hpro::apply_normal, *A_uni, *x, *y, *rowcb, *colcb );
            
    //             toc = timer::since( tic );
    //             runtime.push_back( toc.seconds() );
            
    //             std::cout << term::rollback << term::clearline << "      mvm in   " << format_time( toc ) << term::flush;
            
    //             if ( i < nbench-1 )
    //                 y->fill( 1 );
    //         }// for
        
    //         if ( nbench > 1 )
    //             std::cout << term::rollback << term::clearline << "      runtime = "
    //                       << format_time( min( runtime ), median( runtime ), max( runtime ) );
    //         std::cout << std::endl;

    //         std::cout << "    flops  = " << format_flops( flops_uni, min( runtime ) ) << std::endl;
        
    //         runtime.clear();
    //     }// if
    // }// if
}
