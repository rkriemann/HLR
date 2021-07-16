//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_accu.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/sum.hh>
#include <hlr/matrix/product.hh>
#include <hlr/matrix/triinv_eval.hh>
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/matrix/identity.hh>
#include <hlr/utils/io.hh>
#include <hlr/bem/aca.hh>
#include <hlr/approx/rrqr.hh>

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
    auto  prnopt  = "noid";
    auto  acc     = gen_accuracy();
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
    auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = bem::aca_lrapx( pcoeff );
    auto  A      = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    // auto  A      = io::hpro::read( "A.hm" );
    auto  toc    = timer::since( tic );

    // io::hpro::write( *A, "A.hm" );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    seq::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *A, "A", prnopt );
        io::eps::print_mem( *A, "Am" );
        io::eps::print_lvl( *A, "L" );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    auto  apx = approx::SVD< value_t >();
    
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

    auto  diff  = matrix::sum( value_t(1), *A, value_t(-1), *A2 );
    auto  error = hlr::seq::norm::spectral( *diff, true, 1e-4 );
        
    std::cout << "    error  = " << format_error( error ) << std::endl;

    if ( hpro::verbose( 3 ) )
    {
        io::eps::print( *A2, "A2", prnopt );
        io::eps::print_mem( *A2, "A2m", prnopt );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////

    HLR_ASSERT( cmdline::cluster == "tlr" );
    
    auto  M1 = seq::matrix::copy_nonuniform< value_t >( *A2 );

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
            
        auto  M3 = impl::matrix::copy( *M1 );
                
        tic = timer::now();
                
        impl::tlr::lu< value_t >( *M3, acc, apx );
                
        toc = timer::since( tic );
        std::cout << "      done in  " << format_time( toc ) << std::endl;
        std::cout << "      mem    = " << format_mem( M3->byte_size() ) << std::endl;
                
        io::eps::print( *M3, "LU", prnopt );
                
        {
            // auto  A_inv   = hpro::TLUInvMatrix( M3.get(), hpro::block_wise, hpro::store_inverse );
            auto  A_inv   = matrix::luinv_eval( *M3 );
            auto  AxLU    = matrix::product( *M1, A_inv );
            auto  I       = matrix::identity( M1->block_is() );
            auto  inv_err = matrix::sum( 1.0, *I, -1.0, *AxLU );
                    
            std::cout << "      error  = " << format_error( norm::spectral( *inv_err ) ) << std::endl;
        }

        {
            auto  L       = impl::matrix::copy_ll( *M3, unit_diag );
            auto  U       = impl::matrix::copy_ur( *M3, general_diag );
            auto  L_inv   = matrix::triinv_eval( *L, blas::lower_triangular, unit_diag );
            auto  U_inv   = matrix::triinv_eval( *U, blas::upper_triangular, general_diag );
            auto  AxLU    = matrix::product( *M1, U_inv, L_inv );
            auto  I       = matrix::identity( M1->block_is() );
            auto  inv_err = matrix::sum( 1.0, *I, -1.0, *AxLU );
                        
            std::cout << "      error  = " << format_error( norm::spectral( *inv_err ) ) << std::endl;
        }
    }// if

    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "H-LU (lazy)" << term::reset << std::endl;
            
        auto  M3 = seq::matrix::copy( *M1 );
                
        tic = timer::now();
                
        impl::tlr::lu_lazy< value_t >( *M3, acc, apx );
                
        toc = timer::since( tic );
        std::cout << "      done in  " << format_time( toc ) << std::endl;
        std::cout << "      mem    = " << format_mem( M3->byte_size() ) << std::endl;
                
        {
            auto  A_inv   = matrix::luinv_eval( *M3 );
                    
            std::cout << "      error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;
                    
            // hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                    
            // std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
        }
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-LU" << term::reset << std::endl;
            
        auto  A3     = impl::matrix::copy( *A2 );
        auto  rowcb2 = rowcb->copy();
        auto  colcb2 = colcb->copy();
                
        matrix::replace_cluster_basis( *A3, *rowcb2, *colcb2 );

        tic = timer::now();
                
        impl::uniform::tlr::lu< value_t >( *A3, acc, apx );
                
        toc = timer::since( tic );
        std::cout << "      done in  " << format_time( toc ) << std::endl;
        std::cout << "      mem    = " << format_mem( A3->byte_size(), rowcb2->byte_size() + colcb2->byte_size() ) << std::endl;
                
        auto  M2 = seq::matrix::copy_nonuniform< value_t >( *A3 );
                
        // {
        //     hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                
        //     std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
        // }
                
        {
            auto  A_inv   = matrix::luinv_eval( *M2 );
                    
            std::cout << "      error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;

            // hpro::TLUInvMatrix  A_inv( M2.get(), hpro::block_wise, hpro::store_inverse );
                    
            // std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
        }
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-LU (sep. L/U)" << term::reset << std::endl;
            
        auto  L      = impl::matrix::copy_ll( *A2, unit_diag );
        auto  U      = impl::matrix::copy_ur( *A2, general_diag );
        auto  rowcbL = rowcb->copy();
        auto  colcbL = colcb->copy();
        auto  rowcbU = rowcb->copy();
        auto  colcbU = colcb->copy();
                
        matrix::replace_cluster_basis( *L, *rowcbL, *colcbL );
        matrix::replace_cluster_basis( *U, *rowcbU, *colcbU );

        tic = timer::now();
                
        impl::uniform::tlr::lu_sep< value_t >( *L, *U, acc, apx );
                
        toc = timer::since( tic );
        std::cout << "      done in  " << format_time( toc ) << std::endl;
        std::cout << "      mem    = " << format_mem( L->byte_size() + U->byte_size(),
                                                      rowcbL->byte_size() + colcbL->byte_size(),
                                                      rowcbU->byte_size() + colcbU->byte_size() ) << std::endl;
                
        io::eps::print( *L, "L2", prnopt );
        io::eps::print( *U, "U2", prnopt );
                
        auto  L2 = seq::matrix::copy_nonuniform< value_t >( *L );
        auto  U2 = seq::matrix::copy_nonuniform< value_t >( *U );
                
        {
            auto  L_inv   = matrix::triinv_eval( *L2, blas::lower_triangular, unit_diag );
            auto  U_inv   = matrix::triinv_eval( *U2, blas::upper_triangular, general_diag );
            auto  AxLU    = matrix::product( *M1, U_inv, L_inv );
            auto  I       = matrix::identity( M1->block_is() );
            auto  inv_err = matrix::sum( 1.0, *I, -1.0, *AxLU );
                        
            std::cout << "      error  = " << format_error( norm::spectral( *inv_err ) ) << std::endl;
                        
            // std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), LU_inv.get() ) ) << std::endl;
        }
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "H²-LU (lazy)" << term::reset << std::endl;
            
        auto  A3     = impl::matrix::copy( *A2 );
        auto  rowcb2 = rowcb->copy();
        auto  colcb2 = colcb->copy();
                
        matrix::replace_cluster_basis( *A3, *rowcb2, *colcb2 );
                
        tic = timer::now();
                
        impl::uniform::tlr::lu_lazy< value_t >( *A3, acc, apx, *A );
                
        toc = timer::since( tic );
        std::cout << "      done in  " << format_time( toc ) << std::endl;
        std::cout << "      mem    = " << format_mem( A3->byte_size(), rowcb2->byte_size() + colcb2->byte_size() ) << std::endl;
                
        auto  M2 = seq::matrix::copy_nonuniform< value_t >( *A3 );
                
        // {
        //     hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                
        //     std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
        // }
                
        {
            auto  A_inv   = matrix::luinv_eval( *M2 );
                    
            std::cout << "      error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;

            // hpro::TLUInvMatrix  A_inv( M2.get(), hpro::block_wise, hpro::store_inverse );
                    
            // std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
        }
    }// if
}
