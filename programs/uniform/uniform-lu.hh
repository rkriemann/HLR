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
#include <hlr/seq/arith.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>
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
        matrix::print_eps( *A, "A" );
        matrix::print_lvl_eps( *A, "L" );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to uniform
    //
    //////////////////////////////////////////////////////////////////////

    auto  apx = approx::RandSVD< value_t >();
    
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
    auto  error = hlr::seq::norm::spectral( *diff, true, 1e-4 );
        
    std::cout << "    error  = " << format_error( error ) << std::endl;

    if ( hpro::verbose( 3 ) )
        matrix::print_eps( *A2, "A2" );

    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////

    if ( cmdline::cluster == "tlr" )
    {
        if ( true )
        {
            auto  M1  = seq::matrix::copy_nonuniform< value_t >( *A2 );

            {
                std::cout << "  " << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
            
                auto  M3 = seq::matrix::copy( *M1 );
                
                tic = timer::now();
                
                impl::tlr::lu< value_t >( *M3, acc, apx );
                
                toc = timer::since( tic );
                std::cout << "      done in  " << format_time( toc ) << std::endl;
                std::cout << "      mem    = " << format_mem( M3->byte_size() ) << std::endl;
                
                {
                    hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                    
                    std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
                }
            }

            {
                std::cout << "  " << term::bullet << term::bold << "H-LU (lazy)" << term::reset << std::endl;
            
                auto  M3 = seq::matrix::copy( *M1 );
                
                tic = timer::now();
                
                impl::tlr::lu_lazy< value_t >( *M3, acc, apx );
                
                toc = timer::since( tic );
                std::cout << "      done in  " << format_time( toc ) << std::endl;
                std::cout << "      mem    = " << format_mem( M3->byte_size() ) << std::endl;
                
                {
                    hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                    
                    std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
                }
            }

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
                std::cout << "      mem    = " << format_mem( A3->byte_size() ) << std::endl;
                
                auto  M2 = seq::matrix::copy_nonuniform< value_t >( *A3 );
                
                // {
                //     hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                
                //     std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
                // }
                
                {
                    hpro::TLUInvMatrix  A_inv( M2.get(), hpro::block_wise, hpro::store_inverse );
                    
                    std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
                }
            }

            {
                std::cout << "  " << term::bullet << term::bold << "H²-LU (lazy)" << term::reset << std::endl;
            
                auto  A3     = impl::matrix::copy( *A2 );
                auto  rowcb2 = rowcb->copy();
                auto  colcb2 = colcb->copy();
                
                matrix::replace_cluster_basis( *A3, *rowcb2, *colcb2 );
                
                tic = timer::now();
                
                impl::uniform::tlr::lu_lazy< value_t >( *A3, acc, *A );
                
                toc = timer::since( tic );
                std::cout << "      done in  " << format_time( toc ) << std::endl;
                std::cout << "      mem    = " << format_mem( A3->byte_size() ) << std::endl;
                
                auto  M2 = seq::matrix::copy_nonuniform< value_t >( *A3 );
                
                // {
                //     hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                
                //     std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
                // }
                
                {
                    hpro::TLUInvMatrix  A_inv( M2.get(), hpro::block_wise, hpro::store_inverse );
                    
                    std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
                }
            }
        }
    }// if
    else
    {
        auto  M1  = seq::matrix::copy_nonuniform< value_t >( *A2 );
        auto  REF = std::unique_ptr< hpro::TMatrix >();
            
        if ( true )
        {
            std::cout << "  " << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
                
            auto  M3 = seq::matrix::copy( *M1 );
                
            tic = timer::now();
                
            impl::lu< value_t >( *M3, acc, apx );
                
            toc = timer::since( tic );
            std::cout << "      done in  " << format_time( toc ) << std::endl;
            std::cout << "      mem    = " << format_mem( M3->byte_size() ) << std::endl;
                
            matrix::print_eps( *M3, "HLU" );
                
            {
                hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                    
                std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
            }

            REF = std::move( M3 );
        }// if

        if ( true )
        {
            std::cout << "  " << term::bullet << term::bold << "H²-LU" << term::reset << std::endl;
            
            auto  A3     = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();
                
            matrix::replace_cluster_basis( *A3, *rowcb2, *colcb2 );
                
            tic = timer::now();
                
            impl::uniform::lu< value_t >( *A3, acc, *REF );
                
            toc = timer::since( tic );
            std::cout << "      done in  " << format_time( toc ) << std::endl;
            std::cout << "      mem    = " << format_mem( A3->byte_size() ) << std::endl;
                
            matrix::print_eps( *A3, "H2LU" );
                
            auto  M2 = seq::matrix::copy_nonuniform< value_t >( *A3 );
                
            // {
            //     hpro::TLUInvMatrix  A_inv( M3.get(), hpro::block_wise, hpro::store_inverse );
                
            //     std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
            // }
                
            {
                hpro::TLUInvMatrix  A_inv( M2.get(), hpro::block_wise, hpro::store_inverse );
                    
                std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
            }
        }// if

        if ( true )
        {
            std::cout << "  " << term::bullet << term::bold << "H²-LU (accumulated)" << term::reset << std::endl;
            
            auto  A3     = impl::matrix::copy( *A2 );
            auto  rowcb2 = rowcb->copy();
            auto  colcb2 = colcb->copy();
                
            matrix::replace_cluster_basis( *A3, *rowcb2, *colcb2 );
                
            tic = timer::now();
                
            impl::uniform::accu::lu< value_t >( *A3, acc, *REF );
                
            toc = timer::since( tic );
            std::cout << "      done in  " << format_time( toc ) << std::endl;
            std::cout << "      mem    = " << format_mem( A3->byte_size() ) << std::endl;
                
            matrix::print_eps( *A3, "H2LUa" );
                
            auto  M2 = seq::matrix::copy_nonuniform< value_t >( *A3 );
                
            {
                hpro::TLUInvMatrix  A_inv( M2.get(), hpro::block_wise, hpro::store_inverse );
                    
                std::cout << "      error  = " << format_error( inv_approx_2( M1.get(), & A_inv ) ) << std::endl;
            }
        }// if
    }// else
}
