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

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

namespace hlr { namespace uniform { namespace accu2 {
double  t_basis       = 0.0;
double  t_apply       = 0.0;
double  t_apply_uni   = 0.0;
double  t_apply_dense = 0.0;
double  t_eval        = 0.0;
double  t_eval_uni    = 0.0;
double  t_eval_rest   = 0.0;
double  t_eval_rec    = 0.0;
size_t  n_inner       = 0;
size_t  n_prodA       = 0;
size_t  n_prodB       = 0;
}}}

namespace hlr { namespace uniform { namespace accu3 {
double  t_basis      = 0.0;
double  t_apply      = 0.0;
double  t_eval       = 0.0;
double  t_eval_uni   = 0.0;
double  t_eval_rest  = 0.0;
double  t_eval_rec   = 0.0;
}}}

namespace hlr { namespace seq { namespace accu {
double  t_apply = 0.0;
double  t_eval  = 0.0;
}}}

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
        io::eps::print( *ct->root(), "ct" );
        io::eps::print( *bct->root(), "bct" );
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
    // H-LU factorization
    //
    //////////////////////////////////////////////////////////////////////

    auto  M1  = seq::matrix::copy_nonuniform< value_t >( *A2 );
    auto  REF = std::unique_ptr< hpro::TMatrix >();
            
    std::cout << term::bullet << term::bold << "H-LU" << term::reset << std::endl;
    
    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "eager" << term::reset << std::endl;
                
        auto  LU = seq::matrix::copy( *M1 );
                
        tic = timer::now();
                
        impl::lu< value_t >( *LU, acc, apx );
                
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;
                
        io::eps::print( *LU, "HLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;

        REF = std::move( LU );

        {
            //
            // try to represent L/U factors as uniform matrices with given shared bases
            //

            std::cout << "    " << term::bullet << term::bold << "testing original bases" << term::reset << std::endl;

            auto  LU2     = impl::matrix::copy_uniform< value_t >( *REF, *rowcb, *colcb );
            auto  LU2_inv = matrix::luinv_eval( *LU2 );
                    
            std::cout << "      mem    = " << format_mem( LU2->byte_size() ) << std::endl;
            std::cout << "      error  = " << format_error( norm::inv_error_2( *M1, LU2_inv ) ) << std::endl;
        }
    }// if

    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulator" << term::reset << std::endl;
                
        auto  LU = seq::matrix::copy( *M1 );
                
        tic = timer::now();
                
        impl::accu::lu< value_t >( *LU, acc, apx );
                
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        // std::cout << "      apply  " << format_time( hlr::seq::accu::t_apply ) << std::endl;
        // std::cout << "      eval   " << format_time( hlr::seq::accu::t_eval ) << std::endl;
        std::cout << "    mem    = " << format_mem( LU->byte_size() ) << std::endl;
                
        io::eps::print( *LU, "HLU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;

        {
            std::cout << "    " << term::bullet << term::bold << "sep. factors/bases" << term::reset << std::endl;
            
            auto  L                  = impl::matrix::copy_ll( *LU, unit_diag );
            auto  U                  = impl::matrix::copy_ur( *LU, general_diag );
            auto  [ rowcbL, colcbL ] = impl::matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *L, acc );
            auto  [ rowcbU, colcbU ] = impl::matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *U, acc );

            auto  L2 = impl::matrix::copy_uniform< value_t >( *L, *rowcbL, *colcbL );
            auto  U2 = impl::matrix::copy_uniform< value_t >( *U, *rowcbU, *colcbU );
            
            std::cout << "      mem L  = " << format_mem( L2->byte_size(), rowcbL->byte_size(), colcbL->byte_size() ) << std::endl;
            std::cout << "      mem U  = " << format_mem( U2->byte_size(), rowcbU->byte_size(), colcbU->byte_size() ) << std::endl;

            auto  L2_inv   = matrix::triinv_eval( *L2, blas::lower_triangular, unit_diag );
            auto  U2_inv   = matrix::triinv_eval( *U2, blas::upper_triangular, general_diag );
            auto  AxLU2    = matrix::product( *M1, U2_inv, L2_inv );
            auto  I        = matrix::identity( M1->block_is() );
            auto  inv_err2 = matrix::sum( 1.0, *I, -1.0, *AxLU2 );

            std::cout << "      error  = " << format_error( norm::spectral( *inv_err2 ) ) << std::endl;
        }
        
        {
            std::cout << "    " << term::bullet << term::bold << "joined bases" << term::reset << std::endl;
            
            auto  [ rowcb2, colcb2 ] = impl::matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *LU, hpro::fixed_prec( 1e-8 ) );
            auto  LU2                = impl::matrix::copy_uniform< value_t >( *LU, *rowcb2, *colcb2 );
                
            std::cout << "      mem    = " << format_mem( LU2->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;

            auto  A2_inv = matrix::luinv_eval( *LU2 );
                    
            std::cout << "      error  = " << format_error( norm::inv_error_2( *M1, A2_inv ) ) << std::endl;
        }
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // Uniform H-LU factorization
    //
    //////////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "Uniform H-LU" << term::reset << std::endl;
    
    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "eager" << term::reset << std::endl;
            
        auto  LU     = impl::matrix::copy( *A2 );
        auto  rowcb2 = rowcb->copy();
        auto  colcb2 = colcb->copy();
                
        matrix::replace_cluster_basis( *LU, *rowcb2, *colcb2 );
                
        tic = timer::now();
                
        impl::uniform::lu< value_t >( *LU, acc, apx, *REF );
                
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( LU->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;
                
        io::eps::print( *LU, "H2LU", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;
    }// if

    if ( false )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulated" << term::reset << std::endl;
            
        auto  LU     = impl::matrix::copy( *A2 );
        auto  rowcb2 = rowcb->copy();
        auto  colcb2 = colcb->copy();
                
        matrix::replace_cluster_basis( *LU, *rowcb2, *colcb2 );
                
        tic = timer::now();
                
        impl::uniform::accu::lu< value_t >( *LU, acc, apx, *REF );
                
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( LU->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;
                
        io::eps::print( *LU, "H2LUa", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulated v2" << term::reset << std::endl;
            
        auto  LU     = impl::matrix::copy( *A2 );
        auto  rowcb2 = rowcb->copy();
        auto  colcb2 = colcb->copy();
                
        matrix::replace_cluster_basis( *LU, *rowcb2, *colcb2 );
                
        tic = timer::now();
                
        impl::uniform::accu2::lu< value_t >( *LU, acc, apx, *REF );
                
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        // std::cout << "      basis   " << format_time( hlr::uniform::accu2::t_basis ) << std::endl;
        // std::cout << "      apply   " << format_time( hlr::uniform::accu2::t_apply ) << std::endl;
        // std::cout << "      eval    " << format_time( hlr::uniform::accu2::t_eval ) << std::endl;
        // std::cout << "        uni   " << format_time( hlr::uniform::accu2::t_eval_uni ) << std::endl;
        // std::cout << "        rest  " << format_time( hlr::uniform::accu2::t_eval_rest ) << std::endl;
        // std::cout << "        rec   " << format_time( hlr::uniform::accu2::t_eval_rec ) << std::endl;
        std::cout << "    mem    = " << format_mem( LU->byte_size(), rowcb2->byte_size(), colcb2->byte_size() ) << std::endl;
                
        io::eps::print( *LU, "H2LUa", prnopt );
                
        auto  A_inv = matrix::luinv_eval( *LU );
                    
        std::cout << "    error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;
    }// if

    if ( true )
    {
        std::cout << "  " << term::bullet << term::bold << "accumulated v3" << term::reset << std::endl;
            
        auto  A3     = impl::matrix::copy( *A2 );
        auto  L      = impl::matrix::copy_ll( *A2 );
        auto  U      = impl::matrix::copy_ur( *A2 );
        auto  rowcbA = rowcb->copy();
        auto  colcbA = colcb->copy();
        auto  rowcbL = rowcb->copy();
        auto  colcbL = colcb->copy();
        auto  rowcbU = rowcb->copy();
        auto  colcbU = colcb->copy();
                
        matrix::replace_cluster_basis( *A3, *rowcbA, *colcbA );
        matrix::replace_cluster_basis( *L, *rowcbL, *colcbL );
        matrix::replace_cluster_basis( *U, *rowcbU, *colcbU );
                
        tic = timer::now();
                
        impl::uniform::accu3::lu< value_t >( *A, *L, *U, acc, apx, *REF );
                
        toc = timer::since( tic );
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        // std::cout << "      basis   " << format_time( hlr::uniform::accu2::t_basis ) << std::endl;
        // std::cout << "      apply   " << format_time( hlr::uniform::accu2::t_apply ) << std::endl;
        // std::cout << "      eval    " << format_time( hlr::uniform::accu2::t_eval ) << std::endl;
        // std::cout << "        uni   " << format_time( hlr::uniform::accu2::t_eval_uni ) << std::endl;
        // std::cout << "        rest  " << format_time( hlr::uniform::accu2::t_eval_rest ) << std::endl;
        // std::cout << "        rec   " << format_time( hlr::uniform::accu2::t_eval_rec ) << std::endl;
        std::cout << "    mem L  = " << format_mem( L->byte_size(), rowcbL->byte_size(), colcbL->byte_size() ) << std::endl;
        std::cout << "    mem U  = " << format_mem( U->byte_size(), rowcbU->byte_size(), colcbU->byte_size() ) << std::endl;
                
        auto  L_inv  = matrix::triinv_eval( *L, blas::lower_triangular, unit_diag );
        auto  U_inv  = matrix::triinv_eval( *U, blas::upper_triangular, general_diag );
        auto  LU_inv = matrix::product( U_inv, L_inv );

        std::cout << "    error  = " << format_error( norm::inv_error_2( *M1, *LU_inv ) ) << std::endl;
        
        // auto  A_inv = matrix::luinv_eval( *LU );
                    
        // std::cout << "      error  = " << format_error( norm::inv_error_2( *M1, A_inv ) ) << std::endl;
    }// if
}
