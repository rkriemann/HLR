//
// Project     : HLR
// Module      : tile-hodlr.hh
// Description : geeric code for tile-based HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/algebra/mul_vec.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/hodlr.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/seq/norm.hh"
#include "hlr/seq/arith_tiled_v2.hh"
#include "hlr/seq/arith_tiled_v3.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    
    auto  tic     = timer::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::hodlr::cluster( *coord, ntile );
    auto  bct     = cluster::hodlr::blockcluster( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    auto  acc    = gen_accuracy();
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    if ( false )
    {
        std::cout << term::bullet << term::bold << "LU ( Tiled-HODLR v3 " << impl_name << " )" << term::reset << std::endl;

        // auto  B   = ptrcast( A.get(), TBlockMatrix );
        // auto  A01 = ptrcast( B->block( 0, 1 ), TRkMatrix );

        // DBG::write( A01, "A.mat", "A" );
        
        // auto  C01 = std::make_unique< matrix::tiled_lrmatrix< double > >( A01->row_is(), A01->col_is(), ntile, A01->blas_rmat_A(), A01->blas_rmat_B() );
            
        // std::cout << norm_2( A01 ) << ", " << seq::norm::norm_F( *A01 ) << std::endl;
        // std::cout << norm_2( C01.get() ) << ", " << seq::norm::norm_F( *C01 ) << std::endl;
        
        // std::cout << norm_2( A.get() ) << std::endl;
        
        A = impl::matrix::copy_tiled< double >( *A, ntile );
        
        auto  C = impl::matrix::copy( *A );

        // std::cout << norm_2( A.get() ) << std::endl;
        // std::cout << norm_2( C.get() ) << std::endl;

        std::vector< double >  runtime;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            hlr::seq::tiled3::hodlr::lu( C.get(), acc, ntile );

            toc = timer::since( tic );

            std::cout << "    done in " << format_time( toc ) << std::endl;
            
            runtime.push_back( toc.seconds() );

            if ( i < nbench-1 )
                impl::matrix::copy_to( *A, *C );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        // {
        //     auto  T1 = hlr::seq::matrix::copy_nontiled< double >( *C );
        //     auto  T2 = hpro::to_dense( T1.get() );

        //     hpro::write_matrix( T2.get(), "A.mat", "A" );
        // }
    
        // std::cout << norm_2( C.get() ) << std::endl;
        
        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem   = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
        
        return;
    }// if
    else if ( true )
    {
        std::cout << term::bullet << term::bold << "LU ( Tiled-HODLR v2 " << impl_name << " )" << term::reset << std::endl;

        // auto  B   = ptrcast( A.get(), TBlockMatrix );
        // auto  A01 = ptrcast( B->block( 0, 1 ), TRkMatrix );

        // DBG::write( A01, "A.mat", "A" );
        
        // auto  C01 = std::make_unique< matrix::tiled_lrmatrix< double > >( A01->row_is(), A01->col_is(), ntile, A01->blas_rmat_A(), A01->blas_rmat_B() );
            
        // std::cout << norm_2( A01 ) << ", " << seq::norm::norm_F( *A01 ) << std::endl;
        // std::cout << norm_2( C01.get() ) << ", " << seq::norm::norm_F( *C01 ) << std::endl;
        
        // std::cout << norm_2( A.get() ) << std::endl;

        {
            auto  get_vec = [] ()
            {
                blas::Vector< double >  v( 10 );

                return v;
            };
            
            auto  t  = get_vec();
            auto  t1 = t.copy();
            auto  t2 = t.reference();

            std::cout << t(0) << std::endl;
        }
        
        {
            auto  get_mat = [] ()
            {
                blas::Matrix< double >  M( 10, 10 );

                return M;
            };
            
            auto  T  = get_mat();
            auto  T1 = T.copy();
            auto  T2 = T.reference();

            std::cout << T(0,0) << std::endl;
        }
        
        {
            auto  x = A->row_vector();
            auto  y = A->row_vector();

            for ( size_t  i = 0; i < A->nrows(); ++i )
                x->set_entry( i, i ); // fill( 1.0 );

            y->fill( 0.0 );

            std::cout << A->typestr() << std::endl;
            std::cout << x->norm2() << std::endl;

            A->set_hierarchy_data();

            // burnin
            for ( uint i = 0; i < 10; ++i )
                hpro::mul_vec( hpro::ps_single( 0 ), 1.0, A.get(), x.get(), 1.0, y.get(), hpro::apply_normal );
            
            tic = timer::now();
            
            for ( uint i = 0; i < 100; ++i )
                hpro::mul_vec( hpro::ps_single( 0 ), 1.0, A.get(), x.get(), 1.0, y.get(), hpro::apply_normal ); 

            toc = timer::since( tic );
            
            std::cout << y->norm2() << std::endl;
            std::cout << format_time( toc ) << std::endl;
        }
        
        A = impl::matrix::copy_tiled< double >( *A, ntile );
        
        {
            blas::Vector< double >  ref( A->nrows() );

            for ( size_t  i = 0; i < A->nrows(); ++i )
                ref(i) = i;
            
            vector::tiled_scalarvector< double >  x( A->row_is(), ntile, ref );
            vector::tiled_scalarvector< double >  y( A->row_is(), ntile, ref );

            y.fill( 0.0 );

            std::cout << x.norm2() << std::endl;

            // burnin
            for ( uint i = 0; i < 10; ++i )
                impl::tiled2::mul_vec< double >( 1.0, hpro::apply_normal, *A, x, y );
            
            tic = timer::now();

            for ( uint i = 0; i < 100; ++i )
                impl::tiled2::mul_vec< double >( 1.0, hpro::apply_normal, *A, x, y );

            toc = timer::since( tic );
            
            std::cout << y.norm2() << std::endl;
            std::cout << format_time( toc ) << std::endl;
        }

        auto  C = impl::matrix::copy( *A );

        // std::cout << norm_2( A.get() ) << std::endl;
        // std::cout << norm_2( C.get() ) << std::endl;

        std::vector< double >  runtime;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            impl::tiled2::hodlr::lu< HLIB::real >( C.get(), acc, ntile );

            toc = timer::since( tic );

            std::cout << "    done in " << format_time( toc ) << std::endl;
            
            runtime.push_back( toc.seconds() );

            if ( i < nbench-1 )
                impl::matrix::copy_to( *A, *C );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        // {
        //     auto  T1 = hlr::seq::matrix::copy_nontiled< double >( *C );
        //     auto  T2 = hpro::to_dense( T1.get() );

        //     write_matrix( T2.get(), "A.mat", "A" );
        // }
    
        // std::cout << norm_2( C.get() ) << std::endl;
        
        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem   = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
        
        return;
    }
    else
    {
        std::cout << term::bullet << term::bold << "LU ( Tiled-HODLR " << impl_name << " )" << term::reset << std::endl;
        
        auto  C = impl::matrix::copy( *A );

        std::vector< double >  runtime;
    
        for ( int  i = 0; i < nbench; ++i )
        {
            tic = timer::now();
            
            impl::tiled::hodlr::lu< HLIB::real >( C.get(), acc, ntile );
            
            toc = timer::since( tic );

            std::cout << "    done in " << format_time( toc ) << std::endl;
            
            runtime.push_back( toc.seconds() );

            if ( i < nbench-1 )
                impl::matrix::copy_to( *A, *C );
        }// for
        
        if ( nbench > 1 )
            std::cout << "  runtime = " << format_time( min( runtime ), median( runtime ), max( runtime ) ) << std::endl;

        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem   = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }
}
