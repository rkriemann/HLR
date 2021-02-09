//
// Project     : HLR
// File        : cuda.cc
// Description : program for testing CUDA based arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/io/TClusterBasisVis.hh>

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>

#include "hlr/tf/matrix.hh"
//#include "hlr/tf/approx.hh"
#include "hlr/tf/dag.hh"

#include <hlr/approx/svd.hh>
#include "hlr/arith/cuda.hh"

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

void
cuda_hostfn ( void *  data )
{
    std::cout << "cuda_hostfn" << std::endl;
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    using real_t  = typename hpro::real_type< value_t >::type_t;

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
    
    A = hlr::tf::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    hlr::tf::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    //
    // for tests below
    //
    
    // assuming HODLR adm., so use first off-diagonal block as low-rank matrix
    auto  A01 = ptrcast( ptrcast( A.get(), hpro::TBlockMatrix )->block( 0, 1 ), hpro::TRkMatrix );
    
    //////////////////////////////////////////////////////////////////
    //
    // compute QR
    //
    //////////////////////////////////////////////////////////////////

    {
        auto  U = blas::copy( hpro::blas_mat_A< value_t >( *A01 ) );
        
        //
        // using default blas
        //

        {
            auto                     U1 = blas::copy( U );
            blas::matrix< value_t >  R1( U.ncols(), U.ncols() );
        
            blas::qr2( U1, R1 );
        
            hpro::DBG::write( U1, "U1.mat", "U1" );
            hpro::DBG::write( R1, "R1.mat", "R1" );
        }

        {
            auto                     U1 = blas::copy( U );
            blas::matrix< value_t >  R1( U.ncols(), U.ncols() );
            std::vector< value_t >   tau( U.ncols() );
        
            blas::qr( U1, R1 );
        
            hpro::DBG::write( U1, "U2.mat", "U2" );
            hpro::DBG::write( R1, "R2.mat", "R2" );
        }
    }
    
    //////////////////////////////////////////////////////////////////
    //
    // compute QR
    //
    //////////////////////////////////////////////////////////////////

    blas::cuda::init();
    
    if ( false )
    {
        auto  U   = blas::copy( hpro::blas_mat_A< value_t >( *A01 ) );
        // auto  V   = blas::copy( hpro::blas_mat_B< value_t >( *A01 ) );
        
        //
        // using default blas
        //

        {
            auto                     U1 = blas::copy( U );
            blas::matrix< value_t >  R1( U.ncols(), U.ncols() );
        
            blas::qr( U1, R1 );
        
            hpro::DBG::write( U1, "U1.mat", "U1" );
            hpro::DBG::write( R1, "R1.mat", "R1" );
        }
    
        //
        // using CUDA
        //

        {
            auto                     U2 = blas::copy( U );
            blas::matrix< value_t >  R2( U.ncols(), U.ncols() );
    
            blas::cuda::init();
            blas::cuda::qr( blas::cuda::default_handle, U2, R2 );

            hpro::DBG::write( U2, "U2.mat", "U2" );
            hpro::DBG::write( R2, "R2.mat", "R2" );
        }
    }

    //////////////////////////////////////////////////////////////////
    //
    // compute SVD
    //
    //////////////////////////////////////////////////////////////////

    if ( false )
    {
        auto  U = blas::copy( hpro::blas_mat_A< value_t >( *A01 ) );
        auto  V = blas::copy( hpro::blas_mat_B< value_t >( *A01 ) );
        auto  M = blas::prod( value_t(1), U, blas::adjoint( V ) );
        
        //
        // using default blas
        //

        {
            auto  M1 = blas::copy( M );
            auto  S1 = blas::vector< real_t >( M1.nrows() );
            auto  V1 = blas::matrix< value_t >( M1.ncols(), M1.nrows() );
        
            blas::svd( M1, S1, V1 );
        
            hpro::DBG::write( M1, "U1.mat", "U1" );
            hpro::DBG::write( S1, "S1.mat", "S1" );
            hpro::DBG::write( V1, "V1.mat", "V1" );
        }
    
        //
        // using CUDA
        //

        {
            auto  M2 = blas::copy( M );
            auto  S2 = blas::vector< real_t >( M2.nrows() );
            auto  V2 = blas::matrix< value_t >( M2.ncols(), M2.nrows() );

            hpro::DBG::write( M, "M.mat", "M" );
        
            blas::cuda::svd( blas::cuda::default_handle, M2, S2, V2 );
        
            hpro::DBG::write( M2, "U2.mat", "U2" );
            hpro::DBG::write( S2, "S2.mat", "S2" );
            hpro::DBG::write( V2, "V2.mat", "V2" );
        }
    }

    //////////////////////////////////////////////////////////////////
    //
    // truncation
    //
    //////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "truncation" << term::reset << std::endl;
    
    if ( true )
    {
        auto  U = blas::copy( hpro::blas_mat_A< value_t >( *A01 ) );
        auto  V = blas::copy( hpro::blas_mat_B< value_t >( *A01 ) );
        
        hpro::DBG::write( U, "U.mat", "U" );
        hpro::DBG::write( V, "V.mat", "V" );
        
        //
        // using default blas
        //

        {
            auto  U1 = blas::copy( U );
            auto  V1 = blas::copy( V );
        
            auto [ U2, V2 ] = approx::svd( U1, V1, hpro::fixed_rank( 2 ) );
        
            hpro::DBG::write( U2, "U1.mat", "U1" );
            hpro::DBG::write( V2, "V1.mat", "V1" );

            auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

            blas::prod( value_t(-1), U2, blas::adjoint(V2), value_t(1), M );

            std::cout << "error = " << format_error( blas::norm_F( M ) ) << std::endl;
        }
    
        //
        // using CUDA
        //

        {
            auto  U1 = blas::copy( U );
            auto  V1 = blas::copy( V );
        
            auto [ U2, V2 ] = blas::cuda::svd( blas::cuda::default_handle, U1, V1, hpro::fixed_rank( 2 ) );
        
            hpro::DBG::write( U2, "U2.mat", "U2" );
            hpro::DBG::write( V2, "V2.mat", "V2" );

            auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

            blas::prod( value_t(-1), U2, blas::adjoint(V2), value_t(1), M );

            std::cout << "error = " << format_error( blas::norm_F( M ) ) << std::endl;
        }
    }

    //////////////////////////////////////////////////////////////////
    //
    // asynchronous truncation
    //
    //////////////////////////////////////////////////////////////////

    std::cout << term::bullet << term::bold << "async. truncation" << term::reset << std::endl;
    
    if ( true )
    {
        using  cuda_t      = typename blas::cuda::cuda_type< value_t >::type_t;
        using  cuda_real_t = typename blas::cuda::real_type< cuda_t >::type_t;

        auto      U     = blas::copy( hpro::blas_mat_A< value_t >( *A01 ) );
        auto      V     = blas::copy( hpro::blas_mat_B< value_t >( *A01 ) );

        blas::cuda::handle  handle;

        HLR_CUDA_CHECK( cudaStreamCreate, ( & handle.stream ) );
    
        HLR_CUBLAS_CHECK( cublasCreate,    ( & handle.blas ) );
        HLR_CUBLAS_CHECK( cublasSetStream, (   handle.blas, handle.stream ) );

        HLR_CUSOLVER_CHECK( cusolverDnCreate,    ( & handle.solver ) );
        HLR_CUSOLVER_CHECK( cusolverDnSetStream, (   handle.solver, handle.stream ) );
        
        // cuda_t *  dev_U = nullptr;
        // cuda_t *  dev_V = nullptr;

        // cudaMalloc( & dev_U, U.nrows() * U.ncols() * sizeof(cuda_t) );
        // cudaMalloc( & dev_V, U.nrows() * V.ncols() * sizeof(cuda_t) );

        // HLR_CUDA_CHECK( cudaMemcpyAsync, ( dev_U, U.data(), sizeof(value_t) * U.nrows() * U.ncols(),
        //                                    cudaMemcpyHostToDevice, handle.stream ) );
        // HLR_CUDA_CHECK( cudaMemcpyAsync, ( dev_V, V.data(), sizeof(value_t) * V.nrows() * V.ncols(),
        //                                    cudaMemcpyHostToDevice, handle.stream ) );

        // int  rank = 0;

        
        // blas::cuda::svd_dev2<<<1,1>>>( handle, U.nrows(), V.nrows(), U.ncols(), dev_U, dev_V, acc );

        // std::cout << "svd_dev" << std::endl;

        // HLR_CUDA_CHECK( cudaMemcpyAsync, ( U.data(), dev_U, sizeof(value_t) * U.nrows() * U.ncols(),
        //                                    cudaMemcpyDeviceToHost, handle.stream ) );
        // HLR_CUDA_CHECK( cudaMemcpyAsync, ( V.data(), dev_V, sizeof(value_t) * V.nrows() * V.ncols(),
        //                                    cudaMemcpyDeviceToHost, handle.stream ) );

        // HLR_CUDA_CHECK( cudaLaunchHostFunc, ( handle.stream, cuda_hostfn, nullptr ) );
        // std::cout << "launch hostfn" << std::endl;
        
        // HLR_CUDA_CHECK( cudaStreamSynchronize, ( handle.stream ) );
        // std::cout << "synchronise" << std::endl;

        const int  n = U.nrows();
        
        auto  A2 = blas::random< value_t >( n, n );
        auto  B2 = blas::random< value_t >( n, n );
        auto  C2 = blas::random< value_t >( n, n );

        // first, copy to pinned memory
        value_t *  host_A2 = nullptr;
        value_t *  host_B2 = nullptr;
        value_t *  host_C2 = nullptr;

        cudaMallocHost( & host_A2, n * n * sizeof(value_t) );
        cudaMallocHost( & host_B2, n * n * sizeof(value_t) );
        cudaMallocHost( & host_C2, n * n * sizeof(value_t) );

        memcpy( host_A2, A2.data(), n * n * sizeof(value_t) ); 
        memcpy( host_B2, B2.data(), n * n * sizeof(value_t) ); 
        memcpy( host_C2, C2.data(), n * n * sizeof(value_t) ); 
        
        cuda_t *  dev_A2   = nullptr;
        cuda_t *  dev_B2   = nullptr;
        cuda_t *  dev_C2   = nullptr;
        cuda_t *  dev_tau  = nullptr;
        cuda_t *  dev_work = nullptr;
        int *     dev_info = nullptr;
        cuda_real_t *  dev_S = nullptr;

        // auto  one   = blas::cuda::make_constant< cuda_t >( 1 );
        // auto  zero  = blas::cuda::make_constant< cuda_t >( 0 );

        gesvdjInfo_t  gesvdj_params;

        cusolverDnCreateGesvdjInfo( & gesvdj_params );
        cusolverDnXgesvdjSetTolerance( gesvdj_params, 1e-7 );
        cusolverDnXgesvdjSetMaxSweeps( gesvdj_params, 15 );

        cudaMalloc( & dev_A2, n * n * sizeof(cuda_t) );
        cudaMalloc( & dev_B2, n * n * sizeof(cuda_t) );
        cudaMalloc( & dev_C2, n * n * sizeof(cuda_t) );
        cudaMalloc( & dev_tau, n * sizeof(cuda_t) );
        cudaMalloc( & dev_S, n * sizeof(cuda_real_t) );

        // auto  lwork = blas::cuda::qr_worksize< cuda_t >( handle, n, n, dev_A2, dev_tau );
        auto  lwork = blas::cuda::svd_worksize< cuda_t >( handle, n, n, dev_A2, dev_B2, dev_S, dev_C2, gesvdj_params );
            
        cudaMalloc( & dev_work, lwork * sizeof(cuda_t) );
        cudaMalloc( & dev_info, sizeof(int) );

        tic = timer::now();
        HLR_CUDA_CHECK( cudaMemcpyAsync, ( dev_A2, host_A2, sizeof(value_t) * n * n, cudaMemcpyHostToDevice, handle.stream ) );
        HLR_CUDA_CHECK( cudaMemcpyAsync, ( dev_B2, host_B2, sizeof(value_t) * n * n, cudaMemcpyHostToDevice, handle.stream ) );
        HLR_CUDA_CHECK( cudaMemcpyAsync, ( dev_C2, host_C2, sizeof(value_t) * n * n, cudaMemcpyHostToDevice, handle.stream ) );

        for ( int  i = 0; i < 5; ++i )
        {
            // blas::cuda::prod( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, one, dev_A2, n, dev_B2, n, zero, dev_C2, n );
            // blas::cuda::qr_dev( handle, n, n, dev_A2, dev_B2, dev_tau, lwork, dev_work, dev_info );
            blas::cuda::svd_dev( handle, n, n, dev_A2, dev_B2, dev_S, dev_C2, gesvdj_params, lwork, dev_work, dev_info );
        }// for

        HLR_CUDA_CHECK( cudaMemcpyAsync, ( host_A2, dev_A2, sizeof(value_t) * n * n, cudaMemcpyDeviceToHost, handle.stream ) );
        HLR_CUDA_CHECK( cudaMemcpyAsync, ( host_B2, dev_B2, sizeof(value_t) * n * n, cudaMemcpyDeviceToHost, handle.stream ) );
        HLR_CUDA_CHECK( cudaMemcpyAsync, ( host_C2, dev_C2, sizeof(value_t) * n * n, cudaMemcpyDeviceToHost, handle.stream ) );
        
        HLR_CUDA_CHECK( cudaLaunchHostFunc, ( handle.stream, cuda_hostfn, nullptr ) );
        toc = timer::since( tic );
        std::cout << "time to setup stream = " << format_time( toc ) << std::endl;
        
        HLR_CUDA_CHECK( cudaStreamSynchronize, ( handle.stream ) );
        toc = timer::since( tic );
        std::cout << "time to run          = " << format_time( toc ) << std::endl;

        // {
        //     auto  U1 = blas::copy( U );
        //     auto  V1 = blas::copy( V );
        
        //     auto [ U2, V2 ] = blas::cuda::svd( blas::cuda::default_handle, U1, V1, hpro::fixed_rank( 2 ) );
        
        //     hpro::DBG::write( U2, "U2.mat", "U2" );
        //     hpro::DBG::write( V2, "V2.mat", "V2" );

        //     auto  M = blas::prod( value_t(1), U, blas::adjoint(V) );

        //     blas::prod( value_t(-1), U2, blas::adjoint(V2), value_t(1), M );

        //     std::cout << "error = " << format_error( blas::norm_F( M ) ) << std::endl;
        // }
    }
}

template < typename problem_t >
void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
