//
// Project     : HLR
// Program     : combustion
// Description : compression of datasets from combustion simulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <filesystem>

// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range2d.h>

#include <hlr/utils/io.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/approx/svd.hh>
#include <hlr/approx/rrqr.hh>
#include <hlr/approx/randsvd.hh>
#include <hlr/approx/aca.hh>
#include <hlr/arith/norm.hh>
#include <hlr/utils/tensor.hh>

//#include <hlr/arith/cuda.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

using indexset = hpro::TIndexSet;

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

template < typename value_t >
blas::matrix< value_t >
gen_matrix_log ( const size_t  n )
{
    double  h = 2 * math::pi< value_t >() / value_t(n);
    auto    M = blas::matrix< value_t >( n, n );

    #if 0
    
    ::tbb::parallel_for(
        ::tbb::blocked_range2d< size_t >( 0, n, 1024, 0, n, 1024 ),
        [&,h] ( const auto &  r )
        {
            auto  rofs  = r.rows().begin();
            auto  cofs  = r.cols().begin();
            auto  nrows = r.rows().end() - r.rows().begin();
            auto  ncols = r.cols().end() - r.cols().begin();

            // std::cout << nrows << " x " << ncols << std::endl;

            auto  T = blas::matrix< double >( nrows, ncols );
            
            for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
            {
                const double  x2[2] = { std::sin(j*h), std::cos(j*h) };
        
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    const double  x1[2] = { std::sin(i*h), std::cos(i*h) };
                    const double  dist2 = math::square( x1[0] - x2[0] ) + math::square( x1[1] - x2[1] );

                    if ( dist2 < 1e-12 )
                        T(i-rofs,j-cofs) = 0.0;
                    else
                        T(i-rofs,j-cofs) = math::log( math::sqrt(dist2) );
                }// for
            }// for

            auto  M_ij = blas::matrix< double >( M,
                                                 blas::range( rofs, rofs+nrows-1 ),
                                                 blas::range( cofs, cofs+ncols-1 ) );

            blas::copy( T, M_ij );
        } );
    #else

    #pragma omp parallel for collapse(2) schedule(static,1024)
    for ( size_t  i = 0; i < n; ++i )
    {
        for ( size_t  j = 0; j < n; ++j )
        {
            const double  x1[2] = { std::sin(i*h), std::cos(i*h) };
            const double  x2[2] = { std::sin(j*h), std::cos(j*h) };
            const double  dist2 = math::square( x1[0] - x2[0] ) + math::square( x1[1] - x2[1] );

            if ( dist2 < 1e-12 )
                M(i,j) = 0.0;
            else
                M(i,j) = math::log( math::sqrt(dist2) );
        }// for
    }// for
    
    #endif

    return M;
}

template < typename value_t >
blas::matrix< value_t >
gen_matrix_exp ( const size_t  n )
{
    double  h = 4 * math::pi< value_t >() / value_t(n+1);
    auto    M = blas::matrix< value_t >( n, n );

    for ( size_t  i = 0; i < n; ++i )
    {
        const double  x = i*h - 2.0 * math::pi< value_t >();
        
        for ( size_t  j = 0; j < n; ++j )
        {
            const double  y = j*h - 2.0 * math::pi< value_t >();

            // M(i,j) = std::real( math::exp( 3.0 * std::complex< value_t >( 0, 1 ) * math::sqrt( math::abs( math::square(x) - math::square(y) ) ) ) );
            M(i,j) = std::real( math::exp( 3.0 * std::complex< value_t >( 0, 1 ) * math::sqrt( math::abs( x*x - 4*y ) ) ) );
        }// for
    }// for

    return M;
}

template < typename approx_t,
           typename value_t >
void
do_compress ( blas::matrix< value_t > &  D,
              const double               delta,
              const double               norm_D,
              const size_t               mem_D )
{
    auto    T     = blas::copy( D );
    auto    acc   = local_accuracy( delta );
    auto    apx   = approx_t();
    size_t  csize = 0;

    #if defined(HAS_SZ)

    auto    zconf = ( cmdline::compress == 0 ? std::unique_ptr< sz::sz_config >() : std::make_unique< sz::sz_config >() );
    
    if ( cmdline::compress > 0 )
    {
        *zconf = sz::sz_config_rel( cmdline::compress );
    }// if
    
    #elif defined(HAS_ZFP)
    
    auto    zconf = ( cmdline::compress == 0 ? std::unique_ptr< zfp_config >() : std::make_unique< zfp_config >() );

    if ( cmdline::compress > 0 )
    {
        if ( cmdline::compress > 1 ) *zconf = zfp_config_rate( int( cmdline::compress ), false );
        else                    *zconf = zfp_config_accuracy( cmdline::compress );
    }// if

    #else

    auto  zconf = std::unique_ptr< void >();
    
    #endif
            
    // auto  zconf  = zfp_config_reversible();
    // auto  zconf  = zfp_config_rate( rate, false );
    // auto  zconf  = zfp_config_precision( rate );
    // auto  zconf  = zfp_config_accuracy( rate );
    auto    tic   = timer::now();

    // impl::matrix::compress_replace< value_t, approx_t >( indexset( 0, D.nrows()-1 ),
    //                                                      indexset( 0, D.ncols()-1 ),
    //                                                      T, csize,
    //                                                      acc, apx,
    //                                                      cmdline::ntile,
    //                                                      zconf.get() );
    impl::matrix::compress_ml< value_t, approx_t >( indexset( 0, D.nrows()-1 ),
                                                    indexset( 0, D.ncols()-1 ),
                                                    T, csize,
                                                    4, acc, apx,
                                                    cmdline::ntile,
                                                    zconf.get() );
            
    auto    toc   = timer::since( tic );
        
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( csize ) << std::endl;
    std::cout << "     %full = " << format( "%.4e %%" ) % ( 100.0 * ( double( csize ) / double( mem_D ) )) << std::endl;

    if ( hpro::verbose( 3 ) )
        io::matlab::write( T, "T" );
        
    blas::add( value_t(-1), D, T );

    const auto  error = blas::norm_F( T );
        
    std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;
}

template < typename approx_t,
           typename value_t >
void
do_H ( blas::matrix< value_t > &  D,
       const double               delta,
       const double               norm_D,
       const size_t               mem_D )
{
    auto  acc   = local_accuracy( delta );
    auto  apx   = approx_t();
    
    #if defined(HAS_SZ)

    auto    zconf = ( cmdline::compress == 0 ? std::unique_ptr< sz::sz_config >() : std::make_unique< sz::sz_config >() );
    
    if ( cmdline::compress > 0 )
    {
        *zconf = sz::sz_config_rel( cmdline::compress );
    }// if
    
    #elif defined(HAS_ZFP)
    
    auto    zconf = ( cmdline::compress == 0 ? std::unique_ptr< zfp_config >() : std::make_unique< zfp_config >() );

    if ( cmdline::compress > 0 )
    {
        if ( cmdline::compress > 1 )
            *zconf = zfp_config_rate( int( cmdline::compress ), false );
        else
            *zconf = zfp_config_accuracy( cmdline::compress );
    }// if

    #else

    auto  zconf = std::unique_ptr< void >();
    
    #endif

    auto  tic = timer::now();
    auto  toc = timer::since( tic );
    auto  A   = std::unique_ptr< hpro::TMatrix >();

    for ( int i = 0; i < cmdline::nbench; ++i )
    {
        tic = timer::now();
        // A     = impl::matrix::compress_topdown( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, apx, cmdline::ntile, zconf.get() );
        A   = impl::matrix::compress( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D, acc, apx, cmdline::ntile, zconf.get() );
        toc = timer::since( tic );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;
    }// for
    
    if ( hpro::verbose( 3 ) )
        io::eps::print( *A, "A", "noid,nosize,rank" );
        
    auto  mem_A   = A->byte_size();

    std::cout << "    mem    = " << format_mem( mem_A ) << std::endl;
    std::cout << "     %full = " << format( "%.4e %%" ) % ( 100.0 * ( double( mem_A ) / double( mem_D ) )) << std::endl;

    impl::matrix::uncompress( *A );
    
    auto  DM      = hpro::TDenseMatrix( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D );
    auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), DM );
    auto  error   = hlr::norm::spectral( *diff, true, 1e-4, 20 );
        
    std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;

    // auto  TA   = impl::matrix::copy_nongeneric( *A );
    // auto  B    = impl::matrix::copy( *TA );
    // auto  acc2 = hpro::fixed_prec( 1e-6 );
    
    // for ( int i = 0; i < cmdline::nbench; ++i )
    // {
    //     impl::matrix::copy_to( *TA, *B );
        
    //     tic = timer::now();
        
    //     impl::add< value_t, approx_t >( value_t(1), *TA, *B, acc2, apx );
        
    //     toc = timer::since( tic );
        
    //     std::cout << "    done in  " << format_time( toc ) << std::endl;
    // }// for

    // std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

    // auto  norm_B  = norm::frobenius( *B );
    // auto  diff2   = matrix::sum( value_t(1), *TA, value_t(1), *TA, value_t(-1), *B );
    // auto  error2  = hlr::norm::spectral( *diff2, true, 1e-4, 20 );
        
    // std::cout << "    error  = " << format_error( error2 ) << " / " << format_error( error2 / norm_B ) << std::endl;
}

#if 0
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
compress_cuda ( blas::matrix< value_t > &  D,
                const double               delta,
                const double               norm_D,
                const size_t               mem_D );

template <>
std::unique_ptr< hpro::TMatrix >
compress_cuda< double > ( blas::matrix< double > &  D,
                          const double              delta,
                          const double              norm_D,
                          const size_t              mem_D )
{
    using namespace hlr::matrix;
    
    auto  acc   = local_accuracy( delta );

    #if defined(HAS_SZ)

    auto    zconf = ( cmdline::compress == 0 ? std::unique_ptr< sz::sz_config >() : std::make_unique< sz::sz_config >() );
    
    if ( cmdline::compress > 0 )
    {
        *zconf = sz::sz_config_rel( cmdline::compress );
    }// if
    
    #elif defined(HAS_ZFP)
    
    auto    zconf = ( cmdline::compress == 0 ? std::unique_ptr< zfp_config >() : std::make_unique< zfp_config >() );

    if ( cmdline::compress > 0 )
    {
        if ( cmdline::compress > 1 ) *zconf = zfp_config_rate( int( cmdline::compress ), false );
        else                    *zconf = zfp_config_accuracy( cmdline::compress );
    }// if

    #else

    auto  zconf = std::unique_ptr< void >();
    
    #endif

    //
    // first level, do batched SVD of all tiles
    //

    const size_t  nrows     = D.nrows();
    // const size_t  ncols     = D.ncols();
    size_t        tilesize  = cmdline::ntile;
    size_t        ntiles    = nrows / tilesize;

    #if 0
    
    const size_t  batchsize = ntiles * ntiles;
    
    //
    // copy D as 
    //

    std::vector< double >  A( tilesize*tilesize*batchsize );
    std::vector< double >  U( tilesize*tilesize*batchsize );
    std::vector< double >  V( tilesize*tilesize*batchsize );
    std::vector< double >  S( tilesize*batchsize );

    for ( uint  i = 0; i < ntiles; ++i )
    {
        for ( uint  j = 0; j < ntiles; ++j )
        {
            const size_t  ofs_ij = ( i * ntiles + j ) * tilesize * tilesize;
            auto          D_ij   = blas::copy( D( blas::range( i*tilesize, (i+1)*tilesize - 1 ),
                                                  blas::range( j*tilesize, (j+1)*tilesize - 1 ) ) );

            memcpy( A.data() + ofs_ij, D_ij.data(), tilesize*tilesize*sizeof(double) );
        }// for
    }// for
    
    auto    tic   = timer::now();

    cusolverDnHandle_t  cusolverH = NULL;
    cudaStream_t        stream = NULL;
    gesvdjInfo_t        gesvdj_params = NULL;

    const double             tol        = 1.e-7;
    const int                max_sweeps = 15;
    const cusolverEigMode_t  jobz       = CUSOLVER_EIG_MODE_VECTOR;
    
    // step 1: create cusolver handle, bind a stream
    HLR_CUSOLVER_CHECK( cusolverDnCreate, ( & cusolverH ) );
    HLR_CUDA_CHECK( cudaStreamCreateWithFlags, ( &stream, cudaStreamNonBlocking ) );
    HLR_CUSOLVER_CHECK( cusolverDnSetStream, ( cusolverH, stream ) );

    // step 2: configuration of gesvdj
    HLR_CUSOLVER_CHECK( cusolverDnCreateGesvdjInfo, ( &gesvdj_params ) );

    // default value of tolerance is machine zero
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetTolerance, ( gesvdj_params, tol ) );

    // default value of max. sweeps is 100
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetMaxSweeps, ( gesvdj_params, max_sweeps ) );

    // /* disable sorting */
    // status = cusolverDnXgesvdjSetSortEig(
    //     gesvdj_params,
    //     sort_svd);
    // assert(CUSOLVER_STATUS_SUCCESS == status);

    // step 3: copy A to device
    double *  d_A    = NULL; // lda-by-n-by-batchsize
    double *  d_U    = NULL; // ldu-by-m-by-batchsize
    double *  d_V    = NULL; // ldv-by-n-by-batchsize
    double *  d_S    = NULL; // minmn-by-batchsizee
    int *     d_info = NULL; // batchsize
    int       lwork  = 0;    // size of workspace
    double *  d_work = NULL; // device workspace for gesvdjBatched
    
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_A   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_U   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_V   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_S   , sizeof(double)*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_info, sizeof(int   )*batchsize ) );

    HLR_CUDA_CHECK( cudaMemcpy, ( d_A, A.data(), sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyHostToDevice ) );
    HLR_CUDA_CHECK( cudaDeviceSynchronize, () );
    
    auto  toc = timer::since( tic );

    std::cout << "  time to copy to device : " << format_time( toc ) << std::endl;

    tic = timer::now();
    
    // step 4: query working space of gesvdjBatched
    HLR_CUSOLVER_CHECK( cusolverDnDgesvdjBatched_bufferSize,
                        ( cusolverH,
                          jobz,
                          tilesize, tilesize,
                          d_A, tilesize,
                          d_S,
                          d_U, tilesize,
                          d_V, tilesize,
                          &lwork,
                          gesvdj_params,
                          batchsize ) );

    HLR_CUDA_CHECK( cudaMalloc, ((void**)&d_work, sizeof(double)*lwork) );

    // step 5: compute singular values of A0 and A1
    HLR_CUSOLVER_CHECK( cusolverDnDgesvdjBatched,
                        ( cusolverH,
                          jobz,
                          tilesize, tilesize,
                          d_A, tilesize,
                          d_S,
                          d_U, tilesize,
                          d_V, tilesize,
                          d_work, lwork,
                          d_info,
                          gesvdj_params,
                          batchsize ) );
    HLR_CUDA_CHECK( cudaDeviceSynchronize, () );

    toc = timer::since( tic );

    std::cout << "  time for svd : " << format_time( toc ) << std::endl;

    tic = timer::now();
    
    std::vector< int >  info( batchsize );
    
    HLR_CUDA_CHECK( cudaMemcpy, ( U.data(), d_U, sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyDeviceToHost) );
    HLR_CUDA_CHECK( cudaMemcpy, ( V.data(), d_V, sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyDeviceToHost) );
    HLR_CUDA_CHECK( cudaMemcpy, ( S.data(), d_S, sizeof(double)*tilesize*batchsize, cudaMemcpyDeviceToHost) );
    HLR_CUDA_CHECK( cudaMemcpy, ( info.data(), d_info, sizeof(int) * batchsize, cudaMemcpyDeviceToHost) );

    #else

    //
    // batch limited number of block columns instead of full matrix
    //

    auto    tic   = timer::now();
    auto    toc   = timer::since( tic );
    
    const size_t  nbcols    = std::min< size_t >( 16, ntiles );
    const size_t  batchsize = nbcols * ntiles;
    const size_t  fullsize  = ntiles * ntiles;

    std::vector< double >  A( tilesize*tilesize*batchsize );
    std::vector< double >  U( tilesize*tilesize*fullsize );
    std::vector< double >  V( tilesize*tilesize*fullsize );
    std::vector< double >  S( tilesize*fullsize );

    //
    // pre-allocate device memory
    //
    
    cusolverDnHandle_t  cusolverH = NULL;
    cudaStream_t        stream = NULL;
    gesvdjInfo_t        gesvdj_params = NULL;

    const double             tol        = 1.e-7;
    const int                max_sweeps = 15;
    const cusolverEigMode_t  jobz       = CUSOLVER_EIG_MODE_VECTOR;
    
    // create cusolver handle, bind a stream
    HLR_CUSOLVER_CHECK( cusolverDnCreate, ( & cusolverH ) );
    HLR_CUDA_CHECK( cudaStreamCreateWithFlags, ( &stream, cudaStreamNonBlocking ) );
    HLR_CUSOLVER_CHECK( cusolverDnSetStream, ( cusolverH, stream ) );

    // configuration of gesvdj
    HLR_CUSOLVER_CHECK( cusolverDnCreateGesvdjInfo, ( &gesvdj_params ) );
        
    // default value of tolerance is machine zero
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetTolerance, ( gesvdj_params, tol ) );
        
    // default value of max. sweeps is 100
    HLR_CUSOLVER_CHECK( cusolverDnXgesvdjSetMaxSweeps, ( gesvdj_params, max_sweeps ) );
        
    // step 3: copy A to device
    double *  d_A    = NULL; // lda-by-n-by-batchsize
    double *  d_U    = NULL; // ldu-by-m-by-batchsize
    double *  d_V    = NULL; // ldv-by-n-by-batchsize
    double *  d_S    = NULL; // minmn-by-batchsizee
    int *     d_info = NULL; // batchsize
    int       lwork  = 0;    // size of workspace
    double *  d_work = NULL; // device workspace for gesvdjBatched
        
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_A   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_U   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_V   , sizeof(double)*tilesize*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_S   , sizeof(double)*tilesize*batchsize ) );
    HLR_CUDA_CHECK( cudaMalloc, ( (void**) & d_info, sizeof(int   )*batchsize ) );
    
    // step 4: query working space of gesvdjBatched
    HLR_CUSOLVER_CHECK( cusolverDnDgesvdjBatched_bufferSize,
                        ( cusolverH,
                          jobz,
                          tilesize, tilesize,
                          d_A, tilesize,
                          d_S,
                          d_U, tilesize,
                          d_V, tilesize,
                          &lwork,
                          gesvdj_params,
                          batchsize ) );
    
    HLR_CUDA_CHECK( cudaMalloc, ((void**)&d_work, sizeof(double)*lwork) );
    
    std::vector< int >  info( batchsize );
    
    //
    // loop over columns
    //

    size_t  bcol = 0;
    double  tcpy = 0;
    double  tsvd = 0;
    double  th2d = 0;
    double  td2h = 0;
    const size_t  step_A = tilesize * tilesize;
    size_t        ofs_D  = 0;

    while ( bcol < ntiles )
    {
        tic   = timer::now();

        for ( uint  bc = 0; bc < nbcols; ++bc )
        {
            for ( uint  j = 0; j < tilesize; ++j )
            {
                size_t  ofs_A = j * tilesize + ( bc * tilesize * tilesize * ntiles );
            
                for ( uint  i = 0; i < ntiles; ++i )
                {
                    memcpy( A.data() + ofs_A, D.data() + ofs_D, tilesize*sizeof(double) );
                    ofs_A += step_A;
                    ofs_D += tilesize;
                }// for
            }// for
        }// for

        toc   = timer::since( tic );
        tcpy += toc.seconds();
        std::cout << "  time to prepare copy    : " << format_time( toc ) << std::endl;

        // {
        //     auto  T = blas::matrix< double >( tilesize, ntiles * nbcols * tilesize );

        //     memcpy( T.data(), A.data(), sizeof(double) * A.size() );
        //     io::matlab::write( T, "A" );
        // }

        //
        // copy to device
        //
        
        tic   = timer::now();

        HLR_CUDA_CHECK( cudaMemcpy, ( d_A, A.data(), sizeof(double)*A.size(), cudaMemcpyHostToDevice ) );
        HLR_CUDA_CHECK( cudaDeviceSynchronize, () );
        
        toc   = timer::since( tic );
        th2d += toc.seconds();

        std::cout << "  time to copy to device   : " << format_time( toc ) << std::endl;

        tic = timer::now();
    
        // compute singular values of A0 and A1
        HLR_CUSOLVER_CHECK( cusolverDnDgesvdjBatched,
                            ( cusolverH,
                              jobz,
                              tilesize, tilesize,
                              d_A, tilesize,
                              d_S,
                              d_U, tilesize,
                              d_V, tilesize,
                              d_work, lwork,
                              d_info,
                              gesvdj_params,
                              batchsize ) );
        HLR_CUDA_CHECK( cudaDeviceSynchronize, () );
        
        toc   = timer::since( tic );
        tsvd += toc.seconds();
        
        std::cout << "  time for svd             : " << format_time( toc ) << std::endl;

        //
        // copy back to host
        //

        tic = timer::now();
        
        const size_t  ofs_UV = tilesize*tilesize * bcol * ntiles;
        const size_t  ofs_S  = tilesize          * bcol * ntiles;
            
        // std::cout << ofs_UV << ' ' << ofs_S << std::endl;

        HLR_CUDA_CHECK( cudaMemcpy, ( U.data() + ofs_UV, d_U, sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyDeviceToHost) );
        HLR_CUDA_CHECK( cudaMemcpy, ( V.data() + ofs_UV, d_V, sizeof(double)*tilesize*tilesize*batchsize, cudaMemcpyDeviceToHost) );
        HLR_CUDA_CHECK( cudaMemcpy, ( S.data() + ofs_S,  d_S, sizeof(double)*tilesize*batchsize, cudaMemcpyDeviceToHost) );
        HLR_CUDA_CHECK( cudaMemcpy, ( info.data(), d_info, sizeof(int) * batchsize, cudaMemcpyDeviceToHost) );

        // {
        //     auto  T = blas::matrix< double >( tilesize, fullsize * tilesize );

        //     memcpy( T.data(), U.data(), U.size() * sizeof(double) );
        //     io::matlab::write( T, "U" );

        //     memcpy( T.data(), V.data(), V.size() * sizeof(double) );
        //     io::matlab::write( T, "V" );
        // }

        toc   = timer::since( tic );
        td2h += toc.seconds();
        
        std::cout << "  time to copy from device : " << format_time( toc ) << std::endl;
        
        //
        // proceed
        //

        bcol += nbcols;
    }// while
   
    std::cout << "  time to prepare copy     : " << format_time( tcpy ) << std::endl;
    std::cout << "  time to copy to device   : " << format_time( th2d ) << std::endl;
    std::cout << "  time for svd             : " << format_time( tsvd ) << std::endl;
    std::cout << "  time to copy from device : " << format_time( td2h ) << std::endl;

    #endif
    
    //
    // construct lowrank/dense matrix blocks
    //

    tic = timer::now();
    
    auto  blocks   = tensor2< std::unique_ptr< hpro::TMatrix > >( ntiles, ntiles );
    auto  acc_tile = acc( indexset( 0, tilesize-1 ), indexset( 0, tilesize-1 ) );
    auto  r        = ::tbb::blocked_range2d< size_t >( 0, ntiles, 0, ntiles );

    ::tbb::parallel_for( r,
        [&,tilesize] ( const auto &  r )
        {
            for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    auto  rowis = indexset( i * tilesize, (i+1) * tilesize - 1 );
                    auto  colis = indexset( j * tilesize, (j+1) * tilesize - 1 );

                    //
                    // determine approximation rank
                    //
            
                    const size_t  ofs_S = ( j * ntiles + i ) * tilesize;
                    auto          S_ij  = S.data() + ofs_S;
                    auto          sv    = blas::vector< double >( tilesize );

                    for ( size_t  l = 0; l < tilesize; ++l )
                        sv(l) = S_ij[l];
                    
                    auto  k = acc_tile.trunc_rank( sv );
                    
                    //
                    // decode if lowrank or dense and build matrix block for tile
                    //
                    
                    const size_t  ofs_A = ( j * ntiles + i ) * tilesize * tilesize;
                    
                    if ( k < tilesize/2 )
                    {
                        // std::cout << i << ", " << j << " = " << k << std::endl;
                        
                        auto  U_ij = blas::matrix< double >( tilesize, k );
                        auto  V_ij = blas::matrix< double >( tilesize, k );
                        
                        for ( uint  l = 0; l < k; ++l )
                        {
                            for ( uint  ii = 0; ii < tilesize; ++ii )
                                U_ij( ii, l ) = sv(l) * U.data()[ofs_A + (l*tilesize) + ii];
                            for ( uint  ii = 0; ii < tilesize; ++ii )
                                V_ij( ii, l ) = V.data()[ofs_A + (l*tilesize) + ii];
                        }// for
                        
                        blocks( i, j ) = std::make_unique< lrmatrix >( rowis, colis, std::move( U_ij ), std::move( V_ij ) );
                    }// if
                    else
                    {
                        // std::cout << i << ", " << j << " = dense" << std::endl;
                        
                        auto  D_ij  = blas::copy( D( blas::range( i*tilesize, (i+1)*tilesize - 1 ),
                                                     blas::range( j*tilesize, (j+1)*tilesize - 1 ) ) );

                        blocks( i, j ) = std::make_unique< dense_matrix >( rowis, colis, std::move( blas::copy( D_ij ) ) );
                    }// else
                }// for
            }// for
        } );

    toc = timer::since( tic );

    std::cout << "  time for bottom layer : " << format_time( toc ) << std::endl;
    
    //
    // iterate upwards till the roof
    //

    tic = timer::now();
    
    auto  approx = hlr::approx::SVD< double >();

    while ( true )
    {
        const auto  tilesize_up = tilesize * 2;
        const auto  ntiles_up   = ntiles / 2;

        //
        // join 2x2 small blocks into a larger block
        //
        //   mapping:  larger (i,j) -> small (2*i,  2*j) (2*i,   2*j+1)
        //                                   (2*i+1,2*j) (2*i+1, 2*j+1)
        //

        auto  blocks_up = tensor2< std::unique_ptr< hpro::TMatrix > >( ntiles_up, ntiles_up );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< size_t >( 0, ntiles_up,
                                              0, ntiles_up ),
            [&,tilesize_up] ( const auto &  r )
            {
                for ( auto  ii = r.rows().begin(); ii != r.rows().end(); ++ii )
                {
                    for ( auto  jj = r.cols().begin(); jj != r.cols().end(); ++jj )
                    {
                        auto  rowis       = indexset( ii * tilesize_up, (ii+1) * tilesize_up - 1 );
                        auto  colis       = indexset( jj * tilesize_up, (jj+1) * tilesize_up - 1 );
                
                        auto  ofs_i       = 2*ii;
                        auto  ofs_j       = 2*jj;
                        auto  sub_blocks  = tensor2< std::unique_ptr< hpro::TMatrix > >( 2, 2 );
                        bool  all_lowrank = true;
                        bool  all_dense   = true;

                        for ( uint  i = 0; i < 2; ++i )
                        {
                            for ( uint  j = 0; j < 2; ++j )
                            {
                                sub_blocks(i,j) = std::move( blocks( ofs_i + i, ofs_j + j ) );

                                HLR_ASSERT( ! is_null( sub_blocks(i,j).get() ) );

                                if ( ! is_generic_lowrank( *sub_blocks(i,j) ) )
                                    all_lowrank = false;

                                if ( ! is_generic_dense( *sub_blocks(i,j) ) )
                                    all_dense = false;
                            }// for
                        }// for

                        if ( all_lowrank )
                        {
                            //
                            // construct larger lowrank matrix out of smaller sub blocks
                            //

                            // compute initial total rank
                            uint  rank_sum = 0;

                            for ( uint  i = 0; i < 2; ++i )
                                for ( uint  j = 0; j < 2; ++j )
                                    rank_sum += ptrcast( sub_blocks(i,j).get(), lrmatrix )->rank();

                            // copy sub block data into global structure
                            auto    U    = blas::matrix< double >( rowis.size(), rank_sum );
                            auto    V    = blas::matrix< double >( colis.size(), rank_sum );
                            auto    pos  = 0; // pointer to next free space in U/V
                            size_t  smem = 0; // holds memory of sub blocks
            
                            for ( uint  i = 0; i < 2; ++i )
                            {
                                for ( uint  j = 0; j < 2; ++j )
                                {
                                    auto  Rij   = ptrcast( sub_blocks(i,j).get(), lrmatrix );
                                    auto  Uij   = Rij->U< double >();
                                    auto  Vij   = Rij->V< double >();
                                    auto  U_sub = U( Rij->row_is() - rowis.first(), blas::range( pos, pos + Uij.ncols() ) );
                                    auto  V_sub = V( Rij->col_is() - colis.first(), blas::range( pos, pos + Uij.ncols() ) );

                                    blas::copy( Uij, U_sub );
                                    blas::copy( Vij, V_sub );

                                    pos  += Uij.ncols();
                                    smem += Uij.byte_size() + Vij.byte_size();
                                }// for
                            }// for

                            //
                            // try to approximate again in lowrank format and use
                            // approximation if it uses less memory 
                            //
            
                            auto  [ W, X ] = approx( U, V, acc( rowis, colis ) );

                            if ( W.byte_size() + X.byte_size() < smem )
                            {
                                blocks_up( ii, jj ) = std::make_unique< lrmatrix >( rowis, colis, std::move( W ), std::move( X ) );
                            }// if
                        }// if
                        else if ( all_dense )
                        {
                            //
                            // always join dense blocks
                            //
        
                            auto  D = blas::matrix< double >( rowis.size(), colis.size() );
                        
                            for ( uint  i = 0; i < 2; ++i )
                            {
                                for ( uint  j = 0; j < 2; ++j )
                                {
                                    auto  sub_ij = ptrcast( sub_blocks(i,j).get(), dense_matrix );
                                    auto  sub_D = D( sub_ij->row_is() - rowis.first(),
                                                     sub_ij->col_is() - colis.first() );

                                    blas::copy( std::get< blas::matrix< double > >( sub_ij->matrix() ), sub_D );
                                }// for
                            }// for
                    
                            blocks_up( ii, jj ) = std::make_unique< dense_matrix >( rowis, colis, std::move( D ) );
                        }// if

                        if ( is_null( blocks_up( ii, jj ) ) )
                        {
                            //
                            // either not all low-rank or memory gets larger: construct block matrix
                            //

                            auto  B = std::make_unique< hpro::TBlockMatrix >( rowis, colis );

                            B->set_block_struct( 2, 2 );
        
                            for ( uint  i = 0; i < 2; ++i )
                            {
                                for ( uint  j = 0; j < 2; ++j )
                                {
                                    if ( ! is_null( zconf ) )
                                    {
                                        if ( is_generic_lowrank( *sub_blocks(i,j) ) )
                                            ptrcast( sub_blocks(i,j).get(), lrmatrix )->compress( *zconf );
                
                                        if ( is_generic_dense( *sub_blocks(i,j) ) )
                                            ptrcast( sub_blocks(i,j).get(), dense_matrix )->compress( *zconf );
                                    }// if
                
                                    B->set_block( i, j, sub_blocks(i,j).release() );
                                }// for
                            }// for

                            blocks_up( ii, jj ) = std::move( B );
                        }// if
                    }// for
                }// for
            } );

        blocks = std::move( blocks_up );

        if ( ntiles_up == 1 )
            break;

        tilesize = tilesize_up;
        ntiles   = ntiles_up;
    }// while

    toc = timer::since( tic );

    std::cout << "  time for upper layers : " << format_time( toc ) << std::endl;
    
    //
    // return single, top-level matrix in "blocks"
    //
    
    return std::move( blocks( 0, 0 ) );
    
    //
    // return matrix
    //

    // auto  B = std::make_unique< hpro::TBlockMatrix >( indexset( 0, nrows-1 ),
    //                                                   indexset( 0, ncols-1 ) );

    // B->set_block_struct( ntiles, ntiles );

    // for ( uint  i = 0; i < ntiles; ++i )
    // {
    //     for ( uint  j = 0; j < ntiles; ++j )
    //     {
    //         B->set_block( i, j, blocks(i,j).release() );
    //     }// for
    // }// for

    // return B;
}
#endif

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    #if defined(HAS_SZ)

    sz_params  params;

    params.quantization_intervals = 0;
    params.max_quant_intervals    = 65536;
    params.szMode                 = SZ_BEST_COMPRESSION;
    params.gzipMode               = Z_BEST_COMPRESSION;

    SZ_Init_Params( & params );

    #endif
    
    //
    // read dataset
    //
    
    auto  D = blas::matrix< value_t >();

    if ( cmdline::matrixfile != "" )
    {
        std::cout << "  " << term::bullet << term::bold << "reading data (" << cmdline::matrixfile << ")" << term::reset << std::endl;
        
        if ( std::filesystem::path( cmdline::matrixfile ).extension() == ".mat" ) 
            D = io::matlab::read< value_t >( cmdline::matrixfile );
        else if ( std::filesystem::path( cmdline::matrixfile ).extension() == ".h5" ) 
            D = io::hdf5::read< value_t >( cmdline::matrixfile );
        else
            HLR_ERROR( "unsupported file type" );
    }// if
    else
    {
        std::cout << "  " << term::bullet << term::bold << "generating matrix (" << cmdline::appl << ")" << term::reset << std::endl;

        if      ( cmdline::appl == "log" ) D = std::move( gen_matrix_log< value_t >( cmdline::n ) );
        else if ( cmdline::appl == "exp" ) D = std::move( gen_matrix_exp< value_t >( cmdline::n ) );
        else
            HLR_ERROR( "unknown matrix : " + cmdline::appl );
        
        if ( hpro::verbose( 3 ) )
            io::matlab::write( D, "D" );
    }// else
        
    auto  mem_D  = D.byte_size();
    auto  norm_D = blas::norm_F( D );

    std::cout << "    size   = " << D.nrows() << " × " << D.ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( mem_D ) << std::endl;
    std::cout << "    |D|    = " << format_norm( norm_D ) << std::endl;

    //
    // compress data and replace content
    //

    auto  delta = norm_D * hlr::cmdline::eps / D.nrows();
    
    std::cout << "  " << term::bullet << term::bold << "compression, "
              << cmdline::approx << ", ε = " << cmdline::eps << ", δ = " << delta << ", "
              << "SZ/ZFP = " << cmdline::compress
              << term::reset << std::endl;

    #if 0
    if ( false )
    {
        // blas::cuda::init();
    
        auto  tic   = timer::now();
        auto  A     = compress_cuda( D, delta, norm_D, mem_D );
        auto  toc   = timer::since( tic );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;
        
        if ( hpro::verbose( 3 ) )
        {
            io::eps::print( *A, "A", "noid,nosize,rank" );

            auto  DA = hlr::matrix::convert_to_dense< value_t >( *A );

            io::matlab::write( *DA, "T" );
        }// if
        
        auto  mem_A   = A->byte_size();
        
        std::cout << "    mem    = " << format_mem( mem_A ) << std::endl;
        std::cout << "     %full = " << format( "%.4e %%" ) % ( 100.0 * ( double( mem_A ) / double( mem_D ) )) << std::endl;

        auto  DM      = hpro::TDenseMatrix( indexset( 0, D.nrows()-1 ), indexset( 0, D.ncols()-1 ), D );
        auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), DM );
        auto  error   = hlr::norm::spectral( *diff, true, 1e-4, 20 );
        
        std::cout << "    error  = " << format_error( error ) << " / " << format_error( error / norm_D ) << std::endl;
    
        return;
    }// if
    #endif
    
    if ( false )
    {
        //
        // compress inplace (replace data)
        //
        
        if ( cmdline::approx == "svd" ||
             cmdline::approx == "default" ) do_compress< hlr::approx::SVD< value_t > >(     D, delta, norm_D, mem_D );
        if ( cmdline::approx == "rrqr"    ) do_compress< hlr::approx::RRQR< value_t > >(    D, delta, norm_D, mem_D );
        if ( cmdline::approx == "randsvd" ) do_compress< hlr::approx::RandSVD< value_t > >( D, delta, norm_D, mem_D );
        if ( cmdline::approx == "aca"     ) do_compress< hlr::approx::ACA< value_t > >(     D, delta, norm_D, mem_D );
    }// if
    else
    {
        //
        // compress to H-matrix
        //

        if ( cmdline::approx == "svd" ||
             cmdline::approx == "default" ) do_H< hlr::approx::SVD< value_t > >(     D, delta, norm_D, mem_D );
        if ( cmdline::approx == "rrqr"    ) do_H< hlr::approx::RRQR< value_t > >(    D, delta, norm_D, mem_D );
        if ( cmdline::approx == "randsvd" ) do_H< hlr::approx::RandSVD< value_t > >( D, delta, norm_D, mem_D );
        if ( cmdline::approx == "aca"     ) do_H< hlr::approx::ACA< value_t > >(     D, delta, norm_D, mem_D );
    }// else

    #if defined(HAS_SZ)

    SZ_Finalize();

    #endif
}
    
