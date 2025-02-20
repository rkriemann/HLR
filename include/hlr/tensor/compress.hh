#ifndef __HLR_TENSOR_COMPRESS_HH
#define __HLR_TENSOR_COMPRESS_HH
//
// Project     : HLR
// Module      : tensor/compress
// Description : tensor (data) compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/tensor/dense_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/structured_tensor.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/arith/hosvd.hh>

namespace hlr { namespace tensor {

//
// apply compression to compressible objects
//
template < typename value_t >
void
compress ( base_tensor3< value_t > &  T,
           const accuracy &           acc )
{
    if ( is_structured( T ) )
    {
        auto  B = ptrcast( &T, structured_tensor3< value_t > );

        for ( uint  l = 0; l < B->nblocks(2); ++l )
        {
            for ( uint  j = 0; j < B->nblocks(1); ++j )
            {
                for ( uint  i = 0; i < B->nblocks(0); ++i )
                {
                    if ( is_null( B->block( i, j, l ) ) )
                        continue;
                
                    compress( *B->block( i, j, l ), acc );
                }// for
            }// for
        }// for
    }// if
    else if ( compress::is_compressible( T ) )
    {
        dynamic_cast< compress::compressible * >( &T )->compress( acc );
    }// if
}

//
// decompress compressible objects
//
template < typename value_t >
void
decompress ( base_tensor3< value_t > &  T )
{
    if ( is_structured( T ) )
    {
        auto  B = ptrcast( &T, structured_tensor3< value_t > );

        for ( uint  l = 0; l < B->nblocks(2); ++l )
        {
            for ( uint  j = 0; j < B->nblocks(1); ++j )
            {
                for ( uint  i = 0; i < B->nblocks(0); ++i )
                {
                    if ( is_null( B->block( i, j, l ) ) )
                        continue;
                    
                    decompress( *B->block( i, j, l ) );
                }// for
            }// for
        }// for
    }// if
    else if ( compress::is_compressible( T ) )
    {
        dynamic_cast< compress::compressible * >( &T )->decompress();
    }// if
}

//
// compress/uncompress given tensor in-place and return compressed size
//
template < typename value_t >
size_t
compress_tucker ( blas::tensor3< value_t > &  D,
                  const size_t                ntile,
                  const double &              eps,
                  const bool                  fpcompress )
{
    HLR_ASSERT(( D.size(0) == D.size(1) ) &&
               ( D.size(0) == D.size(2) )); // assuming equal size in all dimensions

    const auto  N    = D.size(0);
    const auto  Nc   = (N % ntile == 0 ? N / ntile : ( N / ntile ) + 1 );
    auto        apx  = approx::SVD< value_t >();
    size_t      zmem = 0;

    const auto  norm_D = blas::norm_F( D );
    const auto  tol    = eps * norm_D / std::sqrt( 3 * double(N) / double(ntile) ); // √3 = √d from HOSVD
    const auto  acc    = absolute_prec( Hpro::frobenius_norm, tol );
    // const auto  cacc   = absolute_prec( Hpro::frobenius_norm, cmdline::eps );
    // const auto  acc    = relative_prec( Hpro::frobenius_norm, tol );
    const auto  cacc   = relative_prec( Hpro::frobenius_norm, eps );
    
    for ( size_t  nz = 0; nz < N; nz += ntile )
    {
        for ( size_t  ny = 0; ny < N; ny += ntile )
        {
            for ( size_t  nx = 0; nx < N; nx += ntile )
            {
                const auto  is0 = indexset( nx, std::min( nx + ntile - 1, N-1 ) );
                const auto  is1 = indexset( ny, std::min( ny + ntile - 1, N-1 ) );
                const auto  is2 = indexset( nz, std::min( nz + ntile - 1, N-1 ) );

                auto        O_sub      = D( is0, is1, is2 );
                auto        D_sub      = blas::copy( O_sub );
                const auto  norm_D_sub = blas::norm_F( D_sub );

                // O_sub.check_data();
                // D_sub.check_data();
                        
                if ( norm_D_sub == value_t(0) )
                    continue;

                // if ( verbose(4) )
                //     std::cout << is0 << " × " << is1 << " × " << is2 << " : " << format_norm( norm_D_sub, acc.abs_eps() * norm_D_sub ) << std::endl;

                auto  G  = blas::tensor3< value_t >();
                auto  X0 = blas::matrix< value_t >();
                auto  X1 = blas::matrix< value_t >();
                auto  X2 = blas::matrix< value_t >();
                        
                //
                // HOSVD
                //
                
                // if (( cmdline::tapprox == "hosvd" ) || ( cmdline::tapprox == "default" ))
                {
                    std::tie( G, X0, X1, X2 ) = blas::hosvd( D_sub, acc, apx );
                }// if

                //
                // decide about format
                //

                const auto  mem_full   = D_sub.data_byte_size();
                const auto  mem_tucker = G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size();
                        
                // if ( mem_tucker < mem_full )
                // {
                //     // if ( verbose(4) )
                //     //     std::cout << is0 << " × " << is1 << " × " << is2 << " : ranks = "
                //     //               << G.size(0) << " / " << G.size(1) << " / " << G.size(2) << " / " << G.size(3) << std::endl;
                            
                //     //
                //     // replace data by uncompressed version
                //     //

                //     auto  T_sub = std::make_unique< tensor::tucker_tensor3< value_t > >( is0, is1, is2,
                //                                                                          std::move( G ),
                //                                                                          std::move( X0 ),
                //                                                                          std::move( X1 ),
                //                                                                          std::move( X2 ) );

                //     if ( fpcompress )
                //         T_sub->compress( cacc );
                            
                //     auto  T0 = blas::tensor_product( T_sub->G(), T_sub->X(0), 0 );
                //     auto  T1 = blas::tensor_product(         T0, T_sub->X(1), 1 );
                //     auto  Y  = blas::tensor_product(         T1, T_sub->X(2), 2 );

                //     blas::copy( Y, O_sub );

                //     zmem += T_sub->data_byte_size();
                // }// if
                // else
                {
                    //
                    // use original data (nothing to do)
                    //

                    auto  T_sub = std::make_unique< tensor::dense_tensor3< value_t > >( is0, is1, is2, std::move( D_sub ) );
                                
                    if ( fpcompress )
                        T_sub->compress( cacc );

                    auto  Y = T_sub->tensor();

                    // Y.check_data();
                            
                    blas::copy( Y, O_sub );

                    zmem += T_sub->data_byte_size();
                }// else

                O_sub.check_data();
            }// for
        }// for
    }// for

    return zmem;
}

template < typename value_t >
size_t
compress_tucker ( blas::tensor4< value_t > &  D,
                  const size_t                ntile,
                  const double &              eps,
                  const bool                  fpcompress )
{
    HLR_ASSERT(( D.size(0) == D.size(1) ) &&
               ( D.size(0) == D.size(2) )); // assuming equal size in all dimensions
    HLR_ASSERT( ( D.size(0) / ntile ) * ntile == D.size(0) );             // no padding for now

    const auto  N    = D.size(0);
    const auto  n3   = D.size(3);
    auto        apx  = approx::SVD< value_t >();
    auto        zmem = size_t( 0 );

    const auto  norm_D = blas::norm_F( D );
    const auto  tol    = 1e-1 * eps * norm_D / std::sqrt( 4 * double(N) / double(ntile) ); // √4 = √d from HOSVD
    // const auto  acc    = absolute_prec( Hpro::frobenius_norm, tol );
    // const auto  cacc   = absolute_prec( Hpro::frobenius_norm, cmdline::eps );
    const auto  acc    = relative_prec( tol );
    const auto  cacc   = relative_prec( eps );
    
    for ( size_t  nz = 0; nz < N; nz += ntile )
    {
        for ( size_t  ny = 0; ny < N; ny += ntile )
        {
            for ( size_t  nx = 0; nx < N; nx += ntile )
            {
                const auto  is0 = indexset( nx, std::min( nx + ntile - 1, N-1 ) );
                const auto  is1 = indexset( ny, std::min( ny + ntile - 1, N-1 ) );
                const auto  is2 = indexset( nz, std::min( nz + ntile - 1, N-1 ) );
                const auto  is3 = indexset( 0, n3 - 1 );

                auto        O_sub      = D( is0, is1, is2, is3 );
                auto        D_sub      = blas::copy( O_sub );
                const auto  norm_D_sub = blas::norm_F( D_sub );

                // O_sub.check_data();
                // D_sub.check_data();
                        
                if ( norm_D_sub == value_t(0) )
                    continue;

                // if ( verbose(4) )
                //     std::cout << is0 << " × " << is1 << " × " << is2 << " : " << format_norm( norm_D_sub, acc.abs_eps() * norm_D_sub ) << std::endl;

                auto  G  = blas::tensor4< value_t >();
                auto  X0 = blas::matrix< value_t >();
                auto  X1 = blas::matrix< value_t >();
                auto  X2 = blas::matrix< value_t >();
                auto  X3 = blas::matrix< value_t >();
                        
                //
                // HOSVD
                //
                
                // if (( cmdline::tapprox == "hosvd" ) || ( cmdline::tapprox == "default" ))
                {
                    std::tie( G, X0, X1, X2, X3 ) = blas::hosvd( D_sub, acc, apx );
                }// if

                //
                // decide about format
                //

                const auto  mem_full   = D_sub.data_byte_size();
                const auto  mem_tucker = G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size() + X3.data_byte_size();
                        
                if ( mem_tucker < mem_full )
                {
                    // if ( verbose(4) )
                        std::cout << is0 << " × " << is1 << " × " << is2 << " : lowrank = "
                                  << G.size(0) << " / " << G.size(1) << " / " << G.size(2) << " / " << G.size(3) << std::endl;
                            
                    //
                    // replace data by uncompressed version
                    //

                    auto  T_sub = std::make_unique< tensor::tucker_tensor4< value_t > >( is0, is1, is2, is3,
                                                                                         std::move( G ),
                                                                                         std::move( X0 ),
                                                                                         std::move( X1 ),
                                                                                         std::move( X2 ),
                                                                                         std::move( X3 ) );

                    if ( fpcompress )
                    {
                        std::cout << T_sub->data_byte_size() << std::endl;
                        T_sub->compress( cacc );
                        std::cout << T_sub->data_byte_size() << std::endl << std::endl;
                    }// if
                            
                    auto  T0 = blas::tensor_product( T_sub->G(), T_sub->X(0), 0 );
                    auto  T1 = blas::tensor_product(         T0, T_sub->X(1), 1 );
                    auto  T2 = blas::tensor_product(         T1, T_sub->X(2), 2 );
                    auto  Y  = blas::tensor_product(         T2, T_sub->X(3), 3 );

                    blas::copy( Y, O_sub );

                    zmem += T_sub->data_byte_size();
                }// if
                else
                {
                    //
                    // use original data (nothing to do)
                    //

                    std::cout << is0 << " × " << is1 << " × " << is2 << " : full = "
                              << G.size(0) << " / " << G.size(1) << " / " << G.size(2) << " / " << G.size(3) << std::endl;
                    
                    auto  T_sub = std::make_unique< tensor::dense_tensor4< value_t > >( is0, is1, is2, is3, std::move( D_sub ) );
                                
                    if ( fpcompress )
                        T_sub->compress( cacc );

                    auto  Y = T_sub->tensor();

                    // Y.check_data();
                            
                    blas::copy( Y, O_sub );

                    zmem += T_sub->data_byte_size();
                }// else

                // O_sub.check_data();
            }// for
        }// for
    }// for

    return zmem;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_COMPRESS_HH
