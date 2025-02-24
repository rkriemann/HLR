#ifndef __HLR_TENSOR_CONSTRUCT_HH
#define __HLR_TENSOR_CONSTRUCT_HH
//
// Project     : HLR
// Module      : tensor/construct
// Description : adaptive construction of an hierarchical tensor
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/approx/traits.hh>
#include <hlr/approx/accuracy.hh>

#include <hlr/tensor/base_tensor.hh>
#include <hlr/tensor/detail/construct.hh>

namespace hlr { namespace tensor {

//
// build hierarchical tensor representation from given dense tensor, starting
// with subblocks of size ntile × ntile × ntile and then trying to merge
// tucker blocks as long as not increasing memory
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< base_tensor3< value_t > >
build_hierarchical_tucker ( const blas::tensor3< value_t > &  D,
                            const tensor_accuracy &           acc,
                            const approx_t &                  apx,
                            const size_t                      ntile )
{
    //
    // handle parallel computation of norm because BLAS should be sequential
    //
    
    // const auto  norm_D = blas::norm_F( D );

    //
    // start compressing with per-block accuracy
    //
    
    // const auto  delta  = norm_D * rel_prec / D.size(0);
    // auto        acc_D  = Hpro::fixed_prec( delta );
    auto        M      = std::unique_ptr< base_tensor3< value_t > >();
    
    M = std::move( detail::build_hierarchical_tucker( indexset( 0, D.size(0)-1 ),
                                                      indexset( 0, D.size(1)-1 ),
                                                      indexset( 0, D.size(2)-1 ),
                                                      D, acc, apx, ntile ) );

    HLR_ASSERT( M.get() != nullptr );

    return M;
}

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< tensor::structured_tensor3< value_t > >
blockwise_tucker ( blas::tensor3< value_t > &  D,
                   const size_t                ntile,
                   const accuracy &            acc,  // HOSVD accuracy
                   const accuracy &            cacc, // float compression accuracy
                   const approx_t &            apx )
{
    using  real_t = real_type_t< value_t >;
    
    const auto  N  = D.size(0); // assuming equal size in all dimensions
    const auto  Nb = N % ntile == 0 ? N / ntile : N / ntile + 1;

    auto  T = std::make_unique< tensor::structured_tensor3< value_t > >( indexset( 0, N-1 ), indexset( 0, N-1 ), indexset( 0, N-1 ) );

    T->set_structure( Nb, Nb, Nb );
    
    for ( size_t  nz = 0, z = 0; nz < N; nz += ntile, ++z )
    {
        for ( size_t  ny = 0, y = 0; ny < N; ny += ntile, ++y )
        {
            for ( size_t  nx = 0, x = 0; nx < N; nx += ntile, ++x )
            {
                const auto  is0 = indexset( nx, std::min( nx + ntile - 1, N-1 ) );
                const auto  is1 = indexset( ny, std::min( ny + ntile - 1, N-1 ) );
                const auto  is2 = indexset( nz, std::min( nz + ntile - 1, N-1 ) );

                auto        D_ijk      = blas::copy( D( is0, is1, is2 ) );
                const auto  norm_D_ijk = blas::norm_F( D_ijk );

                // std::cout << is0 << " × " << is1 << " × " << is2 << " : " << norm_D_ijk << " / " << acc.abs_eps() * norm_D_ijk << std::endl;

                if ( norm_D_ijk == value_t(0) )
                    continue;

                auto  G  = blas::tensor3< value_t >();
                auto  X0 = blas::matrix< value_t >();
                auto  X1 = blas::matrix< value_t >();
                auto  X2 = blas::matrix< value_t >();
                        
                //
                // HOSVD
                //
                
                // if (( tapprox == "hosvd" ) || ( tapprox == "default" ))
                {
                    auto  S0 = blas::vector< real_t >();
                    auto  S1 = blas::vector< real_t >();
                    auto  S2 = blas::vector< real_t >();
                            
                    std::tie( G, X0, S0, X1, S1, X2, S2 ) = blas::hosvd_sv( D_ijk, acc, apx );
                            
                    // std::cout << is0 << " × " << is1 << " × " << is2 << " : "
                    //           << boost::format( "%.4e" ) % std::max( std::max( S0(0), S1(0) ), S2(0) ) << std::endl;

                    // std::tie( G, X0, X1, X2 ) = blas::hosvd( D_ijk, acc, apx );
                }// if

                //
                // ST-HOSVD
                //

                // if ( tapprox == "sthosvd" )
                // {
                //     std::tie( G, X0, X1, X2 ) = blas::sthosvd( D_ijk, acc, apx );
                // }// if

                //
                // Greedy-HOSVD
                //

                // if ( tapprox == "ghosvd" )
                // {
                //     HLR_ERROR( "TODO" );
                //     // std::tie( G, X0, X1, X2 ) = impl::blas::greedy_hosvd( D_ijk, acc, apx );
                // }// if

                // std::cout << is0 << " × " << is1 << " × " << is2 << " : " << G.size(0) << " × " << G.size(1) << " × " << G.size(2) << std::endl;
                        
                //
                // decide about format
                //

                const auto  mem_full   = D_ijk.data_byte_size();
                const auto  mem_tucker = G.data_byte_size() + X0.data_byte_size() + X1.data_byte_size() + X2.data_byte_size();
                        
                if ( mem_tucker < mem_full )
                {
                    auto  T_sub = std::make_unique< tensor::tucker_tensor3< value_t > >( is0, is1, is2,
                                                                                         std::move( G ),
                                                                                         std::move( X0 ),
                                                                                         std::move( X1 ),
                                                                                         std::move( X2 ) );

                    if ( ! cacc.is_exact() )
                        T_sub->compress( cacc );
                    
                    T->set_block( x, y, z, T_sub.release() );
                }// if
                else
                {
                    auto  T_sub = std::make_unique< tensor::dense_tensor3< value_t > >( is0, is1, is2, std::move( D_ijk ) );
                    
                    if ( ! cacc.is_exact() )
                        T_sub->compress( cacc );
                    
                    T->set_block( x, y, z, T_sub.release() );
                }// else

                // std::cout << is0 << " × " << is1 << " × " << is2 << " : "
                //           << "mem = " << T->block(x,y,z)->data_byte_size() << " / "
                //           << boost::format( "%.1fx" ) % ( double(mem_full) / double(T->block(x,y,z)->data_byte_size()) ) << std::endl;
            }// for
        }// for
    }// for

    return  T;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_CONSTRUCT_HH
