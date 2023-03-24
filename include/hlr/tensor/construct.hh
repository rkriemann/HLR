#ifndef __HLR_TENSOR_CONSTRUCT_HH
#define __HLR_TENSOR_CONSTRUCT_HH
//
// Project     : HLR
// Module      : tensor/construct
// Description : adaptive construction of an hierarchical tensor
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/arith/tensor.hh>
#include <hlr/approx/traits.hh>

#include <hlr/tensor/base_tensor.hh>
#include <hlr/tensor/dense_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/structured_tensor.hh>

namespace hlr { namespace tensor {

//
// build hierarchical tensor representation from given dense tensor, starting
// with subblocks of size ntile × ntile × ntile and then trying to merge
// tucker blocks as long as not increasing memory
//
namespace detail
{

template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< base_tensor3< value_t > >
build_hierarchical_tucker ( const indexset &                  is0,
                            const indexset &                  is1,
                            const indexset &                  is2,
                            const blas::tensor3< value_t > &  D,
                            const accuracy &                  acc,
                            const approx_t &                  approx,
                            const size_t                      ntile )
{
    if ( std::min( D.size(0), std::min( D.size(1), D.size(2) ) ) <= ntile )
    {
        //
        // build leaf
        //

        if ( ! acc.is_exact() )
        {
            auto  Dc                = blas::copy( D );  // do not modify D (!)
            auto  [ G, X0, X1, X2 ] = blas::sthosvd( Dc, acc, approx );
            
            if ( G.byte_size() + X0.byte_size() + X1.byte_size() + X2.byte_size() < Dc.byte_size() )
            {
                // std::cout << "R: " << to_string( is0 ) << " x " << to_string( is1 ) << " x " << to_string( is2 )
                //           << " : " << X0.ncols() << " / " << X1.ncols() << " / " << X2.ncols()
                //           << std::endl;

                return std::make_unique< tucker_tensor3< value_t > >( is0, is1, is2,
                                                                      std::move( G ),
                                                                      std::move( X0 ),
                                                                      std::move( X1 ),
                                                                      std::move( X2 ) );
            }// if
        }// if

        // std::cout << "D: " << to_string( is0 ) << " x " << to_string( is1 ) << " x " << to_string( is2 ) << std::endl;

        return std::make_unique< dense_tensor3< value_t > >( is0, is1, is2, std::move( blas::copy( D ) ) );
    }// if
    else
    {
        //
        // Recursion
        //

        const auto  mid0 = ( is0.first() + is0.last() + 1 ) / 2;
        const auto  mid1 = ( is1.first() + is1.last() + 1 ) / 2;
        const auto  mid2 = ( is2.first() + is2.last() + 1 ) / 2;

        indexset    sub_is0[2] = { indexset( is0.first(), mid0-1 ), indexset( mid0, is0.last() ) };
        indexset    sub_is1[2] = { indexset( is1.first(), mid1-1 ), indexset( mid1, is1.last() ) };
        indexset    sub_is2[2] = { indexset( is2.first(), mid2-1 ), indexset( mid2, is2.last() ) };
        auto        sub_D      = hlr::tensor3< std::unique_ptr< base_tensor3< value_t > > >( 2, 2, 2 );

        for ( uint  l = 0; l < 2; ++l )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                for ( uint  i = 0; i < 2; ++i )
                {
                    const auto  D_sub = D( sub_is0[i] - is0.first(),
                                           sub_is1[j] - is1.first(),
                                           sub_is2[l] - is2.first() );
                    
                    sub_D(i,j,l) = build_hierarchical_tucker( sub_is0[i], sub_is1[j], sub_is2[l], D_sub, acc, approx, ntile );
                    
                    HLR_ASSERT( sub_D(i,j,l).get() != nullptr );
                }// for
            }// for
        }// for

        //
        // construct structured tensor
        //

        // std::cout << "B: " << to_string( is0 ) << " x " << to_string( is1 ) << " x " << to_string( is2 ) << std::endl;
        
        auto  B = std::make_unique< structured_tensor3< value_t > >( is0, is1, is2 );

        B->set_structure( 2, 2, 2 );
        
        for ( uint  l = 0; l < 2; ++l )
            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    B->set_block( i, j, l, sub_D(i,j,l).release() );
            
        return B;
    }// else
}

}// namespace detail

//
// compression function with all user defined options
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< base_tensor3< value_t > >
build_hierarchical_tucker ( const blas::tensor3< value_t > &  D,
                            const accuracy &                  acc,
                            const approx_t &                  approx,
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
                                                      D, acc, approx, ntile ) );

    HLR_ASSERT( M.get() != nullptr );

    return M;
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_CONSTRUCT_HH
