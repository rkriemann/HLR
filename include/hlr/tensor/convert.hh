#ifndef __HLR_TENSOR_CONVERT_HH
#define __HLR_TENSOR_CONVERT_HH
//
// Project     : HLR
// Module      : tensor/convert
// Description : various conversion functions between tensors
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/tensor/dense_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/structured_tensor.hh>

namespace hlr { namespace tensor {

//
// forward decl. as used within detail::to_dense
//
template < typename value_t >
std::unique_ptr< dense_tensor3< value_t > >
to_dense ( const base_tensor3< value_t > &  X );

namespace detail
{

//
// copy given tensor into D
//
template < typename value_t >
void
to_dense ( const base_tensor3< value_t > &  X,
           blas::tensor3< value_t > &       D )
{
    if ( is_structured( X ) )
    {
        auto  BX    = cptrcast( &X, structured_tensor3< value_t > );

        for ( uint  l = 0; l < BX->nblocks(2); ++l )
        {
            for ( uint  j = 0; j < BX->nblocks(1); ++j )
            {
                for ( uint  i = 0; i < BX->nblocks(0); ++i )
                {
                    if ( ! is_null( BX->block(i,j,l) ) )
                        to_dense( *BX->block(i,j,l), D );
                }// for
            }// for
        }// for
    }// if
    else if ( is_tucker( X ) )
    {
        auto  TX    = cptrcast( &X, tucker_tensor3< value_t > );
        auto  T0    = blas::tensor_product( TX->G(), TX->X(0), 0 );
        auto  T1    = blas::tensor_product( T0,      TX->X(1), 1 );
        auto  DX    = blas::tensor_product( T1,      TX->X(2), 2 );
        auto  D_sub = D( X.is(0), X.is(1), X.is(2) );
        
        blas::copy( DX, D_sub );
    }// if
    else if ( is_dense( X ) )
    {
        auto  DX    = cptrcast( &X, dense_tensor3< value_t > );
        auto  D_sub = D( X.is(0), X.is(1), X.is(2) );

        blas::copy( DX->tensor(), D_sub );
    }// if
    else
    {
        HLR_ERROR( "unknown tensor type" );
    }// if
}

}// namespace detail

//
// convert to dense tensor
//
template < typename value_t >
std::unique_ptr< dense_tensor3< value_t > >
to_dense ( const base_tensor3< value_t > &  X )
{
    // only zero offsets for now
    HLR_ASSERT( ( X.is(0).first() == 0 ) &&
                ( X.is(1).first() == 0 ) &&
                ( X.is(2).first() == 0 ) )
        
    if ( is_dense( X ) )
    {
        auto  D = X.copy();

        return std::unique_ptr< dense_tensor3< value_t > >( ptrcast( D.release(), dense_tensor3< value_t > ) );
    }// if
    else
    {
        auto  D = std::make_unique< dense_tensor3< value_t > >( X.is(0), X.is(1), X.is(2) );
        
        detail::to_dense( X, D->tensor() );

        return D;
    }// if
}

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_CONVERT_HH
