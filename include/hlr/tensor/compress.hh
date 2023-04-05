#ifndef __HLR_TENSOR_COMPRESS_HH
#define __HLR_TENSOR_COMPRESS_HH
//
// Project     : HLR
// Module      : tensor/compress
// Description : tensor (data) compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/tensor/dense_tensor.hh>
#include <hlr/tensor/tucker_tensor.hh>
#include <hlr/tensor/structured_tensor.hh>
#include <hlr/approx/accuracy.hh>

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

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_COMPRESS_HH
