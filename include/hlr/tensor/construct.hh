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

}}// namespace hlr::tensor

#endif // __HLR_TENSOR_CONSTRUCT_HH
