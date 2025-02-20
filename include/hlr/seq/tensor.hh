#ifndef __HLR_SEQ_TENSOR_HH
#define __HLR_SEQ_TENSOR_HH
//
// Project     : HLR
// Module      : seq/tensor
// Description : sequential tensor algorithms
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/tensor/construct.hh>
#include <hlr/tensor/convert.hh>
#include <hlr/tensor/compress.hh>

namespace hlr { namespace seq { namespace tensor {

using hlr::tensor::build_hierarchical_tucker;
using hlr::tensor::to_dense;
using hlr::tensor::compress_tucker;
using hlr::tensor::blockwise_tucker;

}}}// namespace hlr::seq::tensor

#endif // __HLR_SEQ_TENSOR_HH
