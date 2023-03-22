#ifndef __HLR_SEQ_TENSOR_HH
#define __HLR_SEQ_TENSOR_HH
//
// Project     : HLR
// Module      : seq/tensor
// Description : sequential tensor algorithms
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/tensor/construct.hh>
#include <hlr/tensor/convert.hh>

namespace hlr { namespace seq { namespace tensor {

using hlr::tensor::build_hierarchical_tucker;
using hlr::tensor::to_dense;

}}}// namespace hlr::seq::tensor

#endif // __HLR_SEQ_TENSOR_HH
