#ifndef __HLR_VECTOR_SCALAR_VECTOR_HH
#define __HLR_VECTOR_SCALAR_VECTOR_HH
//
// Project     : HLR
// Module      : scalar_vector.hh
// Description : standard scalar vector
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/vector/TScalarVector.hh>

namespace hlr { namespace vector {

//
// just import class/functions from HLIBpro
//

using scalar_vector = hpro::TScalarVector;

using hpro::blas_vec;
using hpro::is_scalar;

}}// namespace hlr::vector

#endif // __HLR_VECTOR_SCALAR_VECTOR_HH
