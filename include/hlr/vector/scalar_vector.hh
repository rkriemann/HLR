#ifndef __HLR_VECTOR_SCALAR_VECTOR_HH
#define __HLR_VECTOR_SCALAR_VECTOR_HH
//
// Project     : HLR
// Module      : scalar_vector.hh
// Description : standard scalar vector
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hpro/vector/TScalarVector.hh>

namespace hlr { namespace vector {

template < typename value_t > using scalar_vector = Hpro::TScalarVector< value_t >;

}}// namespace hlr::vector

#endif // __HLR_VECTOR_SCALAR_VECTOR_HH
