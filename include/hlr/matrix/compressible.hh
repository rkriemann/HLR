#ifndef __HLR_MATRIX_COMPRESSIBLE_HH
#define __HLR_MATRIX_COMPRESSIBLE_HH
//
// Project     : HLR
// Module      : matrix/compressible
// Description : defines interface for compressible objects
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/utils/compression.hh>
#include <hlr/utils/checks.hh>

namespace hlr { namespace matrix {

using hlr::compress::is_compressible;

HLR_TEST_ALL( is_compressible, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_compressible, Hpro::TMatrix< value_t > )

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_COMPRESSIBLE_HH
