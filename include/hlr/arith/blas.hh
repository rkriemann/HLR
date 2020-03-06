#ifndef __HLR_ARITH_BLAS_HH
#define __HLR_ARITH_BLAS_HH
//
// Project     : HLR
// Module      : arith/blas
// Description : basic linear algebra functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/blas/Matrix.hh>
#include <hpro/blas/Vector.hh>
#include <hpro/blas/Algebra.hh>

namespace hlr { namespace blas {

//
// import functions from HLIBpro and adjust naming
//

using namespace HLIB::BLAS;

using range = HLIB::BLAS::Range;

template < typename value_t > using vector = HLIB::BLAS::Vector< value_t >;
template < typename value_t > using matrix = HLIB::BLAS::Matrix< value_t >;

}}// namespace hlr::blas

#endif // __HLR_ARITH_BLAS_HH
