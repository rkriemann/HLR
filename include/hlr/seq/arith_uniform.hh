#ifndef __HLR_SEQ_ARITH_UNIFORM_HH
#define __HLR_SEQ_ARITH_UNIFORM_HH
//
// Project     : HLib
// Module      : seq/arith_uniform
// Description : sequential arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/uniform.hh>

namespace hlr { namespace seq { namespace uniform {

using hlr::uniform::mul_vec;
using hlr::uniform::multiply;
using hlr::uniform::lu;

namespace accu
{

using hlr::uniform::accu::lu;

}// namespace accu

namespace accu2
{

using hlr::uniform::accu2::multiply;
using hlr::uniform::accu2::lu;

}// namespace accu

namespace accu3
{

using hlr::uniform::accu3::lu;

}// namespace accu

namespace tlr
{

// using hlr::uniform::tlr::addlr;
using hlr::uniform::tlr::multiply;
using hlr::uniform::tlr::lu;
using hlr::uniform::tlr::lu_sep;
using hlr::uniform::tlr::lu_lazy;
using hlr::uniform::tlr::ldu;

}// namespace tlr

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
