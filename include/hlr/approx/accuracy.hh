#ifndef __HLR_APPROX_ACCURACY_HH
#define __HLR_APPROX_ACCURACY_HH
//
// Project     : HLR
// Module      : approx/accuracy
// Description : truncation accuracy handling
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/base/TTruncAcc.hh>
#include <hpro/cluster/TIndexSet.hh>

namespace hlr
{

using indexset = Hpro::TIndexSet;
using accuracy = Hpro::TTruncAcc;

using Hpro::fixed_prec;
using Hpro::relative_prec;
using Hpro::absolute_prec;
using Hpro::fixed_rank;

using Hpro::spectral_norm;
using Hpro::frobenius_norm;

//
// extend interface for tensor blocks
//
struct tensor_accuracy : public accuracy
{
    //
    // ctors
    //

    tensor_accuracy ()
            : accuracy()
    {}

    tensor_accuracy ( const Hpro::trunc_norm_t  anorm_mode,
                      const double              arelative_eps,
                      const double              aabsolute_eps = Hpro::CFG::Arith::abs_eps )
            : accuracy( anorm_mode, arelative_eps, aabsolute_eps )
    {}
    
    tensor_accuracy ( const accuracy &  aacc )
            : accuracy( aacc )
    {}
    
    //
    // return local accuracy for individual (tensor) subblock
    //
    
    virtual
    const tensor_accuracy
    acc ( const indexset &  /* is0 */,
          const indexset &  /* is1 */,
          const indexset &  /* is2 */ ) const
    {
        return *this;
    }
    using accuracy::acc;

    // same in operator form
    const tensor_accuracy
    operator () ( const indexset &  is0,
                  const indexset &  is1,
                  const indexset &  is2 ) const
    {
        return acc( is0, is1, is2 );
    }
};

}// namespace hlr

#endif  // __HLR_APPROX_ACCURACY_HH
