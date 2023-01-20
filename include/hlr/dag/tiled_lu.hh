#ifndef __HLR_DAG_TILED_LU_HH
#define __HLR_DAG_TILED_LU_HH
//
// Project     : HLib
// Module      : dag/tiled_lu
// Description : functions for tiled LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>

#include "hlr/dag/graph.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"

namespace hlr { namespace dag {

//
// compute DAG for tile-based LU of <A> in HODLR format
//
graph
gen_dag_lu_hodlr_tiled  ( Hpro::TMatrix< value_t > &  A,
                          const size_t                ntile,
                          refine_func_t               refine );

//
// compute DAG for tile-based LU of <A> in HODLR format
// with lazy update evaluation
//
graph
gen_dag_lu_hodlr_tiled_lazy  ( Hpro::TMatrix< value_t > &  A,
                               const size_t                ntile,
                               refine_func_t               refine );

}}// namespace hlr::dag

#endif // __HLR_DAG_TILED_LU_HH
