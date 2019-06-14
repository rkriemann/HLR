#ifndef __HLR_DAG_LU_HH
#define __HLR_DAG_LU_HH
//
// Project     : HLib
// File        : lu.hh
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <functional>

#include <matrix/TMatrix.hh>
#include <algebra/mat_fac.hh>

#include "hlr/dag/graph.hh"
#include "hlr/matrix/level_matrix.hh"

namespace hlr
{

namespace dag
{

//
// return graph representing compute DAG for LU of <A>
//
graph
gen_lu_dag ( HLIB::TMatrix *                              A,
             std::function< dag::graph ( dag::node * ) >  refine );

graph
gen_lu_dag ( hlr::matrix::level_matrix &                  L,
             std::function< dag::graph ( dag::node * ) >  refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_LU_HH
