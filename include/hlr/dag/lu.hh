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
gen_dag_lu_rec  ( HLIB::TMatrix *                              A,
                  std::function< dag::graph ( dag::node * ) >  refine );

//
// return graph representing compute DAG for LU of <A>
// - compute DAG level wise
//
graph
gen_dag_lu_lvl  ( HLIB::TMatrix &  A );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_LU_HH
