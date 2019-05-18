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

#include "Graph.hh"

namespace HLR
{

namespace DAG
{

//
// return graph representing compute DAG for LU of <A>
//
Graph
gen_LU_dag ( HLIB::TMatrix *                              A,
             std::function< DAG::Graph ( DAG::Node * ) >  refine );

}// namespace DAG

}// namespace HLR

#endif // __HLR_DAG_LU_HH
