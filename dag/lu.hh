#ifndef __HLR_DAG_LU_HH
#define __HLR_DAG_LU_HH
//
// Project     : HLib
// File        : lu.hh
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/TMatrix.hh>
#include <algebra/mat_fac.hh>

#include "Graph.hh"

namespace DAG
{

namespace LU
{

//
// return graph representing compute DAG for LU of <A>
//
Graph
gen_dag ( HLIB::TMatrix *  A );

}// namespace LU

}// namespace DAG

#endif // __HLR_DAG_LU_HH
