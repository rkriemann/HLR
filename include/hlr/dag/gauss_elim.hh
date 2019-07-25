#ifndef __HLR_DAG_GAUSS_ELIM_HH
#define __HLR_DAG_GAUSS_ELIM_HH
//
// Project     : HLib
// File        : gauss_elim.hh
// Description : generate DAG for Gaussian elimination
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <matrix/TMatrix.hh>

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace dag
{

//
// return graph representing compute DAG for Gaussian elimination of <A>
//
dag::graph
gen_dag_gauss_elim ( HLIB::TMatrix *  A,
                     HLIB::TMatrix *  C,
                     refine_func_t    refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_GAUSS_ELIM_HH
