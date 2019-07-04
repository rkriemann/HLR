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
gen_dag_lu_rec    ( HLIB::TMatrix *  A,
                    refine_func_t    refine );

//
// return graph representing compute DAG for LU of <A>
//
graph
gen_dag_coarselu  ( HLIB::TMatrix *                                            A,
                    refine_func_t                                              coarse_refine,
                    refine_func_t                                              fine_refine,
                    const std::function< void ( hlr::dag::graph &,
                                                const HLIB::TTruncAcc & ) >    fine_run,
                    const size_t                                               ncoarse = 0 );

//
// return graph representing compute DAG for LU of <A>
// - compute DAG level wise
//
graph
gen_dag_lu_lvl    ( HLIB::TMatrix &  A );

//
// return graph representing compute DAG for solving L X = A
//
graph
gen_dag_solve_lower  ( const HLIB::TMatrix *  L,
                       HLIB::TMatrix *        A,
                       refine_func_t          refine );

//
// return graph representing compute DAG for solving X U = A
//
graph
gen_dag_solve_upper  ( const HLIB::TMatrix *  U,
                       HLIB::TMatrix *        A,
                       refine_func_t          refine );

//
// return graph representing compute DAG for C = A B + C
//
graph
gen_dag_update       ( const HLIB::TMatrix *  A,
                       const HLIB::TMatrix *  B,
                       HLIB::TMatrix *        C,
                       refine_func_t          refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_LU_HH
