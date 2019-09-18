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
// compute DAG for in-place LU of <A>
//
graph
gen_dag_lu_rec    ( HLIB::TMatrix &    A,
                    refine_func_t      refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
graph
gen_dag_lu_oop    ( HLIB::TMatrix &    A,
                    refine_func_t      refine );

//
// compute DAG for out-of-place LU of <A> with automatic local dependencies
//
graph
gen_dag_lu_oop_auto ( HLIB::TMatrix &  A,
                      refine_func_t    refine );

//
// compute DAG for in-place LU of <A> using accumulators
//
graph
gen_dag_lu_oop_accu ( HLIB::TMatrix &  A,
                      refine_func_t    refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
graph
gen_dag_lu_oop_accu_sep ( HLIB::TMatrix &  A,
                          refine_func_t    refine );

//
// compute DAG for coarse version of LU with on-the-fly DAGs for small matrices
//
graph
gen_dag_lu_oop_coarse  ( HLIB::TMatrix &    A,
                         refine_func_t      refine,
                         exec_func_t        fine_run,
                         const size_t       ncoarse = 0 );

//
// level wise computation of DAG for LU of <A>
// - recurse on diagonal blocks and perform per-level LU on global scope 
//
graph
gen_dag_lu_lvl    ( HLIB::TMatrix &  A );

//
// compute DAG for solving L X = A with lower-triangular L
//
graph
gen_dag_solve_lower  ( const HLIB::TMatrix *  L,
                       HLIB::TMatrix *        A,
                       refine_func_t          refine );

//
// compute DAG for solving X U = A with upper triangular U
//
graph
gen_dag_solve_upper  ( const HLIB::TMatrix *  U,
                       HLIB::TMatrix *        A,
                       refine_func_t          refine );

//
// compute DAG for C = A B + C
//
graph
gen_dag_update       ( const HLIB::TMatrix *  A,
                       const HLIB::TMatrix *  B,
                       HLIB::TMatrix *        C,
                       refine_func_t          refine );

//
// compute DAG for tile-based LU of <A> in HODLR format
//
graph
gen_dag_lu_hodlr_tiled ( HLIB::TMatrix &  A,
                         refine_func_t    refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_LU_HH
