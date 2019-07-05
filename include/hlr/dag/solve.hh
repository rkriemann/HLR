#ifndef __HLR_DAG_SOLVE_HH
#define __HLR_DAG_SOLVE_HH
//
// Project     : HLib
// File        : SOLVE.hh
// Description : DAGs for matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <map>
#include <mutex>

#include <matrix/TMatrix.hh>
#include <vector/TScalarVector.hh>

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace dag
{

// size of lockable chunks for vector updates
constexpr size_t  CHUNK_SIZE = 1024;

//
// return graph representing compute DAG for solving op(L) x = y
// with lower triangular L
//
graph
gen_dag_solve_lower ( const HLIB::matop_t    op_L,
                      HLIB::TMatrix *        L,
                      HLIB::TScalarVector &  x,
                      refine_func_t          refine,
                      std::map< HLIB::idx_t, std::unique_ptr< std::mutex > > &  mtx_map );

//
// return graph representing compute DAG for solving op(U) x = y
// with upper triangular U
//
graph
gen_dag_solve_upper ( const HLIB::matop_t    op_U,
                      HLIB::TMatrix *        U,
                      HLIB::TScalarVector &  x,
                      refine_func_t          refine,
                      std::map< HLIB::idx_t, std::unique_ptr< std::mutex > > &  mtx_map );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_SOLVE_HH
