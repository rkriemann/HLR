#ifndef __HLR_DAG_SOLVE_HH
#define __HLR_DAG_SOLVE_HH
//
// Project     : HLR
// Module      : SOLVE.hh
// Description : DAGs for matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <map>
#include <mutex>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/vector/TScalarVector.hh>

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace dag
{

// size of lockable chunks for vector updates
constexpr size_t  CHUNK_SIZE = 128;

// map from index to mutex
using  mutex_map_t = std::map< Hpro::idx_t, std::unique_ptr< std::mutex > >;

//
// return graph representing compute DAG for solving op(L) x = y
// with lower triangular L
//
template < typename value_t >
graph
gen_dag_solve_lower ( const Hpro::matop_t                op_L,
                      Hpro::TMatrix< value_t > *         L,
                      Hpro::TScalarVector< value_t > **  x,
                      mutex_map_t &                      mtx_map,
                      refine_func_t                      refine );

//
// return graph representing compute DAG for solving op(U) x = y
// with upper triangular U
//
template < typename value_t >
graph
gen_dag_solve_upper ( const Hpro::matop_t                op_U,
                      Hpro::TMatrix< value_t > *         U,
                      Hpro::TScalarVector< value_t > **  x,
                      mutex_map_t &                      mtx_map,
                      refine_func_t                      refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_SOLVE_HH
