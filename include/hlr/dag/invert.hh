#ifndef __HLR_DAG_INVERT_HH
#define __HLR_DAG_INVERT_HH
//
// Project     : HLR
// Module      : invert.hh
// Description : DAGs for matrix inversion
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>

#include "hlr/dag/graph.hh"

namespace hlr
{

namespace dag
{

//
// compute DAG for lower triangular inversion of L
// - if <diag> == unit_diag, diagonal blocks are not modified
//
template < typename value_t >
dag::graph
gen_dag_invert_ll ( Hpro::TMatrix< value_t > &  L,
                    const Hpro::diag_type_t     diag,
                    const size_t                min_size,
                    refine_func_t               refine );

//
// compute DAG for upper triangular inversion of U
// - if <diag> == unit_diag, diagonal blocks are not modified
//
template < typename value_t >
dag::graph
gen_dag_invert_ur ( Hpro::TMatrix< value_t > &  U,
                    const Hpro::diag_type_t     diag,
                    const size_t                min_size,
                    refine_func_t               refine );

//
// compute DAG for WAZ factorization of A
//
template < typename value_t >
dag::graph
gen_dag_waz       ( Hpro::TMatrix< value_t > &  A,
                    const size_t                min_size,
                    refine_func_t               refine );

//
// compute DAG for inversion of A
//
template < typename value_t >
dag::graph
gen_dag_invert    ( Hpro::TMatrix< value_t > &  A,
                    const size_t                min_size,
                    refine_func_t               refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_GAUSS_ELIM_HH
