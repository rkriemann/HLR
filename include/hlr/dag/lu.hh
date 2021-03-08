#ifndef __HLR_DAG_LU_HH
#define __HLR_DAG_LU_HH
//
// Project     : HLib
// Module      : dag/lu
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include "hlr/dag/graph.hh"
#include "hlr/dag/lu_nodes.hh"

namespace hlr { namespace dag {

//
// compute DAG for H-LU of <A> with immediate updates
//
template < typename value_t,
           typename approx_t >
graph
gen_dag_lu ( hpro::TMatrix &  A,
             const size_t     min_size,
             refine_func_t    refine )
{
    auto  dag = refine( new lu::lu_node< value_t, approx_t >( & A ), min_size, use_single_end_node );

    return dag;
}

//
// compute DAG for H-LU of <A> with accumulated updates
// - aside from the DAG, also a map for per-matrix accumulators
//   is generated and returned. This is needed during DAG execution!
//
template < typename value_t,
           typename approx_t >
std::pair< graph, dag::lu::accu::accumulator_map_t >
gen_dag_lu_accu ( hpro::TMatrix &  A,
                  const size_t     min_size,
                  refine_func_t    refine )
{
    // generate DAG for shifting and applying updates
    auto  accu_map                   = dag::lu::accu::accumulator_map_t();
    auto  [ apply_map, apply_nodes ] = lu::accu::build_apply_dag< value_t, approx_t >( & A, accu_map, min_size );

    // construct H-LU DAG
    auto  dag = refine( new lu::accu::lu_node< value_t, approx_t >( & A, apply_map ), min_size, use_single_end_node );

    //
    // add apply/shift nodes and set shift(A) as new start
    //
        
    for ( auto  node : apply_nodes )
    {
        dag.nodes().push_back( node );

        // adjust dependency counters
        for ( auto  succ : node->successors() )
            succ->inc_dep_cnt();
    }// for

    dag.start().clear();
    dag.start().push_back( apply_map[ A.id() ] );
        
    return { std::move( dag ), std::move( accu_map ) };
}

//
// compute DAG for in-place LU of <A>
//
graph
gen_dag_lu_ip     ( hpro::TMatrix &    A,
                    const size_t       min_size,
                    refine_func_t      refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
graph
gen_dag_lu_oop    ( hpro::TMatrix &    A,
                    const size_t       min_size,
                    refine_func_t      refine );

//
// compute DAG for out-of-place LU of <A> with automatic local dependencies
//
graph
gen_dag_lu_oop_auto ( hpro::TMatrix &  A,
                      const size_t     min_size,
                      refine_func_t    refine );

//
// compute DAG for in-place LU of <A> using accumulators
//
graph
gen_dag_lu_oop_accu ( hpro::TMatrix &  A,
                      const size_t     min_size,
                      refine_func_t    refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
graph
gen_dag_lu_oop_accu_sep ( hpro::TMatrix &  A,
                          const size_t     min_size,
                          refine_func_t    refine );

//
// compute DAG for coarse version of LU with on-the-fly DAGs for small matrices
//
graph
gen_dag_lu_oop_coarse  ( hpro::TMatrix &    A,
                         const size_t       ncoarse,
                         refine_func_t      refine,
                         exec_func_t        fine_run );

//
// level wise computation of DAG for LU of <A>
// - recurse on diagonal blocks and perform per-level LU on global scope 
//
graph
gen_dag_lu_lvl    ( hpro::TMatrix &  A,
                    const size_t     min_size );

//
// generate DAG for Tile-H LU factorisation
//
graph
gen_dag_lu_tileh ( hpro::TMatrix &  A,
                   const size_t     min_size,
                   refine_func_t    refine,
                   exec_func_t      exec );

//
// compute DAG for solving L X = A with lower-triangular L
//
graph
gen_dag_solve_lower  ( const hpro::TMatrix &  L,
                       hpro::TMatrix &        A,
                       const size_t           min_size,
                       refine_func_t          refine );

//
// compute DAG for solving X U = A with upper triangular U
//
graph
gen_dag_solve_upper  ( const hpro::TMatrix &  U,
                       hpro::TMatrix &        A,
                       const size_t           min_size,
                       refine_func_t          refine );

//
// compute DAG for C = A B + C
//
graph
gen_dag_update       ( const hpro::TMatrix &  A,
                       const hpro::TMatrix &  B,
                       hpro::TMatrix &        C,
                       const size_t           min_size,
                       refine_func_t          refine );

//
// compute DAG for TSQR( X·T, U )
//
graph
gen_dag_tsqr ( const size_t                                                  n,
               hlr::matrix::tile_storage< hpro::real > &                     X,
               std::shared_ptr< hpro::BLAS::Matrix< hpro::real > > &         T,
               hlr::matrix::tile_storage< hpro::real > &                     U,
               std::shared_ptr< hlr::matrix::tile_storage< hpro::real > > &  Q,
               std::shared_ptr< hpro::BLAS::Matrix< hpro::real > > &         R,
               refine_func_t                                                 refine );

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
graph
gen_dag_truncate ( hlr::matrix::tile_storage< hpro::real > &              X,
                   std::shared_ptr< hpro::BLAS::Matrix< hpro::real > > &  T,
                   hlr::matrix::tile_storage< hpro::real > &              Y,
                   hlr::matrix::tiled_lrmatrix< hpro::real > *            A,
                   refine_func_t                                          refine );

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
graph
gen_dag_addlr ( hlr::matrix::tile_storage< hpro::real > &              X,
                std::shared_ptr< hpro::BLAS::Matrix< hpro::real > > &  T,
                hlr::matrix::tile_storage< hpro::real > &              Y,
                hpro::TMatrix *                                        A,
                refine_func_t                                          refine );

}}// namespace hlr::dag

#endif // __HLR_DAG_LU_HH
