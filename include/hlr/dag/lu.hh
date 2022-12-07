#ifndef __HLR_DAG_LU_HH
#define __HLR_DAG_LU_HH
//
// Project     : HLib
// Module      : dag/lu
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>

#include <hlr/dag/graph.hh>
#include <hlr/dag/detail/lu.hh>
#include <hlr/dag/detail/lu_accu.hh>

namespace hlr { namespace dag {

//
// compute DAG for H-LU of <A> with immediate updates
//
template < typename value_t,
           typename approx_t >
graph
gen_dag_lu ( Hpro::TMatrix< value_t > & A,
             const size_t               min_size,
             refine_func_t              refine,
             const approx_t &           /* apx */ )
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
std::pair< graph, dag::lu::accu::accumulator_map_t< value_t > * >
gen_dag_lu_accu ( Hpro::TMatrix< value_t > & A,
                  const size_t               min_size,
                  refine_func_t              refine,
                  const approx_t &           /* apx */ )
{
    // generate DAG for shifting and applying updates
    auto  accu_map                   = new dag::lu::accu::accumulator_map_t< value_t >();
    auto  [ apply_map, apply_nodes ] = lu::accu::build_apply_dag< value_t, approx_t >( & A, *accu_map, min_size );

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

    // update old nodes as well
    for ( auto  node : dag.nodes() )
        node->finalize();
    
    dag.start().clear();
    dag.start().push_back( apply_map[ A.id() ] );
        
    std::cout << accu_map->contains( 0 ) << std::endl;
        
    return { std::move( dag ), accu_map };
}

//
// compute DAG for in-place LU of <A>
//
template < typename value_t >
graph
gen_dag_lu_ip     ( Hpro::TMatrix< value_t > & A,
                    const size_t               min_size,
                    refine_func_t              refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
template < typename value_t >
graph
gen_dag_lu_oop    ( Hpro::TMatrix< value_t > & A,
                    const size_t               min_size,
                    refine_func_t              refine );

//
// compute DAG for out-of-place LU of <A> with automatic local dependencies
//
template < typename value_t >
graph
gen_dag_lu_oop_auto ( Hpro::TMatrix< value_t > & A,
                      const size_t               min_size,
                      refine_func_t              refine );

//
// compute DAG for in-place LU of <A> using accumulators
//
template < typename value_t >
graph
gen_dag_lu_oop_accu ( Hpro::TMatrix< value_t > &  A,
                      const size_t                min_size,
                      refine_func_t               refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
template < typename value_t >
graph
gen_dag_lu_oop_accu_sep ( Hpro::TMatrix< value_t > &  A,
                          const size_t                min_size,
                          refine_func_t               refine );

//
// compute DAG for coarse version of LU with on-the-fly DAGs for small matrices
//
template < typename value_t >
graph
gen_dag_lu_oop_coarse  ( Hpro::TMatrix< value_t > &    A,
                         const size_t                  ncoarse,
                         refine_func_t                 refine,
                         exec_func_t                   fine_run );

//
// level wise computation of DAG for LU of <A>
// - recurse on diagonal blocks and perform per-level LU on global scope 
//
template < typename value_t >
graph
gen_dag_lu_lvl    ( Hpro::TMatrix< value_t > &  A,
                    const size_t                min_size );

//
// generate DAG for Tile-H LU factorisation
//
template < typename value_t >
graph
gen_dag_lu_tileh ( Hpro::TMatrix< value_t > &  A,
                   const size_t                min_size,
                   refine_func_t               refine,
                   exec_func_t                 exec );

//
// compute DAG for solving L X = A with lower-triangular L
//
template < typename value_t >
graph
gen_dag_solve_lower  ( const Hpro::TMatrix< value_t > &  L,
                       Hpro::TMatrix< value_t > &        A,
                       const size_t                      min_size,
                       refine_func_t                     refine );

//
// compute DAG for solving X U = A with upper triangular U
//
template < typename value_t >
graph
gen_dag_solve_upper  ( const Hpro::TMatrix< value_t > &  U,
                       Hpro::TMatrix< value_t > &        A,
                       const size_t                      min_size,
                       refine_func_t                     refine );

//
// compute DAG for C = A B + C
//
template < typename value_t >
graph
gen_dag_update       ( const Hpro::TMatrix< value_t > &  A,
                       const Hpro::TMatrix< value_t > &  B,
                       Hpro::TMatrix< value_t > &        C,
                       const size_t                      min_size,
                       refine_func_t                     refine );

//
// compute DAG for TSQR( X·T, U )
//
template < typename value_t >
graph
gen_dag_tsqr ( const size_t                                               n,
               hlr::matrix::tile_storage< value_t > &                     X,
               std::shared_ptr< blas::matrix< value_t > > &               T,
               hlr::matrix::tile_storage< value_t > &                     U,
               std::shared_ptr< hlr::matrix::tile_storage< value_t > > &  Q,
               std::shared_ptr< blas::matrix< value_t > > &               R,
               refine_func_t                                              refine );

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
template < typename value_t >
graph
gen_dag_truncate ( hlr::matrix::tile_storage< value_t > &        X,
                   std::shared_ptr< blas::matrix< value_t > > &  T,
                   hlr::matrix::tile_storage< value_t > &        Y,
                   hlr::matrix::tiled_lrmatrix< value_t > *      A,
                   refine_func_t                                 refine );

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
template < typename value_t >
graph
gen_dag_addlr ( hlr::matrix::tile_storage< value_t > &        X,
                std::shared_ptr< blas::matrix< value_t > > &  T,
                hlr::matrix::tile_storage< value_t > &        Y,
                Hpro::TMatrix< value_t > *                    A,
                refine_func_t                                 refine );

}}// namespace hlr::dag

#endif // __HLR_DAG_LU_HH
