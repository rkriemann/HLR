#ifndef __HLR_DAG_LU_HH
#define __HLR_DAG_LU_HH
//
// Project     : HLib
// Module      : dag/lu
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <set>
#include <tuple>

#include <hpro/matrix/TMatrix.hh>

#include <hlr/dag/graph.hh>
#include <hlr/dag/detail/lu.hh>
#include <hlr/dag/detail/lu_accu_eager.hh>
#include <hlr/dag/detail/lu_accu_lazy.hh>
#include <hlr/utils/tools.hh>

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
std::tuple< graph,
            std::unique_ptr< dag::lu::accu::lazy::accumulator_map_t< value_t > >,
            std::unique_ptr< std::mutex > >
gen_dag_lu_accu_lazy ( Hpro::TMatrix< value_t > & A,
                       const size_t               min_size,
                       refine_func_t              refine,
                       const approx_t &           /* apx */ )
{
    // generate DAG for shifting and applying updates
    auto  accu_map                   = std::make_unique< dag::lu::accu::lazy::accumulator_map_t< value_t > >();
    auto  accu_mtx                   = std::make_unique< std::mutex >();
    auto  [ apply_map, apply_nodes ] = lu::accu::lazy::build_apply_dag< value_t, approx_t >( & A, accu_map.get(), accu_mtx.get(), min_size );

    // construct H-LU DAG
    auto  dag = refine( new lu::accu::lazy::lu_node< value_t, approx_t >( & A, apply_map ), min_size, use_single_end_node );

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

    //
    // loop over apply nodes from top to bottom and remove nodes without updates
    //

    using  node_set_t = std::set< node * >;

    dag::node_list_t  work;
    node_set_t        deleted;
    auto              is_apply_node = [] ( node * node ) { return ( dynamic_cast< lu::accu::lazy::apply_node< value_t, approx_t > * >( node ) != nullptr ); };
    
    work.push_back( apply_map[ A.id() ] );

    while ( ! work.empty() )
    {
        dag::node_list_t  succ;
        
        while ( ! work.empty() )
        {
            auto  node = behead( work );

            if ( is_apply_node( node ) )
            {
                if ( node->dep_cnt() == 0 )
                {
                    for ( auto  out : node->successors() )
                    {
                        out->dec_dep_cnt();
                        
                        if ( is_apply_node( out ) )
                            succ.push_back( out );
                    }// for
                    
                    deleted.insert( node );
                }// if
            }// if
        }// while

        work = std::move( succ );
    }// while
    
    dag::node_list_t  nodes, start, end;

    for ( auto  node : dag.nodes() )
    {
        if ( contains( deleted, node ) )
        {
            delete node;
        }// if
        else
        {
            nodes.push_back( node );
            
            if ( node->dep_cnt() == 0 )
                start.push_back( node );

            if ( node->successors().empty() )
                end.push_back( node );
        }// else
    }// for
    
    return  { std::move( dag::graph( std::move( nodes ), std::move( start ), std::move( end ) ) ), std::move( accu_map ), std::move( accu_mtx ) };
    
    // return { std::move( dag ), accu_map };
}

template < typename value_t,
           typename approx_t >
std::tuple< graph,
            std::unique_ptr< dag::lu::accu::eager::accumulator_map_t< value_t > >,
            std::unique_ptr< std::mutex > >
gen_dag_lu_accu_eager ( Hpro::TMatrix< value_t > & A,
                        const size_t               min_size,
                        refine_func_t              refine,
                        const approx_t &           /* apx */ )
{
    // generate DAG for shifting and applying updates
    auto  accu_map                   = std::make_unique< dag::lu::accu::eager::accumulator_map_t< value_t > >();
    auto  accu_mtx                   = std::make_unique< std::mutex >();
    auto  [ apply_map, apply_nodes ] = lu::accu::eager::build_apply_dag< value_t, approx_t >( & A, accu_map.get(), accu_mtx.get(), min_size );

    // construct H-LU DAG
    auto  dag = refine( new lu::accu::eager::lu_node< value_t, approx_t >( & A, apply_map ), min_size, use_single_end_node );

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

    //
    // loop over apply nodes from top to bottom and remove nodes without updates
    //

    using  node_set_t = std::set< node * >;

    dag::node_list_t  work;
    node_set_t        deleted;
    auto              is_apply_node = [] ( node * node ) { return ( dynamic_cast< lu::accu::eager::apply_node< value_t, approx_t > * >( node ) != nullptr ); };
    
    work.push_back( apply_map[ A.id() ] );

    while ( ! work.empty() )
    {
        dag::node_list_t  succ;
        
        while ( ! work.empty() )
        {
            auto  node = behead( work );

            if ( is_apply_node( node ) )
            {
                if ( node->dep_cnt() == 0 )
                {
                    for ( auto  out : node->successors() )
                    {
                        out->dec_dep_cnt();
                        
                        if ( is_apply_node( out ) )
                            succ.push_back( out );
                    }// for
                    
                    deleted.insert( node );
                }// if
            }// if
        }// while

        work = std::move( succ );
    }// while
    
    dag::node_list_t  nodes, start, end;

    for ( auto  node : dag.nodes() )
    {
        if ( contains( deleted, node ) )
        {
            delete node;
        }// if
        else
        {
            nodes.push_back( node );
            
            if ( node->dep_cnt() == 0 )
                start.push_back( node );

            if ( node->successors().empty() )
                end.push_back( node );
        }// else
    }// for
    
    return  { std::move( dag::graph( std::move( nodes ), std::move( start ), std::move( end ) ) ), std::move( accu_map ), std::move( accu_mtx ) };
    
    // return { std::move( dag ), accu_map };
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
