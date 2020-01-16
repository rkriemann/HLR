//
// Project     : HLib
// File        : solve.cc
// Description : DAGs for matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <map>

#include "hlr/utils/term.hh" // DEBUG
#include "hlr/utils/checks.hh"
#include "hlr/dag/solve.hh"
#include "hlr/seq/arith.hh"

namespace hlr
{

using namespace HLIB;

namespace dag
{

namespace
{

//////////////////////////////////////////////////////////////////////
//
// auxiliary functions
//
//////////////////////////////////////////////////////////////////////

// convert index set <is> into block index set { is, {0} }
// TBlockIndexSet
// vec_bis ( const TScalarVector &  v )
// {
//     return bis( v.is(), TIndexSet( 0, 0 ) );
// }

TBlockIndexSet
vec_bis ( const TIndexSet &  is )
{
    return bis( is, TIndexSet( 0, 0 ) );
}

// return sub vector of v corresponding to is
TScalarVector
sub_vec ( TScalarVector *    v,
          const TIndexSet &  is )
{
    return std::move( v->sub_vector( is ) );
}

TIndexSet
row_is ( const TMatrix *  A,
         const matop_t    op_A )
{
    if ( op_A == apply_normal ) return A->row_is();
    else                        return A->col_is();
}

TIndexSet
col_is ( const TMatrix *  A,
         const matop_t    op_A )
{
    if ( op_A == apply_normal ) return A->col_is();
    else                        return A->row_is();
}

//////////////////////////////////////////////////////////////////////
//
// DAG nodes
//
//////////////////////////////////////////////////////////////////////

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_v = 'v';

struct solve_upper_node : public node
{
    const matop_t     op_U;
    const TMatrix *   U;
    TScalarVector **  v; // global vector
    mutex_map_t &     mtx_map;
    
    solve_upper_node ( const matop_t     aop_U,
                       const TMatrix *   aU,
                       TScalarVector **  av,
                       mutex_map_t &     amtx_map)
            : op_U( aop_U )
            , U( aU )
            , v( av )
            , mtx_map( amtx_map )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "solve_U( %d )", U->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_v, vec_bis( row_is( U, op_U ) ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( row_is( U, op_U ) ) } }; }
};

struct solve_lower_node : public node
{
    const matop_t     op_L;
    const TMatrix *   L;
    TScalarVector **  v; // global vector
    mutex_map_t &     mtx_map;

    solve_lower_node ( const matop_t     aop_L,
                       const TMatrix *   aL,
                       TScalarVector **  av,
                       mutex_map_t &     amtx_map)
            : op_L( aop_L )
            , L( aL )
            , v( av )
            , mtx_map( amtx_map )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "solve_L( %d )", L->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, L->block_is() }, { id_v, vec_bis( row_is( L, op_L ) ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( row_is( L, op_L ) ) } }; }
};

template < typename value_t >
struct update_node : public node
{
    const value_t     alpha;
    const matop_t     op_A;
    const TMatrix *   A;
    TScalarVector **  v; // global vector
    mutex_map_t &     mtx_map;

    update_node ( const value_t     aalpha,
                   const matop_t     aop_A,
                   const TMatrix *   aA,
                   TScalarVector **  av,
                   mutex_map_t &     amtx_map)
            : alpha( aalpha )
            , op_A( aop_A )
            , A( aA )
            , v( av )
            , mtx_map( amtx_map )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "mul_vec( %d, " ) + col_is( A, op_A ).to_string() + ", " + row_is( A, op_A ).to_string() + " )"; }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_v, vec_bis( col_is( A, op_A ) ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( row_is( A, op_A ) ) } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_lower_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
solve_lower_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( L ) && ! is_small( min_size, L ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix );
        const auto  nbr = BL->nblock_rows();
        const auto  nbc = BL->nblock_cols();

        if ( op_L == apply_normal )
        {
            for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
            {
                //
                // solve diagonal block
                //

                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    // hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, sub_vec( v, L_ii->col_is() ), mtx_map );
                    hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, v, mtx_map );
                }// if
            
                //
                // update RHS with currently solved vector block
                //

                for ( uint j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );
                
                    if ( ! is_null( L_ji ) )
                    {
                        // hlr::dag::alloc_node< update_node< real > >( g, -1, op_L, L_ji,
                        //                                               sub_vec( v, L_ji->col_is() ),
                        //                                               sub_vec( v, L_ji->row_is() ),
                        //                                               mtx_map );
                        hlr::dag::alloc_node< update_node< real > >( g, -1, op_L, L_ji, v, mtx_map );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            //
        
            for ( int  i = std::min< int >( nbr, nbc )-1; i >= 0; --i )
            {
                //
                // solve diagonal block
                //

                auto  L_ii = BL->block( i, i );
                
                if ( ! is_null( L_ii ) )
                {
                    // hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, sub_vec( v, L_ii->row_is() ), mtx_map );
                    hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, v, mtx_map );
                }// if

                //
                // update RHS
                //

                for ( int j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        // hlr::dag::alloc_node< update_node< real > >( g, -1, op_L, L_ij,
                        //                                               sub_vec( v, L_ij->row_is() ),
                        //                                               sub_vec( v, L_ij->col_is() ),
                        //                                               mtx_map );
                        hlr::dag::alloc_node< update_node< real > >( g, -1, op_L, L_ij, v, mtx_map );
                    }// if
                }// for
            }// for
        }// else
    }// if

    return g;
}

void
solve_lower_node::run_ ( const TTruncAcc & )
{
    HLR_LOG( 4, HLIB::to_string( "trsvl( %d )", L->id() ) );
    
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
    TScalarVector  v_l( std::move( sub_vec( *v, row_is( L, op_L ) ) ) );
    
    hlr::seq::trsvl( op_L, * L, v_l, unit_diag );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_upper_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
solve_upper_node::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( U ) && ! is_small( min_size, U ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        const auto  nbr = BU->nblock_rows();
        const auto  nbc = BU->nblock_cols();

        if ( op_U == apply_normal )
        {
            for ( int  i = std::min< int >(nbr,nbc)-1; i >= 0; --i )
            {
                //
                // solve diagonal block
                //

                auto  U_ii = BU->block( i, i );
                
                if ( ! is_null( U_ii ) )
                {
                    // hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, sub_vec( v, U_ii->col_is() ), mtx_map );
                    hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, v, mtx_map );
                }// if

                //
                // update RHS with currently solved vector block
                //

                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );
                    
                    if ( ! is_null( U_ji ) )
                    {
                        // hlr::dag::alloc_node< update_node< real > >( g, -1, op_U, U_ji,
                        //                                              sub_vec( v, U_ji->col_is() ),
                        //                                              sub_vec( v, U_ji->row_is() ),
                        //                                              mtx_map );
                        hlr::dag::alloc_node< update_node< real > >( g, -1, op_U, U_ji, v, mtx_map );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve from top to bottom
            //
        
            for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
            {
                //
                // solve diagonal block
                //
            
                auto  U_ii = BU->block( i, i );
                
                if ( ! is_null( U_ii ) )
                {
                    // hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, sub_vec( v, U_ii->row_is() ), mtx_map );
                    hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, v, mtx_map );
                }// if

                //
                // update RHS with currently solved vector block
                //

                for ( uint j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        // hlr::dag::alloc_node< update_node< real > >( g, -1, op_U, U_ij,
                        //                                              sub_vec( v, U_ij->row_is() ),
                        //                                              sub_vec( v, U_ij->col_is() ),
                        //                                              mtx_map );
                        hlr::dag::alloc_node< update_node< real > >( g, -1, op_U, U_ij, v, mtx_map );
                    }// if
                }// for
            }// for
        }// else
    }// if

    return g;
}

void
solve_upper_node::run_ ( const TTruncAcc & )
{
    HLR_LOG( 4, HLIB::to_string( "trsvu( %d )", U->id() ) );
    
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    TScalarVector  v_u( std::move( sub_vec( *v, row_is( U, op_U ) ) ) );
        
    hlr::seq::trsvu( op_U, * U, v_u, general_diag );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
update_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix );

        for ( uint  i = 0; i < BA->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->block_cols(); ++j )
            {
                auto  A_ij = BA->block( i, j );
                
                if ( ! is_null( A_ij ) )
                {
                    // hlr::dag::alloc_node< update_node< real > >( g, alpha, op_A, A_ij,
                    //                                              sub_vec( x, A_ij->col_is( op_A ) ),
                    //                                              sub_vec( y, A_ij->row_is( op_A ) ),
                    //                                              mtx_map );
                    hlr::dag::alloc_node< update_node< real > >( g, alpha, op_A, A_ij, v, mtx_map );
                }// if
            }// for
        }// for
    }// if

    return g;
}

//
// apply t to y in chunks of size CHUNK_SIZE
// while only locking currently updated chunk
//
void
update ( const TScalarVector &  t,
         TScalarVector &        y,
         mutex_map_t &          mtx_map )
{
    idx_t        start_idx   = t.is().first();
    const idx_t  last_idx    = t.is().last();
    idx_t        chunk       = start_idx / CHUNK_SIZE;
    idx_t        end_idx     = std::min< idx_t >( (chunk+1) * CHUNK_SIZE - 1, last_idx );

    while ( start_idx <= end_idx )
    {
        const TIndexSet  chunk_is( start_idx, end_idx );
        auto             t_i = t.sub_vector( chunk_is );
        auto             y_i = y.sub_vector( chunk_is );

        {
            // std::cout << term::on_red( HLIB::to_string( "locking %d", chunk ) ) << std::endl;
            std::scoped_lock  lock( * mtx_map[ chunk ] );
                
            y_i.axpy( real(1), & t_i );
        }

        ++chunk;
        start_idx = end_idx + 1;
        end_idx   = std::min< idx_t >( end_idx + CHUNK_SIZE, last_idx );
    }// while
}

template < typename value_t >
void
update_node< value_t >::run_ ( const TTruncAcc & )
{
    HLR_LOG( 4, HLIB::to_string( "update( %d )", A->id() ) );

    TScalarVector  t( row_is( A, op_A ), A->value_type() );
    TScalarVector  x( std::move( sub_vec( *v, col_is( A, op_A ) ) ) );
    TScalarVector  y( std::move( sub_vec( *v, row_is( A, op_A ) ) ) );
    
    A->apply_add( alpha, & x, & t, op_A );

    update( t, y, mtx_map );

    // x.axpy( 1.0, & t );
    
    // A->apply_add( alpha, & x, & y, op_A );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public functions to generate DAGs
//
///////////////////////////////////////////////////////////////////////////////////////

graph
gen_dag_solve_lower ( const matop_t     op_L,
                      TMatrix *         L,
                      TScalarVector **  v,
                      mutex_map_t &     mtx_map,
                      refine_func_t     refine )
{
    // return refine( new solve_lower_node( op_L, L, x.sub_vec( x.is() ), mtx_map ), 256 );
    return refine( new solve_lower_node( op_L, L, v, mtx_map ), 128, use_single_end_node );
}

graph
gen_dag_solve_upper ( const matop_t     op_U,
                      TMatrix *         U,
                      TScalarVector **  v,
                      mutex_map_t &     mtx_map,
                      refine_func_t     refine )
{
    // return refine( new solve_upper_node( op_U, U, x.sub_vec( x.is() ), mtx_map ), 256 );
    return refine( new solve_upper_node( op_U, U, v, mtx_map ), 128, use_single_end_node );
}

}// namespace dag

}// namespace hlr
