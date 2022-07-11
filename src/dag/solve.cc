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

namespace hlr { namespace dag {

namespace
{

//////////////////////////////////////////////////////////////////////
//
// auxiliary functions
//
//////////////////////////////////////////////////////////////////////

// convert index set <is> into block index set { is, {0} }
// Hpro::TBlockIndexSet
// vec_bis ( const Hpro::TScalarVector< value_t > &  v )
// {
//     return bis( v.is(), Hpro::TIndexSet( 0, 0 ) );
// }

Hpro::TBlockIndexSet
vec_bis ( const Hpro::TIndexSet &  is )
{
    return bis( is, Hpro::TIndexSet( 0, 0 ) );
}

// return sub vector of v corresponding to is
template < typename value_t >
Hpro::TScalarVector< value_t >
sub_vec ( Hpro::TScalarVector< value_t > *    v,
          const Hpro::TIndexSet &  is )
{
    return v->sub_vector( is );
}

template < typename value_t >
Hpro::TIndexSet
row_is ( const Hpro::TMatrix< value_t > *  A,
         const matop_t    op_A )
{
    if ( op_A == apply_normal ) return A->row_is();
    else                        return A->col_is();
}

template < typename value_t >
Hpro::TIndexSet
col_is ( const Hpro::TMatrix< value_t > *  A,
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
const Hpro::id_t  id_A = 'A';
const Hpro::id_t  id_v = 'v';

template < typename value_t >
struct solve_upper_node : public node
{
    const matop_t     op_U;
    const Hpro::TMatrix< value_t > *   U;
    Hpro::TScalarVector< value_t > **  v; // global vector
    mutex_map_t &     mtx_map;
    
    solve_upper_node ( const matop_t     aop_U,
                       const Hpro::TMatrix< value_t > *   aU,
                       Hpro::TScalarVector< value_t > **  av,
                       mutex_map_t &     amtx_map)
            : op_U( aop_U )
            , U( aU )
            , v( av )
            , mtx_map( amtx_map )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "solve_U( %d )", U->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_v, vec_bis( row_is( U, op_U ) ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( row_is( U, op_U ) ) } }; }
};

template < typename value_t >
struct solve_lower_node : public node
{
    const matop_t     op_L;
    const Hpro::TMatrix< value_t > *   L;
    Hpro::TScalarVector< value_t > **  v; // global vector
    mutex_map_t &     mtx_map;

    solve_lower_node ( const matop_t     aop_L,
                       const Hpro::TMatrix< value_t > *   aL,
                       Hpro::TScalarVector< value_t > **  av,
                       mutex_map_t &     amtx_map)
            : op_L( aop_L )
            , L( aL )
            , v( av )
            , mtx_map( amtx_map )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "solve_L( %d )", L->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, L->block_is() }, { id_v, vec_bis( row_is( L, op_L ) ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( row_is( L, op_L ) ) } }; }
};

template < typename value_t >
struct update_node : public node
{
    const value_t     alpha;
    const matop_t     op_A;
    const Hpro::TMatrix< value_t > *   A;
    Hpro::TScalarVector< value_t > **  v; // global vector
    mutex_map_t &     mtx_map;

    update_node ( const value_t     aalpha,
                   const matop_t     aop_A,
                   const Hpro::TMatrix< value_t > *   aA,
                   Hpro::TScalarVector< value_t > **  av,
                   mutex_map_t &     amtx_map)
            : alpha( aalpha )
            , op_A( aop_A )
            , A( aA )
            , v( av )
            , mtx_map( amtx_map )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "mul_vec( %d, " ) + col_is( A, op_A ).to_string() + ", " + row_is( A, op_A ).to_string() + " )"; }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_v, vec_bis( col_is( A, op_A ) ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( row_is( A, op_A ) ) } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_lower_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
solve_lower_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( L ) && ! is_small( min_size, L ) )
    {
        auto        BL  = cptrcast( L, Hpro::TBlockMatrix< value_t > );
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
                        hlr::dag::alloc_node< update_node< value_t > >( g, -1, op_L, L_ji, v, mtx_map );
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
                        hlr::dag::alloc_node< update_node< value_t > >( g, -1, op_L, L_ij, v, mtx_map );
                    }// if
                }// for
            }// for
        }// else
    }// if

    return g;
}

template < typename value_t >
void
solve_lower_node< value_t >::run_ ( const Hpro::TTruncAcc & )
{
    HLR_LOG( 4, Hpro::to_string( "trsvl( %d )", L->id() ) );
    
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
    Hpro::TScalarVector< value_t >  v_l = sub_vec( *v, row_is( L, op_L ) );
    
    hlr::trsvl( op_L, * L, v_l, unit_diag );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_upper_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
solve_upper_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( U ) && ! is_small( min_size, U ) )
    {
        auto        BU  = cptrcast( U, Hpro::TBlockMatrix< value_t > );
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
                        hlr::dag::alloc_node< update_node< value_t > >( g, -1, op_U, U_ji, v, mtx_map );
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
                        hlr::dag::alloc_node< update_node< value_t > >( g, -1, op_U, U_ij, v, mtx_map );
                    }// if
                }// for
            }// for
        }// else
    }// if

    return g;
}

template < typename value_t >
void
solve_upper_node< value_t >::run_ ( const Hpro::TTruncAcc & )
{
    HLR_LOG( 4, Hpro::to_string( "trsvu( %d )", U->id() ) );
    
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    Hpro::TScalarVector< value_t >  v_u = sub_vec( *v, row_is( U, op_U ) );
        
    hlr::trsvu( op_U, * U, v_u, general_diag );
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

        auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );

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
                    hlr::dag::alloc_node< update_node< value_t > >( g, alpha, op_A, A_ij, v, mtx_map );
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
template < typename value_t >
void
update ( const Hpro::TScalarVector< value_t > &  t,
         Hpro::TScalarVector< value_t > &        y,
         mutex_map_t &          mtx_map )
{
    idx_t        start_idx   = t.is().first();
    const idx_t  last_idx    = t.is().last();
    idx_t        chunk       = start_idx / CHUNK_SIZE;
    idx_t        end_idx     = std::min< idx_t >( (chunk+1) * CHUNK_SIZE - 1, last_idx );

    while ( start_idx <= end_idx )
    {
        const Hpro::TIndexSet  chunk_is( start_idx, end_idx );
        auto             t_i = t.sub_vector( chunk_is );
        auto             y_i = y.sub_vector( chunk_is );

        {
            // std::cout << term::on_red( Hpro::to_string( "locking %d", chunk ) ) << std::endl;
            std::scoped_lock  lock( * mtx_map[ chunk ] );
                
            y_i.axpy( value_t(1), & t_i );
        }

        ++chunk;
        start_idx = end_idx + 1;
        end_idx   = std::min< idx_t >( end_idx + CHUNK_SIZE, last_idx );
    }// while
}

template < typename value_t >
void
update_node< value_t >::run_ ( const Hpro::TTruncAcc & )
{
    HLR_LOG( 4, Hpro::to_string( "update( %d )", A->id() ) );

    Hpro::TScalarVector< value_t >  t( row_is( A, op_A ), A->value_type() );
    Hpro::TScalarVector< value_t >  x( std::move( sub_vec( *v, col_is( A, op_A ) ) ) );
    Hpro::TScalarVector< value_t >  y( std::move( sub_vec( *v, row_is( A, op_A ) ) ) );
    
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

template < typename value_t >
graph
gen_dag_solve_lower ( const matop_t     op_L,
                      Hpro::TMatrix< value_t > *         L,
                      Hpro::TScalarVector< value_t > **  v,
                      mutex_map_t &     mtx_map,
                      refine_func_t     refine )
{
    // return refine( new solve_lower_node( op_L, L, x.sub_vec( x.is() ), mtx_map ), 256 );
    return refine( new solve_lower_node( op_L, L, v, mtx_map ), 128, use_single_end_node );
}

template < typename value_t >
graph
gen_dag_solve_upper ( const matop_t     op_U,
                      Hpro::TMatrix< value_t > *         U,
                      Hpro::TScalarVector< value_t > **  v,
                      mutex_map_t &     mtx_map,
                      refine_func_t     refine )
{
    // return refine( new solve_upper_node( op_U, U, x.sub_vec( x.is() ), mtx_map ), 256 );
    return refine( new solve_upper_node( op_U, U, v, mtx_map ), 128, use_single_end_node );
}

}}// namespace hlr::dag
