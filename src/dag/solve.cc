//
// Project     : HLib
// File        : solve.cc
// Description : DAGs for matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <matrix/structure.hh>
#include <algebra/solve_tri.hh>
#include <algebra/mat_mul.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/matrix/level_matrix.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/arith.hh"

namespace hlr
{

using namespace HLIB;

namespace dag
{

namespace
{

// convert index set <is> into block index set { is, {0} }
TBlockIndexSet
vec_bis ( const TScalarVector &  v )
{
    return bis( v.is(), TIndexSet( 0, 0 ) );
}

// return sub vector of v corresponding to is
TScalarVector
sub_vector ( TScalarVector &    v,
             const TIndexSet &  is )
{
    return v.sub_vector( is );
}

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_v = 'v';

struct solve_upper_node : public node
{
    const matop_t    op_U;
    const TMatrix *  U;
    TScalarVector    v;
    
    solve_upper_node ( const matop_t     aop_U,
                       const TMatrix *   aU,
                       TScalarVector     av )
            : op_U( aop_U )
            , U( aU )
            , v( av )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "solve_U( %d, ", U->id() ) + v.is().to_string() + " )"; }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_v, vec_bis( v ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( v ) } }; }
};

struct solve_lower_node : public node
{
    const matop_t    op_L;
    const TMatrix *  L;
    TScalarVector    v;

    solve_lower_node ( const matop_t     aop_L,
                       const TMatrix *   aL,
                       TScalarVector     av )
            : op_L( aop_L )
            , L( aL )
            , v( av )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "solve_L( %d, ", L->id() ) + v.is().to_string() + " )"; }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, L->block_is() }, { id_v, vec_bis( v ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( v ) } }; }
};

template < typename value_t >
struct mul_vec_node : public node
{
    const value_t    alpha;
    const matop_t    op_A;
    const TMatrix *  A;
    TScalarVector    x;
    TScalarVector    y;

    mul_vec_node ( const value_t     aalpha,
                   const matop_t     aop_A,
                   const TMatrix *   aA,
                   TScalarVector     ax,
                   TScalarVector     ay )
            : alpha( aalpha )
            , op_A( aop_A )
            , A( aA )
            , x( ax )
            , y( ay )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "mul_vec( %d, " ) + x.is().to_string() + ", " + y.is().to_string() + " )"; }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ();
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_v, vec_bis( x ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( y ) } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_lower_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
solve_lower_node::refine_ ()
{
    local_graph  g;

    if ( is_blocked( L ) && ! is_small( L ) )
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
                    TScalarVector  v_i( sub_vector( v, L_ii->col_is() ) );
                
                    hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, v_i );
                }// if
            
                //
                // update RHS with currently solved vector block
                //

                for ( uint j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );
                
                    if ( ! is_null( L_ji ) )
                    {
                        TScalarVector  v_j( sub_vector( v, L_ji->row_is() ) );
                        TScalarVector  v_i( sub_vector( v, L_ji->col_is() ) );
                    
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_L, L_ji, v_i, v_j );
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
                    TScalarVector  v_i( sub_vector( v, L_ii->row_is() ) );
                
                    hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, v_i );
                }// if

                //
                // update RHS
                //

                for ( int j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        TScalarVector  v_i( sub_vector( v, L_ij->col_is() ) );
                        TScalarVector  v_j( sub_vector( v, L_ij->row_is() ) );
                                   
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_L, L_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if

    return g;
}

void
solve_lower_node::run_ ( const TTruncAcc &  acc )
{
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
    hlr::seq::trsvl( op_L, * L, v, unit_diag );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_upper_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
solve_upper_node::refine_ ()
{
    local_graph  g;

    if ( is_blocked( U ) && ! is_small( U ) )
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
                    TScalarVector  v_i( sub_vector( v, U_ii->col_is() ) );
                
                    hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, v_i );
                }// if

                //
                // update RHS with currently solved vector block
                //

                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );
                    
                    if ( ! is_null( U_ji ) )
                    {
                        TScalarVector  v_j( sub_vector( v, U_ji->row_is() ) );
                        TScalarVector  v_i( sub_vector( v, U_ji->col_is() ) );
                                   
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_U, U_ji, v_i, v_j );
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
                    TScalarVector  v_i( sub_vector( v, U_ii->row_is() ) );
                
                    hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, v_i );
                }// if

                //
                // update RHS with currently solved vector block
                //

                for ( uint j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        TScalarVector  v_i( sub_vector( v, U_ij->col_is() ) );
                        TScalarVector  v_j( sub_vector( v, U_ij->row_is() ) );
                                   
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_U, U_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if

    return g;
}

void
solve_upper_node::run_ ( const TTruncAcc &  acc )
{
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    hlr::seq::trsvu( op_U, * U, v, general_diag );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
mul_vec_node< value_t >::refine_ ()
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( A ) )
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
                    TScalarVector  x_j( sub_vector( x, A_ij->col_is( op_A ) ) );
                    TScalarVector  y_i( sub_vector( y, A_ij->row_is( op_A ) ) );
                                   
                    hlr::dag::alloc_node< mul_vec_node< real > >( g, alpha, op_A, A_ij, x_j, y_i );
                }// if
            }// for
        }// for
    }// if

    return g;
}

template < typename value_t >
void
mul_vec_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    A->apply_add( alpha, & x, & y, op_A );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public functions to generate DAGs
//
///////////////////////////////////////////////////////////////////////////////////////

graph
gen_dag_solve_lower ( const matop_t                      op_L,
                      TMatrix *                          L,
                      TScalarVector                      x,
                      std::function< graph ( node * ) >  refine )
{
    return refine( new solve_lower_node( op_L, L, x ) );
}

graph
gen_dag_solve_upper ( const matop_t                      op_U,
                      TMatrix *                          U,
                      TScalarVector                      x,
                      std::function< graph ( node * ) >  refine )
{
    return refine( new solve_upper_node( op_U, U, x ) );
}

}// namespace dag

}// namespace hlr
