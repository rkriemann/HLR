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
    return std::move( v.sub_vector( is ) );
}

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_v = 'v';

struct solve_upper_node : public node
{
    const matop_t    op_U;
    const TMatrix *  U;
    TScalarVector    v;
    mtx_vec_t &      chunk_mtx;
    
    solve_upper_node ( const matop_t     aop_U,
                       const TMatrix *   aU,
                       TScalarVector &&  av,
                       mtx_vec_t &       achunk_mtx)
            : op_U( aop_U )
            , U( aU )
            , v( std::move( av ) )
            , chunk_mtx( achunk_mtx )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "solve_U( %d, ", U->id() ) + v.is().to_string() + " )"; }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_v, vec_bis( v ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( v ) } }; }
};

struct solve_lower_node : public node
{
    const matop_t    op_L;
    const TMatrix *  L;
    TScalarVector    v;
    mtx_vec_t &      chunk_mtx;

    solve_lower_node ( const matop_t     aop_L,
                       const TMatrix *   aL,
                       TScalarVector &&  av,
                       mtx_vec_t &       achunk_mtx)
            : op_L( aop_L )
            , L( aL )
            , v( std::move( av ) )
            , chunk_mtx( achunk_mtx )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "solve_L( %d, ", L->id() ) + v.is().to_string() + " )"; }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
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
    mtx_vec_t &      chunk_mtx;

    mul_vec_node ( const value_t     aalpha,
                   const matop_t     aop_A,
                   const TMatrix *   aA,
                   TScalarVector &&  ax,
                   TScalarVector &&  ay,
                   mtx_vec_t &       achunk_mtx)
            : alpha( aalpha )
            , op_A( aop_A )
            , A( aA )
            , x( std::move( ax ) )
            , y( std::move( ay ) )
            , chunk_mtx( achunk_mtx )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "mul_vec( %d, " ) + x.is().to_string() + ", " + y.is().to_string() + " )"; }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_v, vec_bis( x ) } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_v, vec_bis( y ) } }; }
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

    if ( is_blocked( L ) && ! hlr::is_small( min_size, L ) )
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
                    hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, sub_vector( v, L_ii->col_is() ), chunk_mtx );
                }// if
            
                //
                // update RHS with currently solved vector block
                //

                for ( uint j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );
                
                    if ( ! is_null( L_ji ) )
                    {
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_L, L_ji,
                                                                      sub_vector( v, L_ji->col_is() ),
                                                                      sub_vector( v, L_ji->row_is() ),
                                                                      chunk_mtx );
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
                    hlr::dag::alloc_node< solve_lower_node >( g, op_L, L_ii, sub_vector( v, L_ii->row_is() ), chunk_mtx );
                }// if

                //
                // update RHS
                //

                for ( int j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_L, L_ij,
                                                                      sub_vector( v, L_ij->row_is() ),
                                                                      sub_vector( v, L_ij->col_is() ),
                                                                      chunk_mtx );
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
    HLR_LOG( 2, HLIB::to_string( "trsvl( %d )", L->id() ) );
    
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
    hlr::seq::trsvl( op_L, * L, v, unit_diag );
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

    if ( is_blocked( U ) && ! hlr::is_small( min_size, U ) )
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
                    hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, sub_vector( v, U_ii->col_is() ), chunk_mtx );
                }// if

                //
                // update RHS with currently solved vector block
                //

                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );
                    
                    if ( ! is_null( U_ji ) )
                    {
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_U, U_ji,
                                                                      sub_vector( v, U_ji->col_is() ),
                                                                      sub_vector( v, U_ji->row_is() ),
                                                                      chunk_mtx );
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
                    hlr::dag::alloc_node< solve_upper_node >( g, op_U, U_ii, sub_vector( v, U_ii->row_is() ), chunk_mtx );
                }// if

                //
                // update RHS with currently solved vector block
                //

                for ( uint j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        hlr::dag::alloc_node< mul_vec_node< real > >( g, -1, op_U, U_ij,
                                                                      sub_vector( v, U_ij->row_is() ),
                                                                      sub_vector( v, U_ij->col_is() ),
                                                                      chunk_mtx );
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
    HLR_LOG( 2, HLIB::to_string( "trsvu( %d )", U->id() ) );
    
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
    hlr::seq::trsvu( op_U, * U, v, general_diag );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

//
// apply y' to y chunkwise while only locking currently updated chunk
//
void
update ( const TScalarVector &  t,
         TScalarVector &        y,
         mtx_vec_t &            chunk_mtx )
{
    const idx_t  ofs_loc_glo = t.is().first();
    idx_t        start_idx   = ofs_loc_glo;
    idx_t        chunk       = start_idx / CHUNK_SIZE;
    const idx_t  last_idx    = t.is().last();
    idx_t        end_idx     = std::min< idx_t >( (chunk+1) * CHUNK_SIZE - 1, last_idx );

    while ( start_idx <= end_idx )
    {
        const B::Range        is_chunk( start_idx, end_idx );
        B::Vector< value_t >  y_glo( blas_vec< value_t >( _data.y ), is_chunk );
        B::Vector< value_t >  y_loc( yc, is_chunk - ofs_loc_glo );

        {
            std::scoped_lock  lock( chunk_mtx[ chunk ] );
                
            B::add( value_t(1), y_loc, y_glo );
        }

        ++chunk;
        start_idx = end_idx + 1;
        end_idx   = std::min< idx_t >( end_idx + CHUNK_SIZE, last_idx );
    }// while
}

template < typename value_t >
local_graph
mul_vec_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! hlr::is_small( min_size, A ) )
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
                    hlr::dag::alloc_node< mul_vec_node< real > >( g, alpha, op_A, A_ij,
                                                                  sub_vector( x, A_ij->col_is( op_A ) ),
                                                                  sub_vector( y, A_ij->row_is( op_A ) ),
                                                                  chunk_mtx );
                }// if
            }// for
        }// for
    }// if

    return g;
}

template < typename value_t >
void
mul_vec_node< value_t >::run_ ( const TTruncAcc & )
{
    HLR_LOG( 2, HLIB::to_string( "update( %d, ", A->id() ) + x.is().to_string() + ", " + y.is().to_string() );

    TScalarVector t( y.is() );
    
    A->apply_add( alpha, & x, & t, op_A );

    update( t, y, chunk_mtx );
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
gen_dag_solve_lower ( const matop_t    op_L,
                      TMatrix *        L,
                      TScalarVector &  x,
                      refine_func_t    refine )
{
    return refine( new solve_lower_node( op_L, L, x.sub_vector( x.is() ) ), 1000 );
}

graph
gen_dag_solve_upper ( const matop_t    op_U,
                      TMatrix *        U,
                      TScalarVector &  x,
                      refine_func_t    refine )
{
    return refine( new solve_upper_node( op_U, U, x.sub_vector( x.is() ) ), 1000 );
}

}// namespace dag

}// namespace hlr
