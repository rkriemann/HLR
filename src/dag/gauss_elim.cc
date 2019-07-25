//
// Project     : HLib
// File        : gauss_elim.cc
// Description : generate DAG for Gaussian elimination
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <matrix/structure.hh>
#include <algebra/mat_mul.hh>
#include <algebra/mat_inv.hh>

#include "hlr/seq/arith.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/gauss_elim.hh"

namespace hlr
{

using namespace HLIB;

namespace dag
{

namespace
{

using id_t = HLIB::id_t;

// memory block identifiers
const id_t  ID_A('A');
const id_t  ID_C('C');
const id_t  ID_T('T');

//
// computes C = A^-1
//
struct gauss_node : public node
{
    TMatrix *   A;
    const id_t  id_A;
    TMatrix *   T;
    const id_t  id_T;
    
    gauss_node ( TMatrix *   aA,
                 const id_t  aid_A,
                 TMatrix *   aT,
                 const id_t  aid_T )
            : A( aA )
            , id_A( aid_A )
            , T( aT )
            , id_T( aid_T )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "gauss( %c%d )", id_A, A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

//
// computes C = β C + α A·B
//
struct update_node : public node
{
    const real       alpha;
    const TMatrix *  A;
    const id_t       id_A;
    const TMatrix *  B;
    const id_t       id_B;
    const real       beta;
    TMatrix *        C;
    const id_t       id_C;

    update_node ( const real       aalpha,
                  const TMatrix *  aA,
                  const id_t       aid_A,
                  const TMatrix *  aB,
                  const id_t       aid_B,
                  const real       abeta,
                  TMatrix *        aC,
                  const id_t       aid_C )
            : alpha( aalpha )
            , A( aA )
            , id_A( aid_A )
            , B( aB )
            , id_B( aid_B )
            , beta( abeta )
            , C( aC )
            , id_C( aid_C )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "update( %c%d, %c%d, %c%d )",
                                                                      char(id_A), A->id(), char(id_B), B->id(), char(id_C), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_B, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_C, C->block_is() } }; }
};

//
// assign B = A
//
struct copy_node : public node
{
    const TMatrix *  A;
    const id_t       id_A;
    TMatrix *        B;
    const id_t       id_B;

    copy_node ( const TMatrix *  aA,
                const id_t       aid_A,
                TMatrix *        aB,
                const id_t       aid_B )
            : A( aA )
            , id_A( aid_A )
            , B( aB )
            , id_B( aid_B )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "copy( %c%d, %c%d )",
                                                                      char(id_A), A->id(), char(id_B), B->id() ); }
    virtual std::string  color     () const { return "75507b"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_B, B->block_is() } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// gauss_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
gauss_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );
        auto  BT = ptrcast( T, TBlockMatrix );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };
    
        // A_00 = A_00⁻¹
        auto  inv_a00  = hlr::dag::alloc_node< gauss_node >( g, MA(0,0), id_A, MT(0,0), id_T );

        // T_01 = A_00⁻¹ · A_01
        auto  upd_t01  = hlr::dag::alloc_node< update_node >( g, 1.0, MA(0,0), id_A, MA(0,1), id_A, 0.0, MT(0,1), id_T );
        upd_t01->after( inv_a00 );

        // T_10 = A_10 · A_00⁻¹
        auto  upd_t10  = hlr::dag::alloc_node< update_node >( g, 1.0, MA(1,0), id_A, MA(0,0), id_A, 0.0, MT(1,0), id_T );
        upd_t10->after( inv_a00 );

        // A_11 = A_11 - T_10 · A_01
        auto  upd_a11  = hlr::dag::alloc_node< update_node >( g, -1.0, MT(1,0), id_T, MA(0,1), id_A, 1.0, MA(1,1), id_A );
        upd_a11->after( upd_t10 );
    
        // A_11 = A_11⁻¹
        auto  inv_a11  = hlr::dag::alloc_node< gauss_node >( g, MA(1,1), id_A, MT(1,1), id_T );
        inv_a11->after( upd_a11 );

        // A_01 = - T_01 · A_11
        auto  upd_a01  = hlr::dag::alloc_node< update_node >( g, -1.0, MT(0,1), id_T, MA(1,1), id_A, 0.0, MA(0,1), id_A );
        upd_a01->after( upd_t01 );
        upd_a01->after( inv_a11 );
            
        // A_10 = - A_11 · T_10
        auto  upd_a10  = hlr::dag::alloc_node< update_node >( g, -1.0, MA(1,1), id_A, MT(1,0), id_T, 0.0, MA(1,0), id_A );
        upd_a10->after( inv_a11 );
        upd_a10->after( upd_t10 );
        
        // A_00 = A_00 - A_01 · T_10
        auto  upd_a00  = hlr::dag::alloc_node< update_node >( g, -1.0, MA(0,1), id_A, MT(1,0), id_T, 1.0, MA(0,0), id_A );
        upd_a00->after( upd_a01 );
        upd_a00->after( upd_t10 );
        upd_a00->after( inv_a00 );
        
        g.finalize();
    }// if

    return g;
}

void
gauss_node::run_ ( const TTruncAcc &  acc )
{
    hlr::seq::gauss_elim( A, T, acc );
    hlr::log( 0, HLIB::to_string( "M%d = %.4e", A->id(), HLIB::norm_F( A ) ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
update_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix );
        auto  BB = cptrcast( B, TBlockMatrix );
        auto  BC = ptrcast(  C, TBlockMatrix );

        for ( uint  i = 0; i < BC->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->block_cols(); ++j )
            {
                if ( BC->block( i, j ) == nullptr )
                    continue;
                
                for ( uint  k = 0; k < BA->block_cols(); ++k )
                {
                    if (( BA->block( i, k ) != nullptr ) && ( BB->block( k, j ) != nullptr ))
                        hlr::dag::alloc_node< update_node >( g,
                                                             alpha,
                                                             BA->block( i, k ), id_A,
                                                             BB->block( k, j ), id_B,
                                                             beta, BC->block( i, j ), id_C );
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
update_node::run_ ( const TTruncAcc &  acc )
{
    multiply( alpha, apply_normal, A, apply_normal, B, beta, C, acc );
    hlr::log( 0, HLIB::to_string( "M%d = %.4e", C->id(), HLIB::norm_F( C ) ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// CopyToNode
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
copy_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( A, B ) && ! is_small_any( min_size, A, B ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix );
        auto  BB = ptrcast(  B, TBlockMatrix );

        for ( uint  i = 0; i < BB->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BB->block_cols(); ++j )
            {
                if ( BB->block( i, j ) == nullptr )
                    continue;

                hlr::dag::alloc_node< copy_node >( g, BA->block( i, j ), id_A, BB->block( i, j ), id_B );
            }// for
        }// for
    }// if

    return g;
}

void
copy_node::run_ ( const TTruncAcc & )
{
    A->copy_to( B );
}

}// namespace anonymous

dag::graph
gen_dag_gauss_elim ( TMatrix *      A,
                     TMatrix *      C,
                     refine_func_t  refine )
{
    return refine( new gauss_node( A, ID_A, C, ID_T ), HLIB::CFG::Arith::max_seq_size );
}

}}// namespace hlr::dag
