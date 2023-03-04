//
// Project     : HLR
// Module      : gauss_elim.cc
// Description : generate DAG for Gaussian elimination
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <cassert>

#include <hpro/matrix/structure.hh>

#include "hlr/arith/norm.hh"
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

using id_t = Hpro::id_t;

// memory block identifiers
const id_t  ID_A('A');
const id_t  ID_C('C');

//
// computes C = A^-1
//
template < typename value_t >
struct gauss_node : public node
{
    TMatrix< value_t > *   A;
    const id_t  id_A;
    const id_t  id_C;
    TMatrix< value_t > *   T;
    
    gauss_node ( TMatrix< value_t > *  aA,
                 const id_t            aid_A,
                 const id_t            aid_C,
                 TMatrix< value_t > *  aT )
            : A( aA )
            , id_A( aid_A )
            , id_C( aid_C )
            , T( aT )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "g( %c/%d, %c/%d )", char(id_A), A->id(), char(id_C), A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_C, A->block_is() } }; }
};

//
// computes C = β C + α A·B
//
template < typename value_t >
struct update_node : public node
{
    const value_t               alpha;
    const TMatrix< value_t > *  A;
    const id_t                  id_A;
    const TMatrix< value_t > *  B;
    const id_t                  id_B;
    const value_t               beta;
    TMatrix< value_t > *        C;
    const id_t                  id_C;

    update_node ( const value_t                 aalpha,
                  const TMatrix< value_t > * aA,
                  const id_t                 aid_A,
                  const TMatrix< value_t > * aB,
                  const id_t                 aid_B,
                  const value_t                 abeta,
                  TMatrix< value_t > *       aC,
                  const id_t                 aid_C )
            : alpha( aalpha )
            , A( aA )
            , id_A( aid_A )
            , B( aB )
            , id_B( aid_B )
            , beta( abeta )
            , C( aC )
            , id_C( aid_C )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "u( %c/%d, %c/%d, %c/%d )",
                                                                      char(id_A), A->id(), char(id_B), B->id(), char(id_C), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_B, B->block_is() }, { id_C, C->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_C, C->block_is() } }; }
};

//
// assign B = A
//
template < typename value_t >
struct copy_node : public node
{
    const TMatrix< value_t > *  A;
    const id_t                  id_A;
    TMatrix< value_t > *        B;
    const id_t                  id_B;

    copy_node ( const TMatrix< value_t > * aA,
                const id_t                 aid_A,
                TMatrix< value_t > *       aB,
                const id_t                 aid_B )
            : A( aA )
            , id_A( aid_A )
            , B( aB )
            , id_B( aid_B )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "copy( %c%d, %c%d )",
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

std::atomic< int >  ID_TEMP( 100 );

template < typename value_t >
local_graph
gauss_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix< value_t > );
        auto  BT = ptrcast( T, TBlockMatrix< value_t > );
        auto  MA = [BA] ( const uint  i, const uint  j ) { return BA->block( i, j ); };
        auto  MT = [BT] ( const uint  i, const uint  j ) { return BT->block( i, j ); };

        id_t  id_temp = ID_TEMP++;
            
        // T_00 = A_00⁻¹
        auto  inv_a00  = hlr::dag::alloc_node< gauss_node >( g, MA(0,0), id_A, id_temp, MT(0,0) );

        // T_01 = T_00 · A_01
        auto  upd_t01  = hlr::dag::alloc_node< update_node >( g, 1.0, MA(0,0), id_temp, MA(0,1), id_A, 0.0, MT(0,1), id_temp );
        upd_t01->after( inv_a00 );

        // T_10 = A_10 · T_00
        auto  upd_t10  = hlr::dag::alloc_node< update_node >( g, 1.0, MA(1,0), id_A, MA(0,0), id_temp, 0.0, MT(1,0), id_temp );
        upd_t10->after( inv_a00 );

        // A_11 = A_11 - T_10 · A_01
        auto  upd_a11  = hlr::dag::alloc_node< update_node >( g, -1.0, MT(1,0), id_temp, MA(0,1), id_A, 1.0, MA(1,1), id_A );
        upd_a11->after( upd_t10 );
    
        // C_11 = A_11⁻¹
        auto  inv_a11  = hlr::dag::alloc_node< gauss_node >( g, MA(1,1), id_A, id_C, MT(1,1) );
        inv_a11->after( upd_a11 );

        // C_01 = - T_01 · C_11
        auto  upd_a01  = hlr::dag::alloc_node< update_node >( g, -1.0, MT(0,1), id_temp, MA(1,1), id_C, 0.0, MA(0,1), id_C );
        upd_a01->after( upd_t01 );
        upd_a01->after( inv_a11 );
            
        // C_10 = - C_11 · T_10
        auto  upd_a10  = hlr::dag::alloc_node< update_node >( g, -1.0, MA(1,1), id_C, MT(1,0), id_temp, 0.0, MA(1,0), id_C );
        upd_a10->after( inv_a11 );
        upd_a10->after( upd_t10 );
        
        // C_00 = A_00 - C_01 · T_10
        auto  upd_a00  = hlr::dag::alloc_node< update_node >( g, -1.0, MA(0,1), id_C, MT(1,0), id_temp, 1.0, MA(0,0), id_C );
        upd_a00->after( upd_a01 );
        upd_a00->after( upd_t10 );
        upd_a00->after( inv_a00 );
        
        g.finalize();
    }// if

    return g;
}

template < typename value_t >
void
gauss_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    hlr::seq::gauss_elim( *A, *T, acc );
    
    hlr::log( 0, Hpro::to_string( "                               %d = %.8e", A->id(), norm::frobenius( *A ) ) );
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
    
    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix< value_t > );
        auto  BB = cptrcast( B, TBlockMatrix< value_t > );
        auto  BC = ptrcast(  C, TBlockMatrix< value_t > );

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

template < typename value_t >
void
update_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    hlr::log( 0, Hpro::to_string( "                               %d = %.8e, %d = %.8e, %d = %.8e" ,
                                  A->id(), norm::frobenius( *A ),
                                  B->id(), norm::frobenius( *B ),
                                  C->id(), norm::frobenius( *C ) ) );

    HLR_ERROR( "todo" );
    
    // multiply( alpha, apply_normal, A, apply_normal, B, beta, C, acc );
    hlr::log( 0, Hpro::to_string( "                               %d = %.8e",
                                  C->id(), norm::frobenius( *C ) ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// CopyToNode
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
copy_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, B ) && ! is_small_any( min_size, A, B ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix< value_t > );
        auto  BB = ptrcast(  B, TBlockMatrix< value_t > );

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

template < typename value_t >
void
copy_node< value_t >::run_ ( const TTruncAcc & )
{
    // A->copy_to( B );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function for DAG generation
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
dag::graph
gen_dag_gauss_elim ( TMatrix< value_t > *  A,
                     TMatrix< value_t > *  T,
                     refine_func_t         refine )
{
    return refine( new gauss_node( A, ID_A, ID_C, T ), Hpro::CFG::Arith::max_seq_size, use_single_end_node );
}

}}// namespace hlr::dag
