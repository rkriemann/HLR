#ifndef __HLR_DAG_DETAIL_LU_HH
#define __HLR_DAG_DETAIL_LU_HH
//
// Project     : HLib
// Module      : dag/lu
// Description : nodes for DAG based LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>

#include "hlr/dag/graph.hh"
#include "hlr/arith/add.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/lu.hh"
#include "hlr/arith/solve.hh"
#include "hlr/utils/tensor.hh"

namespace hlr { namespace dag { namespace lu {

////////////////////////////////////////////////////////////////////////////////
//
// immediate update version
//
////////////////////////////////////////////////////////////////////////////////

// identifiers for memory blocks
constexpr Hpro::id_t  ID_A = 'A';
constexpr Hpro::id_t  ID_L = 'L';
constexpr Hpro::id_t  ID_U = 'U';

template < typename value_t,
           typename approx_t >
struct lu_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    
    lu_node ( Hpro::TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        const approx_t  apx;
    
        hlr::lu< value_t >( *A, acc, apx );
    }
    
    virtual local_graph  refine_  ( const size_t  min_size );
};

template < typename value_t,
           typename approx_t >
struct solve_upper_node : public node
{
    const Hpro::TMatrix< value_t > *  U;
    Hpro::TMatrix< value_t > *        A;
    
    solve_upper_node ( const Hpro::TMatrix< value_t > *  aU,
                       Hpro::TMatrix< value_t > *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "L%d = solve_upper( U%d, A%d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        const approx_t  apx;
    
        hlr::solve_upper_tri< value_t >( from_right, general_diag, *U, *A, acc, apx );
    }
    
    virtual local_graph  refine_  ( const size_t  min_size );
};

template < typename value_t,
           typename approx_t >
struct solve_lower_node : public node
{
    const Hpro::TMatrix< value_t > *  L;
    Hpro::TMatrix< value_t > *        A;

    solve_lower_node ( const Hpro::TMatrix< value_t > *  aL,
                       Hpro::TMatrix< value_t > *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "U%d = solve_lower( L%d, A%d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        const approx_t  apx;
        
        hlr::solve_lower_tri< value_t >( from_left, unit_diag, *L, *A, acc, apx );
    }

    virtual local_graph  refine_  ( const size_t  min_size );
};
    
template < typename value_t,
           typename approx_t >
struct update_node : public node
{
    const Hpro::TMatrix< value_t > *  A;
    const Hpro::TMatrix< value_t > *  B;
    Hpro::TMatrix< value_t > *        C;

    update_node ( const Hpro::TMatrix< value_t > *  aA,
                  const Hpro::TMatrix< value_t > *  aB,
                  Hpro::TMatrix< value_t > *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "A%d = mul( L%d, U%d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A,    C->block_is() } }; }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        const approx_t  apx;
    
        hlr::multiply( value_t(-1), apply_normal, *A, apply_normal, *B, *C, acc, apx );
    }
        
    virtual local_graph  refine_  ( const size_t  min_size );
};

///////////////////////////////////////////////////////////////////////////////////////
//
// node refinement
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
local_graph
lu_node< value_t, approx_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_nd( A ) && ! is_small( min_size, A ) )
    {
        auto        BA       = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BU       = BA;
        auto        BL       = BA;
        const auto  nbr      = BA->block_rows();
        const auto  nbc      = BA->block_cols();
        auto        finished = tensor2< node * >( nbr, nbc );

        for ( uint i = 0; i < std::min( nbr, nbc )-1; ++i )
        {
            auto  A_ii  = BA->block( i, i );
            auto  U_ii  = A_ii;
            auto  L_ii  = A_ii;

            assert( A_ii != nullptr );

            finished( i, i ) = g.alloc_node< lu_node< value_t, approx_t > >( A_ii );

            if ( ! is_null( BA->block( nbr-1, i ) ) )
            {
                finished( nbr-1, i ) = g.alloc_node< solve_upper_node< value_t, approx_t > >( U_ii, BA->block( nbr-1, i ) );
                finished( nbr-1, i )->after( finished( i, i ) );
            }// if

            if ( ! is_null( BA->block( i, nbc-1 ) ) )
            {
                finished( i, nbc-1 ) = g.alloc_node< solve_lower_node< value_t, approx_t > >( L_ii, BA->block( i, nbc-1 ) );
                finished( i, nbc-1 )->after( finished( i, i ) );
            }// if
        }// for
        
        finished( nbr-1, nbc-1 ) = g.alloc_node< lu_node >( BA->block( nbr-1, nbc-1 ) );
        
        for ( uint i = 0; i < std::min( nbr, nbc )-1; ++i )
        {
            if ( ! is_null_any( BL->block( nbr-1, i ), BU->block( i, nbc-1 ), BA->block( nbr-1, nbc-1 ) ) )
            {
                auto  update = g.alloc_node< update_node< value_t, approx_t > >( BL->block( nbr-1, i ),
                                                                                 BU->block( i, nbc-1 ),
                                                                                 BA->block( nbr-1, nbc-1 ) );
                
                update->after( finished( nbr-1, i ) );
                update->after( finished( i, nbc-1 ) );
                finished( nbr-1, nbc-1 )->after( update );
            }// if
        }// for
    }// if
    else if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BL  = BA;
        auto        BU  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();
        auto        finished = tensor2< node * >( nbr, nbc );
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            //
            // factorise diagonal block
            //
            
            auto  A_ii  = BA->block( i, i );
            auto  L_ii  = A_ii;
            auto  U_ii  = A_ii;

            HLR_ASSERT( ! is_null_any( A_ii, L_ii, U_ii ) );

            finished( i, i ) = g.alloc_node< lu_node< value_t, approx_t > >( A_ii );

            for ( uint j = i+1; j < nbr; j++ )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                {
                    finished( j, i ) = g.alloc_node< solve_upper_node< value_t, approx_t > >( U_ii, BA->block( j, i ) );
                    finished( j, i )->after( finished( i, i ) );
                }// if
            }// for

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    finished( i, j ) = g.alloc_node< solve_lower_node< value_t, approx_t > >( L_ii, BA->block( i, j ) );
                    finished( i, j )->after( finished( i, i ) );
                }// if
        }// for
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            for ( uint j = i+1; j < nbr; j++ )
            {
                for ( uint l = i+1; l < nbc; l++ )
                {
                    if ( ! is_null_any( BL->block( j, i ), BU->block( i, l ), BA->block( j, l ) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t, approx_t > >( BL->block( j, i ),
                                                                                         BU->block( i, l ),
                                                                                         BA->block( j, l ) );

                        update->after( finished( j, i ) );
                        update->after( finished( i, l ) );
                        finished( j, l )->after( update );
                    }// if
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           typename approx_t >
local_graph
solve_upper_node< value_t, approx_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, U ) && ! is_small_any( min_size, A, U ) )
    {
        auto        BU  = cptrcast( U, Hpro::TBlockMatrix< value_t > );
        auto        BA  = ptrcast(  A, Hpro::TBlockMatrix< value_t > );
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();

        auto        finished = tensor2< node * >( nbr, nbc );
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            HLR_ASSERT( ! is_null( U_jj ) );
 
            for ( uint i = 0; i < nbr; ++i )
            {
                auto  A_ij = BA->block( i, j );
                    
                if ( ! is_null( A_ij ) )
                    finished( i, j ) = g.alloc_node< solve_upper_node< value_t, approx_t > >( U_jj, A_ij );
            }// for
        }// for
        
        if ( is_nd( U ) )
        {
            for ( uint j = 0; j < nbc-1; ++j )
            {
                for ( uint i = 0; i < nbr; ++i )
                {
                    if ( ! is_null_any( BA->block( i, j ), BU->block( j, nbc-1 ), BA->block( i, nbc-1 ) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t, approx_t > >( BA->block( i, j ),
                                                                                         BU->block( j, nbc-1 ),
                                                                                         BA->block( i, nbc-1 ) );

                        update->after( finished(i,j) );
                        finished(i,nbc-1)->after( update );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            for ( uint j = 0; j < nbc; ++j )
            {
                for ( uint  k = j+1; k < nbc; ++k )
                    for ( uint  i = 0; i < nbr; ++i )
                        if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        {
                            auto  update = g.alloc_node< update_node< value_t, approx_t > >( BX->block( i, j ),
                                                                                             BU->block( j, k ),
                                                                                             BA->block( i, k ) );

                            update->after( finished( i, j ) );
                            finished( i, k )->after( update );
                        }// if
            }// for
        }// if
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           typename approx_t >
local_graph
solve_lower_node< value_t, approx_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, L ) && ! is_small_any( min_size, A, L ) )
    {
        auto        BL  = cptrcast( L, Hpro::TBlockMatrix< value_t > );
        auto        BA  = ptrcast(  A, Hpro::TBlockMatrix< value_t > );
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();

        auto        finished = tensor2< node * >( nbr, nbc );
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
                
            HLR_ASSERT( ! is_null( L_ii ) );
            
            for ( uint j = 0; j < nbc; ++j )
            {
                auto  A_ij = BA->block( i, j );
                
                if ( ! is_null( A_ij ) )
                    finished( i, j ) = g.alloc_node< solve_lower_node< value_t, approx_t > >(  L_ii, A_ij );
            }// for
        }// for

        if ( is_nd( L ) )
        {
            for ( uint j = 0; j < nbc-1; ++j )
            {
                for ( uint i = 0; i < nbr; ++i )
                {
                    if ( ! is_null_any( BL->block( nbr-1, i ), BA->block( i, j ), BA->block( nbr-1, j ) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t, approx_t > >( BL->block( nbr-1, i ),
                                                                                         BA->block( i, j ),
                                                                                         BA->block( nbr-1, j ) );

                        update->after( finished(i,j) );
                        finished(nbr-1,j)->after( update );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            for ( uint i = 0; i < nbr; ++i )
            {
                for ( uint  k = i+1; k < nbr; ++k )
                    for ( uint  j = 0; j < nbc; ++j )
                        if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                        {
                            auto  update = g.alloc_node< update_node< value_t, approx_t > >( BL->block( k, i ),
                                                                                             BX->block( i, j ),
                                                                                             BA->block( k, j ) );

                            update->after( finished( i, j ) );
                            finished( k, j )->after( update );
                        }// if
            }// for
        }// else
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           typename approx_t >
local_graph
update_node< value_t, approx_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  C, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node< value_t, approx_t > >( BA->block( i, k ),
                                                                          BB->block( k, j ),
                                                                          BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

}}}// namespace hlr::dag::lu

#endif // __HLR_DAG_DETAIL_LU_HH
