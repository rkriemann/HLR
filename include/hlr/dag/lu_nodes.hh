#ifndef __HLR_DAG_LU_NODES_HH
#define __HLR_DAG_LU_NODES_HH
//
// Project     : HLib
// Module      : dag/lu_nodes
// Description : nodes for DAG based LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include "hlr/dag/graph.hh"
#include "hlr/arith/add.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/lu.hh"
#include "hlr/arith/solve.hh"
#include "hlr/utils/tensor.hh"
#include "hlr/matrix/restrict.hh"

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

////////////////////////////////////////////////////////////////////////////////
//
// accumulated version
//
////////////////////////////////////////////////////////////////////////////////

namespace accu {

// identifiers for memory blocks
constexpr Hpro::id_t  ID_ACCU = 'X';

//
// local version of accumulator per matrix
// - handles direct updates and shifted down updates
//
template < typename value_t >
struct accumulator
{
    using  accumulator_map_t  = std::unordered_map< Hpro::id_t, accumulator >;

    //
    // represents an update, i.e., matrix product
    //
    struct update
    {
        const matop_t                     op_A;
        const Hpro::TMatrix< value_t > *  A;
        const matop_t                     op_B;
        const Hpro::TMatrix< value_t > *  B;
    };
    
    // represents set of updates
    using  update_list = std::list< update >;

    // computed updates
    std::unique_ptr< Hpro::TMatrix< value_t > >   matrix;
    std::mutex                                    mtx_matrix;

    // pending (recursive) updates
    update_list                                   pending;
    std::mutex                                    mtx_pending;

    //
    // ctors
    //

    accumulator ()
    {}
    
    accumulator ( accumulator &&  aaccu )
            : matrix( std::move( aaccu.matrix ) )
            , pending( std::move( aaccu.pending ) )
    {}
    
    accumulator ( std::unique_ptr< Hpro::TMatrix< value_t > > &&  amatrix,
                  update_list &&                                  apending )
            : matrix( std::move( amatrix ) )
            , pending( std::move( apending ) )
    {}

    //
    // add given product A × B to the accumulator
    //
    template < typename approx_t >
    void
    add ( const matop_t                     op_A,
          const Hpro::TMatrix< value_t > &  A,
          const matop_t                     op_B,
          const Hpro::TMatrix< value_t > &  B,
          const Hpro::TTruncAcc &           acc )
    {
        if ( is_blocked_all( A, B ) )
        {
            std::scoped_lock  lock( mtx_pending );
            
            pending.push_back( { op_A, &A, op_B, &B } );
        }// if
        else
        {
            //
            // compute update (either A or B is a leaf)
            //

            auto  T = hlr::multiply( value_t(1), op_A, A, op_B, B );

            //
            // apply update to accumulator
            //

            {
                std::scoped_lock  lock( mtx_matrix );
                
                if ( is_null( matrix ) )
                {
                    matrix = std::move( T );
                    return;
                }// if
            }

            const approx_t  apx;
            
            if ( is_dense( *T ) )
            {
                std::scoped_lock  lock( mtx_matrix );
                
                if ( is_dense( *matrix ) )
                {
                    hlr::add( value_t(1), *T, *matrix, acc, apx );
                }// if
                else
                {
                    // prefer dense format to avoid unnecessary truncations
                    hlr::add( value_t(1), *matrix, *T, acc, apx );
                    matrix = std::move( T );
                }// else
            }// if
            else
            {
                hlr::add( value_t(1), *T, *matrix, acc, apx );
            }// else
        }// else
    }

    //
    // shift down accumulated updates to sub blocks
    //
    template < typename approx_t >
    void
    shift ( Hpro::TBlockMatrix< value_t > &  M,
            accumulator_map_t &              accu_map,
            const Hpro::TTruncAcc &          acc,
            const approx_t &                 approx )
    {
        //
        // restrict local data and shift to accumulators of subblocks
        //
        
        for ( uint  i = 0; i < M.nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < M.nblock_cols(); ++j )
            {
                auto  M_ij    = M.block( i, j );
                auto  accu_ij = restrict( i, j, M );

                // TODO: guard access to accu_map???
                
                if ( accu_map.find( M_ij->id() ) == accu_map.end() )
                {
                    accu_map.emplace( std::make_pair( M_ij->id(), std::move( accu_ij ) ) );
                }// if
                else
                {
                    auto &  sub_accu = accu_map.at( M_ij->id() );

                    if ( is_null( sub_accu.matrix ) )
                        sub_accu.matrix = std::move( accu_ij.matrix );
                    else
                    {
                        // TODO: check about "dense" status
                        hlr::add( value_t(1), *accu_ij.matrix, *sub_accu.matrix, acc, approx );
                    }// else

                    for ( auto  [ op_A, A, op_B, B ] : accu_ij.pending )
                        sub_accu.template add< approx_t >( op_A, *A, op_B, *B, acc );
                }// else
            }// for
        }// for
    }
    
    //
    // apply accumulated updates and free accumulator matrix
    //
    template < typename approx_t >
    void
    apply ( const value_t               alpha,
            Hpro::TMatrix< value_t > &  M,
            const Hpro::TTruncAcc &     acc,
            const approx_t &            approx )
    {
        if ( ! is_null( matrix ) )
            hlr::add( alpha, *matrix, M, acc, approx );

        // clear_matrix();
    }

    //
    // return restriction of updates to block (i,j) of given block matrix
    //
    accumulator
    restrict ( const uint                             i,
               const uint                             j,
               const Hpro::TBlockMatrix< value_t > &  M ) const
    {
        auto  U_ij = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  P_ij = update_list();
        
        if ( ! is_null( matrix ) )
        {
            HLR_ASSERT( ! is_null( M.block( i, j ) ) );
            
            U_ij = hlr::matrix::restrict( *matrix, M.block( i, j )->block_is() );
        }// if

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            // filter out non-recursive updates
            if ( ! is_blocked_all( A, B ) )
                continue;
                                            
            auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
                                            
            for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
            {
                auto  A_il = BA->block( i, l, op_A );
                auto  B_lj = BB->block( l, j, op_B );
                                                
                if ( is_null_any( A_il, B_lj ) )
                    continue;
                                                
                P_ij.push_back( { op_A, A_il, op_B, B_lj } );
            }// for
        }// for

        return accumulator{ std::move( U_ij ), std::move( P_ij ) };
    }
};

// forward decl. of apply_node
template < typename value_t,
           typename approx_t >
struct apply_node;

// maps matrices to accumulators
template < typename value_t >
using  accumulator_map_t  = typename accumulator< value_t >::accumulator_map_t;

// maps matrices to apply_nodes
template < typename value_t,
           typename approx_t >
using  apply_map_t        = std::unordered_map< Hpro::id_t, apply_node< value_t, approx_t > * >;

// set of DAG nodes
using  nodes_list_t       = std::list< node * >;

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
struct lu_node : public node
{
    Hpro::TMatrix< value_t > *          A;
    apply_map_t< value_t, approx_t > &  apply_map;
    
    lu_node ( Hpro::TMatrix< value_t > *          aA,
              apply_map_t< value_t, approx_t > &  aapply_map )
            : A( aA )
            , apply_map( aapply_map )
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

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_upper_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
struct solve_upper_node : public node
{
    const Hpro::TMatrix< value_t > *    U;
    Hpro::TMatrix< value_t > *          A;
    apply_map_t< value_t, approx_t > &  apply_map;
    
    solve_upper_node ( const Hpro::TMatrix< value_t > *    aU,
                       Hpro::TMatrix< value_t > *          aA,
                       apply_map_t< value_t, approx_t > &  aapply_map )
            : U( aU )
            , A( aA )
            , apply_map( aapply_map )
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

///////////////////////////////////////////////////////////////////////////////////////
//
// solve_lower_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
struct solve_lower_node : public node
{
    const Hpro::TMatrix< value_t > *    L;
    Hpro::TMatrix< value_t > *          A;
    apply_map_t< value_t, approx_t > &  apply_map;

    solve_lower_node ( const Hpro::TMatrix< value_t > *    aL,
                       Hpro::TMatrix< value_t > *          aA,
                       apply_map_t< value_t, approx_t > &  aapply_map )
            : L( aL )
            , A( aA )
            , apply_map( aapply_map )
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
    
///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
struct update_node : public node
{
    const Hpro::TMatrix< value_t > *    A;
    const Hpro::TMatrix< value_t > *    B;
    Hpro::TMatrix< value_t > *          C;
    apply_map_t< value_t, approx_t > &  apply_map;

    update_node ( const Hpro::TMatrix< value_t > *    aA,
                  const Hpro::TMatrix< value_t > *    aB,
                  Hpro::TMatrix< value_t > *          aC,
                  apply_map_t< value_t, approx_t > &  aapply_map )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_map( aapply_map )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "A%d = mul( L%d, U%d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_ACCU, C->block_is() } }; }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        apply_map[ C->id() ]->add( apply_normal, *A,
                                   apply_normal, *B,
                                   acc );
    }

    virtual local_graph  refine_  ( const size_t  min_size );
};

///////////////////////////////////////////////////////////////////////////////////////
//
// accumulator control_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
struct apply_node : public node
{
    Hpro::TMatrix< value_t > *       M;
    accumulator_map_t< value_t > &   accu_map;
    
    apply_node ( Hpro::TMatrix< value_t > *      aM,
                 accumulator_map_t< value_t > &  aaccu_map )
            : M( aM )
            , accu_map( aaccu_map )
    { init(); }

    // wrapper for adding updates
    void  add ( const matop_t            op_A,
                const Hpro::TMatrix< value_t > &    A,
                const matop_t            op_B,
                const Hpro::TMatrix< value_t > &    B,
                const Hpro::TTruncAcc &  acc )
    {
        if ( accu_map.find( M->id() ) == accu_map.end() )
            accu_map.emplace( std::make_pair( M->id(), accumulator< value_t >() ) );

        accu_map[ M->id() ].template add< approx_t >( op_A, A, op_B, B, acc );
    }
    
    virtual std::string  to_string () const { return Hpro::to_string( "apply( %d )", M->id() ); }
    virtual std::string  color     () const { return "edd400"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_ACCU, M->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( is_leaf( M ) ) return { { ID_A,    M->block_is() } };
        else                return { { ID_ACCU, M->block_is() } };
    }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        const approx_t  apx;
        
        if ( is_blocked( M ) && ! Hpro::is_small( M ) )
        {
            if ( accu_map.find( M->id() ) != accu_map.end() )
                accu_map.at( M->id() ).template shift< approx_t >( * ptrcast( M, Hpro::TBlockMatrix< value_t > ), accu_map, acc, apx );
        }// if
        else
        {
            if ( accu_map.find( M->id() ) != accu_map.end() )
                accu_map.at( M->id() ).template apply< approx_t >( value_t(-1), *M, acc, apx );
        }// else
    }

    virtual local_graph  refine_  ( const size_t ) { return {}; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// apply DAG
//
///////////////////////////////////////////////////////////////////////////////////////

//
// construct DAG for applying updates
//
template < typename value_t,
           typename approx_t >
void
build_apply_dag ( Hpro::TMatrix< value_t > *          A,
                  accumulator_map_t< value_t > &      accu_map,
                  node *                              parent,
                  apply_map_t< value_t, approx_t > &  apply_map,
                  node_list_t &                       apply_nodes,
                  const size_t                        min_size )
{
    if ( is_null( A ) )
        return;

    auto  apply = dag::alloc_node< apply_node< value_t, approx_t > >( apply_nodes, A, accu_map );

    apply_map[ A->id() ] = apply;

    if ( parent != nullptr )
        apply->after( parent );
    
    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto  BA = ptrcast( A, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                    build_apply_dag( BA->block( i, j ), accu_map, apply,
                                     apply_map, apply_nodes, min_size );
            }// for
        }// for
    }// if
}

template < typename value_t,
           typename approx_t >
std::pair< apply_map_t< value_t, approx_t >,
           node_list_t >
build_apply_dag ( Hpro::TMatrix< value_t > *      A,
                  accumulator_map_t< value_t > &  accu_map,
                  const size_t                    min_size )
{
    apply_map_t< value_t, approx_t >  apply_map;
    node_list_t                       apply_nodes;

    build_apply_dag( A, accu_map, nullptr, apply_map, apply_nodes, min_size );

    return { std::move( apply_map ), std::move( apply_nodes ) };
}

//
// refinement methods
//

template < typename value_t,
           typename approx_t >
local_graph
lu_node< value_t, approx_t >::refine_  ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto        BA       = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BU       = BA;
        auto        BL       = BA;
        const auto  nbr      = BA->block_rows();
        const auto  nbc      = BA->block_cols();
        auto        finished = tensor2< node * >( nbr, nbc );
        
        if ( is_nd( A ) )
        {
            for ( uint i = 0; i < std::min( nbr, nbc )-1; ++i )
            {
                auto  A_ii  = BA->block( i, i );
                auto  U_ii  = A_ii;
                auto  L_ii  = A_ii;

                HLR_ASSERT( A_ii != nullptr );

                finished( i, i ) = g.alloc_node< lu_node< value_t, approx_t > >( A_ii, apply_map );

                if ( ! is_null( BA->block( nbr-1, i ) ) )
                {
                    finished( nbr-1, i ) = g.alloc_node< solve_upper_node< value_t, approx_t > >( U_ii, BA->block( nbr-1, i ), apply_map );
                    finished( nbr-1, i )->after( finished( i, i ) );
                }// if

                if ( ! is_null( BA->block( i, nbc-1 ) ) )
                {
                    finished( i, nbc-1 ) = g.alloc_node< solve_lower_node< value_t, approx_t > >( L_ii, BA->block( i, nbc-1 ), apply_map );
                    finished( i, nbc-1 )->after( finished( i, i ) );
                }// if
            }// for
        
            finished( nbr-1, nbc-1 ) = g.alloc_node< lu_node >( BA->block( nbr-1, nbc-1 ), apply_map );
        
            for ( uint i = 0; i < std::min( nbr, nbc )-1; ++i )
            {
                if ( ! is_null_any( BL->block( nbr-1, i ), BU->block( i, nbc-1 ), BA->block( nbr-1, nbc-1 ) ) )
                {
                    auto  update = g.alloc_node< update_node< value_t, approx_t > >( BL->block( nbr-1, i ),
                                                                                     BU->block( i, nbc-1 ),
                                                                                     BA->block( nbr-1, nbc-1 ),
                                                                                     apply_map );
                
                    update->after( finished( nbr-1, i ) );
                    update->after( finished( i, nbc-1 ) );
                    finished( nbr-1, nbc-1 )->after( update );
                }// if
            }// for
        }// if
        else
        {
            for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
            {
                //
                // factorise diagonal block
                //
            
                auto  A_ii  = BA->block( i, i );
                auto  L_ii  = A_ii;
                auto  U_ii  = A_ii;

                HLR_ASSERT( ! is_null_any( A_ii, L_ii, U_ii ) );

                finished( i, i ) = g.alloc_node< lu_node< value_t, approx_t > >( A_ii, apply_map );

                for ( uint j = i+1; j < nbr; j++ )
                {
                    if ( ! is_null( BA->block( j, i ) ) )
                    {
                        finished( j, i ) = g.alloc_node< solve_upper_node< value_t, approx_t > >( U_ii, BA->block( j, i ), apply_map );
                        finished( j, i )->after( finished( i, i ) );
                    }// if
                }// for

                for ( uint j = i+1; j < nbc; j++ )
                    if ( ! is_null( BA->block( i, j ) ) )
                    {
                        finished( i, j ) = g.alloc_node< solve_lower_node< value_t, approx_t > >( L_ii, BA->block( i, j ), apply_map );
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
                                                                                             BA->block( j, l ),
                                                                                             apply_map );

                            update->after( finished( j, i ) );
                            update->after( finished( i, l ) );
                            finished( j, l )->after( update );
                        }// if
                    }// for
                }// for
            }// for
        }// if
    }// if
    else
    {
        auto  apply = apply_map[ A->id() ];
        
        HLR_ASSERT( ! is_null( apply ) );

        apply->before( this );
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           typename approx_t >
local_graph
solve_upper_node< value_t, approx_t >::refine_  ( const size_t  min_size )
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

        if ( is_nd( U ) )
        {
            for ( uint j = 0; j < nbc; ++j )
            {
                const auto  U_jj = BU->block( j, j );
                
                HLR_ASSERT( ! is_null( U_jj ) );
                
                for ( uint i = 0; i < nbr; ++i )
                {
                    auto  A_ij = BA->block( i, j );
                    
                    if ( ! is_null( A_ij ) )
                        finished( i, j ) = g.alloc_node< solve_upper_node< value_t, approx_t > >( U_jj, A_ij, apply_map );
                }// for
            }// for
                
            for ( uint j = 0; j < nbc-1; ++j )
            {
                for ( uint i = 0; i < nbr; ++i )
                {
                    if ( ! is_null_any( BA->block( i, j ), BU->block( j, nbc-1 ), BA->block( i, nbc-1 ) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t, approx_t > >( BA->block( i, j ),
                                                                                         BU->block( j, nbc-1 ),
                                                                                         BA->block( i, nbc-1 ),
                                                                                         apply_map );

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
                const auto  U_jj = BU->block( j, j );
        
                assert( ! is_null( U_jj ) );

                for ( uint i = 0; i < nbr; ++i )
                    if ( ! is_null( BA->block(i,j) ) )
                        finished( i, j ) = g.alloc_node< solve_upper_node< value_t, approx_t > >(  U_jj, BA->block( i, j ), apply_map );
            }// for
        
            for ( uint j = 0; j < nbc; ++j )
            {
                for ( uint  k = j+1; k < nbc; ++k )
                    for ( uint  i = 0; i < nbr; ++i )
                        if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                        {
                            auto  update = g.alloc_node< update_node< value_t, approx_t > >( BX->block( i, j ),
                                                                                             BU->block( j, k ),
                                                                                             BA->block( i, k ),
                                                                                             apply_map );

                            update->after( finished( i, j ) );
                            finished( i, k )->after( update );
                        }// if
            }// for
        }// else
    }// if
    else
    {
        auto  apply = apply_map[ A->id() ];
        
        HLR_ASSERT( ! is_null( apply ) );

        apply->before( this );
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           typename approx_t >
local_graph
solve_lower_node< value_t, approx_t >::refine_  ( const size_t  min_size )
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
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                    finished( i, j ) = g.alloc_node< solve_lower_node< value_t, approx_t > >(  L_ii, BA->block( i, j ), apply_map );
        }// for
        
        for ( uint i = 0; i < nbr; ++i )
        {
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t, approx_t > >( BL->block( k, i ),
                                                                                         BX->block( i, j ),
                                                                                         BA->block( k, j ),
                                                                                         apply_map );

                        update->after( finished( i, j ) );
                        finished( k, j )->after( update );
                    }// if
        }// for
    }// if
    else
    {
        auto  apply = apply_map[ A->id() ];
        
        HLR_ASSERT( ! is_null( apply ) );

        apply->before( this );
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           typename approx_t >
local_graph
update_node< value_t, approx_t >::refine_  ( const size_t  min_size )
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
                                                                          BC->block( i, j ),
                                                                          apply_map );
                }// for
            }// for
        }// for
    }// if
    else
    {
        auto  apply = apply_map[ C->id() ];
        
        HLR_ASSERT( ! is_null( apply ) );

        apply->after( this );
    }// if

    g.finalize();
    
    return g;
}

}// namespace accu

}}}// namespace hlr::dag::lu

#endif // __HLR_DAG_LU_NODES_HH
