#ifndef __HLR_DAG_DETAIL_LU_ACCU_EAGER_HH
#define __HLR_DAG_DETAIL_LU_ACCU_EAGER_HH
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
#include "hlr/matrix/restrict.hh"

#include "hlr/arith/norm.hh" // DEBUG

////////////////////////////////////////////////////////////////////////////////
//
// accumulated version
//
////////////////////////////////////////////////////////////////////////////////

namespace hlr { namespace dag { namespace lu { namespace accu { namespace eager {

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

    accumulator &  operator = ( accumulator && aaccu )
    {
        matrix  = std::move( aaccu.matrix );
        pending = std::move( aaccu.pending );

        return *this;
    }
    
    //
    // remove update matrix
    //
    void
    clear_matrix ()
    {
        matrix.reset( nullptr );
    }

    //
    // release matrix
    //
    Hpro::TMatrix< value_t > *
    release_matrix ()
    {
        return matrix.release();
    }

    //
    // add update C += A×B
    //

    template < typename approx_t >
    void
    add_update ( const matop_t                     op_A,
                 const Hpro::TMatrix< value_t > &  A,
                 const matop_t                     op_B,
                 const Hpro::TMatrix< value_t > &  B,
                 const Hpro::TMatrix< value_t > &  C,
                 const Hpro::TTruncAcc &           acc )
    {
        if ( is_blocked_all( A, B, C ) )
        {
            std::scoped_lock  lock( mtx_pending );
            
            pending.push_back( { op_A, &A, op_B, &B } );
        }// if
        else
        {
            //
            // determine if to prefer dense format
            //
            
            const bool  handle_dense = check_dense( C ) || ( ! is_blocked_all( A, B ) && ! is_lowrank_any( A, B ) );

            //
            // compute update
            //

            auto            T = std::unique_ptr< Hpro::TMatrix< value_t > >();
            const approx_t  apx;
            
            if ( is_blocked_all( A, B ) )
            {
                //
                // create temporary block matrix to evaluate product
                //
                
                // TODO: non low-rank M
                HLR_ASSERT( is_lowrank( C ) );
                
                auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
                auto  BB = cptrcast( &B, Hpro::TBlockMatrix< value_t > );
                auto  BC = std::make_unique< Hpro::TBlockMatrix< value_t > >( BA->row_is( op_A ), BB->col_is( op_B ) );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );

                        if ( handle_dense )
                            BC->set_block( i, j, new Hpro::TDenseMatrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                    BB->block( 0, j, op_B )->col_is( op_B ) ) );
                        else
                            BC->set_block( i, j, new Hpro::TRkMatrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                 BB->block( 0, j, op_B )->col_is( op_B ) ) );
                    }// for
                }// for

                if constexpr ( true )
                {
                    //
                    // compute using standard arithmetic
                    //

                    hlr::multiply( value_t(1), op_A, A, op_B, B, *BC, acc, apx );
                }// if
                else
                {
                    //
                    // compute using accumulators starting with A×B
                    //

                    auto  accu_BC  = accumulator( *BC, { op_A, A, op_B, B } );
                    auto  sub_accu = accu_BC.restrict( *BC );

                    //
                    // apply recursive updates
                    //
        
                    for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                    {
                        for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                        {
                            sub_accu(i,j).eval( value_t(1), *BC->block(i,j), acc, apx );

                            // replace block in BC by accumulator matrix for agglomeration below
                            BC->delete_block( i, j );
                            BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                        }// for
                    }// for
                }// else

                //
                // finally convert subblocks to single low-rank update
                //

                if ( handle_dense )
                    T = std::move( seq::matrix::convert_to_dense< value_t >( *BC ) );
                else
                    T = std::move( seq::matrix::convert_to_lowrank( *BC, acc, apx ) );
            }// if
            else
            {
                //
                // directly multiply as A/B is leaf
                //

                T = std::move( hlr::multiply( value_t(1), op_A, A, op_B, B ) );
                
                if ( handle_dense && ! is_dense( *T ) )
                    T = matrix::convert_to_dense< value_t >( *T );
            }// else
            
            //
            // apply update to accumulator
            //
            
            std::scoped_lock  lock( mtx_matrix );
            
            if ( is_null( matrix ) )
            {
                matrix = std::move( T );
            }// if
            else if ( ! is_dense( *matrix ) && is_dense( *T ) )
            {
                // prefer dense format to avoid unnecessary truncations
                hlr::add( value_t(1), *matrix, *T );
                matrix = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *matrix, acc, apx );
            }// else
        }// else
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

    //
    // return restriction of updates to all sub blocks of given block matrix
    //
    tensor2< accumulator >
    restrict ( const Hpro::TBlockMatrix< value_t > &  M ) const
    {
        tensor2< accumulator >  sub_accu( M.nblock_rows(), M.nblock_cols() );
        
        for ( uint  i = 0; i < M.nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < M.nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( M.block( i, j ) ) );

                sub_accu(i,j) = std::move( restrict( i, j, M ) );
            }// for
        }// for

        return sub_accu;
    }
    
    //
    // evaluate all computable updates to matrix M
    //
    template < typename approx_t >
    void
    eval ( const value_t                     alpha,
           const Hpro::TMatrix< value_t > &  M,
           const Hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
    {
        std::unique_ptr< Hpro::TBlockMatrix< value_t > >  BC; // for recursive handling

        //
        // handle all, actually computable updates, i.e., one factor is a leaf block
        //

        bool  handle_dense = check_dense( M );

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( ! is_blocked_all( A, B ) && ! is_lowrank_any( A, B ) )
            {
                handle_dense = true;
                break;
            }// if
        }// for
        
        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            // can not handle pure recursive updates
            if ( is_blocked_all( A, B, &M ) )
                continue;
        
            if ( is_blocked_all( A, B ) )
            {
                //
                // if M is a leaf and A _and_ B are blocked, a temporary matrix
                // is created for further recursive update handling
                //

                if ( ! is_null( BC ) )
                    continue;
                
                // TODO: non low-rank M
                HLR_ASSERT( is_lowrank( M ) );
                
                auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
                auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
                
                BC = std::make_unique< Hpro::TBlockMatrix< value_t > >( A->row_is( op_A ), B->col_is( op_B ) );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                for ( uint  i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                    {
                        HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );

                        if ( handle_dense )
                            BC->set_block( i, j, new Hpro::TDenseMatrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                    BB->block( 0, j, op_B )->col_is( op_B ) ) );
                        else
                            BC->set_block( i, j, new Hpro::TRkMatrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                 BB->block( 0, j, op_B )->col_is( op_B ) ) );
                    }// for
                }// for
            }// if
            else
            {
                //
                // compute update (either A or B is a leaf)
                //

                auto  T = hlr::multiply< value_t >( alpha, op_A, *A, op_B, *B );

                if ( handle_dense && ! is_dense( *T ) )
                    T = matrix::convert_to_dense< value_t >( *T );
                
                //
                // apply update to accumulator
                //
            
                if ( is_null( matrix ) )
                {
                    matrix = std::move( T );
                }// if
                else if ( ! is_dense( *matrix ) && is_dense( *T ) )
                {
                    // prefer dense format to avoid unnecessary truncations
                    hlr::add( value_t(1), *matrix, *T );
                    matrix = std::move( T );
                }// if
                else
                {
                    hlr::add( value_t(1), *T, *matrix, acc, approx );
                }// else
            }// else
        }// for

        //
        // now handle recursive updates if M is a leaf block
        //
    
        if ( ! is_null( BC ) )
        {
            //
            // TODO: try with empty sub_mat, don't release U and add sub results later
            //
        
            //
            // first, split update matrix into subblock updates
            // (to release matrix before recursion)
            //

            auto  sub_accu = restrict( *BC );

            matrix.reset( nullptr );
        
            //
            // apply recursive updates
            //
        
            for ( uint  i = 0; i < BC->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < BC->nblock_cols(); ++j )
                {
                    sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                    // replace block in BC by accumulator matrix for agglomeration below
                    BC->delete_block( i, j );
                    BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                }// for
            }// for

            //
            // finally convert subblocks to single low-rank matrix for new accumulated updates
            //

            if ( handle_dense )
                matrix = seq::matrix::convert_to_dense< value_t >( *BC );
            else
                matrix = seq::matrix::convert_to_lowrank( *BC, acc, approx );
        }// if
    }

    //
    // shift down accumulated updates to sub blocks
    //
    template < typename approx_t >
    void
    shift ( Hpro::TBlockMatrix< value_t > &  M,
            accumulator_map_t &              accu_map,
            std::mutex &                     accu_mtx,
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
                auto           M_ij     = M.block( i, j );
                auto           accu_ij  = restrict( i, j, M );
                accumulator *  sub_accu = nullptr;

                {
                    auto  lock = std::scoped_lock( accu_mtx );

                    if ( accu_map.contains( M_ij->id() ) )
                        sub_accu = & ( accu_map.at( M_ij->id() ) );
                    else
                        accu_map.emplace( std::make_pair( M_ij->id(), std::move( accu_ij ) ) );
                }
                
                if ( ! is_null( sub_accu ) )
                {
                    //
                    // already exists accumulator for M_ij
                    //

                    Hpro::TMatrix< value_t > *  accu_matrix = nullptr;

                    {
                        auto  lock = std::scoped_lock( sub_accu->mtx_matrix );
                        
                        if ( is_null( sub_accu->matrix ) )
                            sub_accu->matrix = std::move( accu_ij.matrix );
                        else
                            accu_matrix = sub_accu->matrix.get();
                    }
                    
                    if ( ! is_null( accu_matrix ) )
                    {
                        //
                        // add update to existing accumulator
                        //
                        
                        hlr::add( value_t(1), *accu_ij.matrix, *sub_accu->matrix, acc, approx );
                        // // TODO: check about "dense" status
                        // if ( is_dense( *sub_accu->matrix ) || ! is_dense( *accu_ij.matrix ) )
                        //     hlr::add( value_t(1), *accu_ij.matrix, *sub_accu->matrix, acc, approx );
                        // else
                        // {
                        //     hlr::add( value_t(1), *sub_accu->matrix, *accu_ij.matrix, acc, approx );
                        //     sub_accu->matrix = std::move( accu_ij.matrix );
                        // }// else
                    }// else

                    for ( auto  [ op_A, A, op_B, B ] : accu_ij.pending )
                        sub_accu->template add_update< approx_t >( op_A, *A, op_B, *B, *M_ij, acc );
                }// if
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

        clear_matrix();
    }
    
    //
    // return true if given matrix is dense
    //
    bool
    check_dense ( const Hpro::TMatrix< value_t > &  M ) const
    {
        // return false;
        if ( is_dense( M ) )
        {
            return true;
        }// if
        else if ( is_blocked( M ) )
        {
            //
            // test if all subblocks are dense
            //

            auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    if ( ! is_null( B->block( i, j ) ) && ! is_dense( B->block( i, j ) ) )
                         return false;
                }// for
            }// for

            return true;
        }// if
        else
        {
            return false;
        }// else
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

        // std::cout << "lu " << A->id() << " : " << norm::frobenius( *A ) << std::endl;
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

        // std::cout << "upper " << A->id() << " : " << norm::frobenius( *A ) << std::endl;
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

        // std::cout << "lower " << A->id() << " : " << norm::frobenius( *A ) << std::endl;
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
    apply_node< value_t, approx_t > *   apply;

    update_node ( const Hpro::TMatrix< value_t > *    aA,
                  const Hpro::TMatrix< value_t > *    aB,
                  Hpro::TMatrix< value_t > *          aC,
                  apply_map_t< value_t, approx_t > &  aapply_map )
            : A( aA )
            , B( aB )
            , C( aC )
            , apply_map( aapply_map )
            , apply( aapply_map[ C->id() ] )
    {
        init();
        HLR_ASSERT( ! is_null( apply ) );
    }

    virtual std::string  to_string () const { return Hpro::to_string( "A%d = mul( L%d, U%d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_ACCU, C->block_is() } }; }

    virtual void  run_  ( const Hpro::TTruncAcc &  acc )
    {
        apply->add( apply_normal, *A,
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
    Hpro::TMatrix< value_t > *      M;
    accumulator_map_t< value_t > *  accu_map;
    std::mutex *                    accu_mtx;
    
    apply_node ( Hpro::TMatrix< value_t > *      aM,
                 accumulator_map_t< value_t > *  aaccu_map,
                 std::mutex *                    aaccu_mtx )
            : M( aM )
            , accu_map( aaccu_map )
            , accu_mtx( aaccu_mtx )
    { init(); }

    // wrapper for adding updates
    void  add ( const matop_t                     op_A,
                const Hpro::TMatrix< value_t > &  A,
                const matop_t                     op_B,
                const Hpro::TMatrix< value_t > &  B,
                const Hpro::TTruncAcc &           acc )
    {
        accumulator< value_t > *  accu = nullptr;
        
        {
            auto  lock = std::scoped_lock( *accu_mtx );
            
            if ( ! accu_map->contains( M->id() ) )
                accu_map->emplace( std::make_pair( M->id(), accumulator< value_t >() ) );

            accu = & ( accu_map->at( M->id() ) );
        }
        
        accu->template add_update< approx_t >( op_A, A, op_B, B, *M, acc );
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
        const approx_t            apx;
        accumulator< value_t > *  accu = nullptr;

        {
            auto  lock = std::scoped_lock( *accu_mtx );
            
            if ( accu_map->contains( M->id() ) )
                accu = & ( accu_map->at( M->id() ) );
        }

        // if ( ! is_null( accu ) )
        //     accu->template eval< approx_t >( value_t(1), * M, acc, apx );
        
        if ( is_blocked( M ) ) // && ! Hpro::is_small( M ) )
        {
            if ( ! is_null( accu ) )
                accu->template shift< approx_t >( * ptrcast( M, Hpro::TBlockMatrix< value_t > ), *accu_map, *accu_mtx, acc, apx );
        }// if
        else
        {
            if ( ! is_null( accu ) )
            {
                // if ( ! is_null( accu_map->at( M->id() ).matrix ) )
                //     std::cout << "apply " << M->id() << " : " << norm::frobenius( *M ) << ", " << norm::frobenius( * (accu_map->at( M->id() ).matrix) ) << std::endl;
            
                accu->template apply< approx_t >( value_t(-1), *M, acc, apx );
            }// if

            // std::cout << "apply " << M->id() << " : " << norm::frobenius( *M ) << std::endl;
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
                  accumulator_map_t< value_t > *      accu_map,
                  std::mutex *                        accu_mtx,
                  node *                              parent,
                  apply_map_t< value_t, approx_t > &  apply_map,
                  node_list_t &                       apply_nodes,
                  const size_t                        min_size )
{
    if ( is_null( A ) )
        return;

    auto  apply = dag::alloc_node< apply_node< value_t, approx_t > >( apply_nodes, A, accu_map, accu_mtx );

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
                    build_apply_dag( BA->block( i, j ), accu_map, accu_mtx,
                                     apply, apply_map, apply_nodes, min_size );
            }// for
        }// for
    }// if
}

template < typename value_t,
           typename approx_t >
std::pair< apply_map_t< value_t, approx_t >,
           node_list_t >
build_apply_dag ( Hpro::TMatrix< value_t > *      A,
                  accumulator_map_t< value_t > *  accu_map,
                  std::mutex *                    accu_mtx,
                  const size_t                    min_size )
{
    apply_map_t< value_t, approx_t >  apply_map;
    node_list_t                       apply_nodes;

    build_apply_dag( A, accu_map, accu_mtx, nullptr, apply_map, apply_nodes, min_size );

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
        apply->after( this );
    }// if

    g.finalize();
    
    return g;
}

}}}}}// namespace hlr::dag::lu::accu::eager

#endif // __HLR_DAG_DETAIL_LU_ACCU_EAGER_HH
