#ifndef __HLR_TBB_ARITH_ACCU_HH
#define __HLR_TBB_ARITH_ACCU_HH
//
// Project     : HLR
// Module      : arith.hh
// Description : arithmetic functions using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/add.hh"

namespace hlr { namespace tbb { namespace accu {

namespace hpro = HLIB;

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

template < typename value_t >
struct accumulator
{
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

    // accumulated computed updates
    std::unique_ptr< Hpro::TMatrix< value_t > >   matrix;

    // accumulated pending (recursive) updates
    update_list                                   pending;

    //
    // ctors
    //

    accumulator ()
    {}
    
    accumulator ( std::unique_ptr< Hpro::TMatrix< value_t > > &&  amatrix,
                  update_list &&                                  apending )
            : matrix( std::move( amatrix ) )
            , pending( std::move( apending ) )
    {}
    
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
    // add update A×B
    //
    void
    add_update ( const Hpro::TMatrix< value_t > &  A,
                 const Hpro::TMatrix< value_t > &  B )
    {
        pending.push_back( { apply_normal, &A, apply_normal, &B } );
    }

    void
    add_update ( const matop_t          op_A,
                 const Hpro::TMatrix< value_t > &  A,
                 const matop_t          op_B,
                 const Hpro::TMatrix< value_t > &  B )
    {
        pending.push_back( { op_A, &A, op_B, &B } );
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

                sub_accu(i,j) = restrict( i, j, M );
            }// for
        }// for

        return sub_accu;
    }

    //
    // compute updates by reduction
    //
    template < typename approx_t >
    std::unique_ptr< Hpro::TMatrix< value_t > >
    compute_reduce ( std::deque< update > &   computable,
                     const size_t             lb,
                     const size_t             ub,
                     const value_t            alpha,
                     const Hpro::TTruncAcc &  acc,
                     const approx_t &         approx,
                     const bool               handle_dense )
    {
        if ( ub - lb <= 1 )
        {
            auto  [ op_A, A, op_B, B ] = computable[ lb ];
            auto  T                    = hlr::multiply< value_t >( alpha, op_A, *A, op_B, *B );

            if ( handle_dense && ! matrix::is_dense( *T ) )
                T = hlr::matrix::convert_to_dense< value_t >( *T );
            
            return T;
        }// if
        else
        {
            const size_t  mid = (ub + lb) / 2;
            auto          T1  = std::unique_ptr< Hpro::TMatrix< value_t > >();
            auto          T2  = std::unique_ptr< Hpro::TMatrix< value_t > >();

            ::tbb::parallel_invoke(
                [&,alpha,lb,mid,handle_dense] { T1 = std::move( compute_reduce( computable, lb, mid, alpha, acc, approx, handle_dense ) ); },
                [&,alpha,mid,ub,handle_dense] { T2 = std::move( compute_reduce( computable, mid, ub, alpha, acc, approx, handle_dense ) ); } );

            // prefer dense format
            if ( matrix::is_dense( *T1 ) )
            {
                hlr::add( value_t(1), *T2, *T1 );

                return T1;
            }// if
            else if ( matrix::is_dense( *T2 ) )
            {
                hlr::add( value_t(1), *T1, *T2 );

                return T2;
            }// if
            else
            {
                // has to be low-rank: truncate
                auto  R1       = ptrcast( T1.get(), matrix::lrmatrix< value_t > );
                auto  R2       = ptrcast( T2.get(), matrix::lrmatrix< value_t > );
                auto  [ U, V ] = approx( { R1->U_direct(), R2->U_direct() },
                                         { R1->V_direct(), R2->V_direct() },
                                         acc );

                return std::make_unique< matrix::lrmatrix< value_t > >( T1->row_is(), T1->col_is(), std::move( U ), std::move( V ) );
            }// else
        }// else
    }// if
    
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
            if ( matrix::is_dense_all( A, B ) ||
                 ( is_blocked( A ) && matrix::is_dense( B ) ) ||
                 ( is_blocked( B ) && matrix::is_dense( A ) ))
            {
                handle_dense = true;
                break;
            }// if
        }// for
        
        std::deque< update >  computable;

        // filter computable updates
        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( is_blocked_all( A, B ) )
            {
                if ( is_blocked_all( *A, *B, M ) )
                    continue;
                
                //
                // if M is a leaf and A _and_ B are blocked, a temporary matrix
                // is needed for further recursive update handling
                //

                if ( ! is_null( BC ) )
                    continue;
                
                // TODO: non low-rank M
                HLR_ASSERT( matrix::is_lowrank( M ) );
                
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
                            BC->set_block( i, j, new matrix::dense_matrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                      BB->block( 0, j, op_B )->col_is( op_B ) ) );
                        else
                            BC->set_block( i, j, new matrix::lrmatrix< value_t >( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                                  BB->block( 0, j, op_B )->col_is( op_B ) ) );
                    }// for
                }// for
            }// if
            else
            {
                computable.push_back( { op_A, A, op_B, B } );
            }// else
        }// for

        // handle computable updates in parallel
        if ( ! computable.empty() )
        {
            auto  T = compute_reduce( computable, 0, computable.size(), alpha, acc, approx, handle_dense );

            //
            // apply update to accumulator
            //
            
            if ( is_null( matrix ) )
            {
                matrix = std::move( T );
            }// if
            else if ( ! matrix::is_dense( *matrix ) && matrix::is_dense( *T ) )
            {
                // prefer dense format to avoid unnecessary truncations
                hlr::add( value_t(1), *matrix, *T );
                matrix = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *matrix, acc, approx );
            }// else
        }// if

        //
        // now handle recursive updates if M is a leaf block
        //
    
        if ( ! is_null( BC ) )
        {
            auto  sub_accu = restrict( *BC );

            matrix.reset( nullptr );
        
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,alpha] ( const auto & r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                            // replace block in BC by accumulator matrix for agglomeration below
                            BC->delete_block( i, j );
                            BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                        }// for
                } );

            //
            // finally convert subblocks to single low-rank matrix for new accumulated updates
            //

            if ( handle_dense )
                matrix = hlr::matrix::convert_to_dense< value_t >( *BC );
            else
                matrix = hlr::matrix::convert_to_lowrank( *BC, acc, approx );
        }// if
    }

    //
    // return true if given matrix is dense
    //
    bool
    check_dense ( const Hpro::TMatrix< value_t > &  M ) const
    {
        // return false;
        if ( matrix::is_dense( M ) )
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
                    if ( ! is_null( B->block( i, j ) ) && ! matrix::is_dense( B->block( i, j ) ) )
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

//
// compute C = C + α op( A ) op( B ) where A and B are provided as accumulated updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t               alpha,
           Hpro::TMatrix< value_t > &  C,
           accumulator< value_t > &    accu,
           const Hpro::TTruncAcc &     acc,
           const approx_t &            approx )
{
    //
    // first handle all computable updates to C, including if C is non-blocked
    //

    accu.eval( alpha, C, acc, approx );
    
    //
    // now handle recursive updates
    //
    
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );

        //
        // first, split update U into subblock updates
        // (to release U before recursion and by that avoid
        //  memory consumption dependent on hierarchy depth)
        //

        auto  sub_accu = accu.restrict( *BC );

        accu.clear_matrix();

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        // TODO: test if creation of sub-accumulators benefits from affinity_partitioner
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                            0, BC->nblock_cols() ),
            [&,BC,alpha] ( const auto & r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                        multiply< value_t >( alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
            } );
    }// if
    else 
    {
        // apply accumulated updates
        accu.apply( alpha, C, acc, approx );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                     alpha,
           const Hpro::matop_t               op_A,
           const Hpro::TMatrix< value_t > &  A,
           const Hpro::matop_t               op_B,
           const Hpro::TMatrix< value_t > &  B,
           Hpro::TMatrix< value_t > &        C,
           const Hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
{
    auto  accu = detail::accumulator< value_t >();

    accu.add_update( op_A, A, op_B, B );
    detail::multiply< value_t >( alpha, C, accu, acc, approx );
}

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based LU factorization
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based LU factorization
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

template < typename value_t,
           typename approx_t >
void
solve_lower_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TMatrix< value_t > &        M,
                  accumulator< value_t > &          accu,
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );
    
    if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict( *BM );

        accu.clear_matrix();

        if ( side == from_left )
        {
            for ( uint i = 0; i < BM->nblock_rows(); ++i )
            {
                const auto  L_ii = BL->block( i, i );

                ::tbb::parallel_for< uint >(
                    0, BM->nblock_cols(),
                    [=,&sub_accu,&acc,&approx] ( const uint j )
                    {
                        solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j), sub_accu(i,j), acc, approx );
                    } );

                for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        sub_accu(k,j).pending.push_back( { apply_normal, BL->block(k,i),
                                                           apply_normal, BM->block(i,j) } );
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), M, acc, approx );

        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                   side,
                  const diag_type_t                   diag,
                  const Hpro::TMatrix< value_t > &    U,
                  Hpro::TMatrix< value_t > &          M,
                  accumulator< value_t > &            accu,
                  const Hpro::TTruncAcc &             acc,
                  const approx_t &                    approx )
{
    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );
    
    if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict( *BM );

        accu.clear_matrix();

        if ( side == from_left )
        {
            HLR_ASSERT( false );
        }// if
        else
        {
            for ( uint j = 0; j < BM->nblock_cols(); ++j )
            {
                const auto  U_jj = BU->block( j, j );
            
                ::tbb::parallel_for< uint >(
                    0, BM->nblock_rows(),
                    [=,&sub_accu,&acc,&approx] ( const uint i )
                    {
                        solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ), sub_accu(i,j), acc, approx );
                    } );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        sub_accu(i,k).pending.push_back( { apply_normal, BM->block(i,j),
                                                           apply_normal, BU->block(j,k) } );
            }// for
        }// else
    }// if
    else
    {
        // no recursive updates left, apply accumulated updates and solve
        accu.apply( value_t(-1), M, acc, approx );
        
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  M,
     accumulator< value_t > &    accu,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    //
    // evaluate all computable updates to M
    //

    accu.eval( value_t(1), M, acc, approx );
    
    //
    // (recursive) LU factorization
    //
    
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

        //
        // first, split accumulated updates U and recursive updates upd_rec
        // into subblock updates
        // - to release U before recursion and by that avoid memory
        //   consumption dependent on hierarchy depth
        //

        auto  sub_accu = accu.restrict( *BM );

        accu.clear_matrix();

        //
        // recursive LU factorization but add updates to accumulator
        // instead of applying them
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            lu< value_t >( *B_ii, sub_accu(i,i), acc, approx );

            ::tbb::parallel_invoke(
                [=,&sub_accu,&acc,&approx]
                {
                    ::tbb::parallel_for< uint >(
                        i+1, BM->nblock_rows(),
                        [=,&sub_accu,&acc,&approx] ( const uint j )
                        {
                            solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );
                        } );
                },

                [=,&sub_accu,&acc,&approx]
                {
                    ::tbb::parallel_for< uint >(
                        i+1, BM->nblock_cols(),
                        [=,&sub_accu,&acc,&approx] ( const uint j )
                        {
                            solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );
                        } );
                } );

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    sub_accu(j,l).pending.push_back( { apply_normal, BM->block( j, i ),
                                                       apply_normal, BM->block( i, l ) } );
        }// for
    }// if
    else
    {
        //
        // no recursive updates left, apply accumulated updates
        // and factorize
        //

        accu.apply( value_t(-1), M, acc, approx );
        
        if ( matrix::is_dense( M ) )
        {
            auto  D              = ptrcast( &M, matrix::dense_matrix< value_t > );
            auto  DD             = D->mat();
            auto  was_compressed = D->is_compressed();
            
            blas::invert( DD );

            if ( was_compressed )
                D->compress( acc );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  M,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    auto  accu = detail::accumulator< value_t >();
    
    detail::lu< value_t >( M, accu, acc, approx );
}

}}}// namespace hlr::tbb::accu

#endif // __HLR_TBB_ARITH_ACCU_HH
