#ifndef __HLR_TBB_DETAIL_UNIFORM_ACCU_LU_HH
#define __HLR_TBB_DETAIL_UNIFORM_ACCU_LU_HH
//
// Project     : HLR
// Module      : arith/detail/uniform_accu
// Description : arithmetic functions for uniform matrices using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/task_arena.h>

#include <hlr/arith/multiply.hh>
#include <hlr/arith/blas.hh>
#include <hlr/arith/solve.hh>
#include <hlr/arith/detail/uniform.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/restrict.hh>
#include <hlr/utils/hash.hh>
#include <hlr/utils/tensor.hh>

#include <hlr/tbb/detail/uniform_basis.hh>

namespace hlr { namespace tbb { namespace uniform { namespace accu { namespace detail2 {

using  hlr::matrix::shared_cluster_basis;
using  hlr::matrix::is_uniform_lowrank;
using  hlr::matrix::is_uniform_lowrank_all;
using  hlr::matrix::uniform_lrmatrix;
using  hlr::uniform::is_matrix_map_t;
using  hlr::tbb::uniform::detail::compute_extended_basis;
using  hlr::tbb::uniform::detail::update_coupling;

// maps index set to product U×V' of inner matrix product
template < typename value_t >
using  inner_map_t   = std::unordered_map< indexset, blas::matrix< value_t >, indexset_hash >;

//
// matrix update accumulator with parallel update handling
//
template < typename value_t >
struct accumulator
{
    //
    // represents an update, i.e., matrix product
    //
    struct update
    {
        const matop_t               op_A;
        Hpro::TMatrix< value_t > *  A;
        const matop_t               op_B;
        Hpro::TMatrix< value_t > *  B;
    };
    
    // represents set of updates
    using  update_list = std::list< update >;

    // accumulated computed updates
    std::unique_ptr< Hpro::TMatrix< value_t > >   matrix;

    // accumulated pending (recursive) updates
    update_list                                   pending;
    std::mutex                                    pend_mtx;
    
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
    // copy operator
    //
    accumulator &  operator = ( accumulator &&  accu )
    {
        matrix  = std::move( accu.matrix );
        pending = std::move( accu.pending );
        
        return *this;
    }
    
    //
    // add update A×B
    //
    void
    add_update ( Hpro::TMatrix< value_t > &  A,
                 Hpro::TMatrix< value_t > &  B )
    {
        if ( pend_mtx.try_lock() ) pend_mtx.unlock(); else std::cout << "mtx locked" << std::endl;
        
        pending.push_back( { apply_normal, &A, apply_normal, &B } );
    }

    void
    add_update ( const matop_t               op_A,
                 Hpro::TMatrix< value_t > &  A,
                 const matop_t               op_B,
                 Hpro::TMatrix< value_t > &  B )
    {
        if ( pend_mtx.try_lock() ) pend_mtx.unlock(); else std::cout << "mtx locked" << std::endl;
        
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
    restrict ( const uint                       i,
               const uint                       j,
               Hpro::TBlockMatrix< value_t > &  M ) const
    {
        auto  U_ij = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  P_ij = update_list();
        
        if ( ! is_null( matrix ) )
        {
            HLR_ASSERT( ! is_null( M.block( i, j ) ) );
            
            U_ij = hlr::matrix::restrict( *matrix, M.block( i, j )->block_is() );

            // prefer dense if destination is dense
            if ( check_dense( *M.block( i, j ) ) && ! hlr::matrix::is_dense( *U_ij ) )
                U_ij = std::move( hlr::matrix::convert_to_dense< value_t >( *U_ij ) );
        }// if

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            // filter out non-recursive updates
            if ( ! is_blocked_all( A, B ) )
                continue;
                                            
            auto  BA = ptrcast( A, Hpro::TBlockMatrix< value_t > );
            auto  BB = ptrcast( B, Hpro::TBlockMatrix< value_t > );
                                            
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
    restrict ( Hpro::TBlockMatrix< value_t > &  M ) const
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
    eval ( const value_t               alpha,
           Hpro::TMatrix< value_t > &  M,
           const Hpro::TTruncAcc &     acc,
           const approx_t &            approx )
    {
        auto  pend_lock = std::scoped_lock( pend_mtx );
        
        //
        // first check for dense handling
        //

        bool  handle_dense = check_dense( M );

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( hlr::matrix::is_dense_all( A, B ) || ( is_blocked_any( A, B ) && hlr::matrix::is_dense_any( A, B ) ) )
            {
                handle_dense = true;
                break;
            }// if
        }// for

        // handle_dense = false;
        
        if ( ! handle_dense )
        {
            //
            // filter out different variants of uniform factors
            //
            
            auto  pending_uniAB = std::vector< update >();
            auto  pending_uniA  = std::vector< update >();
            auto  pending_uniB  = std::vector< update >();
        
            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( is_uniform_lowrank_all( (*it).A, (*it).B ) )
                {
                    pending_uniAB.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for
            
            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( is_uniform_lowrank( (*it).A ) )
                {
                    pending_uniA.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for

            for ( auto  it = pending.begin(); it != pending.end(); )
            {
                if ( is_uniform_lowrank( (*it).B ) )
                {
                    pending_uniB.push_back( *it );
                    it = pending.erase( it );
                }// if
                else
                    ++it;
            }// for

            //
            // initialize matrices
            //
            
            auto  U = blas::matrix< value_t >(); // row basis of uniform × uniform
            auto  V = blas::matrix< value_t >(); // column basis of uniform × uniform
            auto  R = blas::matrix< value_t >(); // sum of couplings for uniform × uniform
            auto  Z = blas::matrix< value_t >(); // result of uniform × *
            auto  Y = blas::matrix< value_t >(); // result of * × uniform

            if ( ! pending_uniAB.empty() )
            {
                auto  [ op_A, A, op_B, B ] = pending_uniAB.front();

                U = ptrcast( A, uniform_lrmatrix< value_t > )->row_basis( op_A );
                V = ptrcast( B, uniform_lrmatrix< value_t > )->col_basis( op_B );
            }// if
            
            if ( ! pending_uniA.empty() )
            {
                auto  [ op_A, A, op_B, B ] = pending_uniA.front();
                
                if ( U.ncols() == 0 )
                {
                    U = ptrcast( A, uniform_lrmatrix< value_t > )->row_basis( op_A );
                }// if
                
                Z = std::move( blas::matrix< value_t >( M.ncols(), U.ncols() ) );
            }// if

            if ( ! pending_uniB.empty() )
            {
                auto  [ op_A, A, op_B, B ] = pending_uniB.front();

                if ( V.ncols() == 0 )
                {
                    V = ptrcast( B, hlr::matrix::uniform_lrmatrix< value_t > )->col_basis( op_B );
                }// if

                Y = std::move( blas::matrix< value_t >( M.nrows(), V.ncols() ) );
            }// if

            //
            // evaluate updates by handling different uniform product variants in parallel
            //

            ::tbb::parallel_invoke(

                [&,this] ()
                {
                    if ( ! pending_uniAB.empty() )
                        R = std::move( sum_uni_uni( 0, pending_uniAB.size(), pending_uniAB ) );
                },

                [&,this] ()
                {
                    if ( ! pending_uniA.empty() )
                        Z = std::move( sum_uni_any( 0, pending_uniA.size(), pending_uniA ) );
                },

                [&,this] ()
                {
                    if ( ! pending_uniB.empty() )
                        Y = std::move( sum_any_uni( 0, pending_uniB.size(), pending_uniB ) );
                }
            );
            
            //
            // sum up individual updates
            //

            if ( R.ncols() > 0 )
            {
                if ( Z.ncols() > 0 )
                {
                    // Z = Z + V·R'
                    blas::prod( value_t(1), V, blas::adjoint( R ), value_t(1), Z );
                }// if
                else if ( Y.ncols() > 0 )
                {
                    // Y = Y + U·R
                    blas::prod( value_t(1), U, R, value_t(1), Y );
                }// if
                else
                {
                    //
                    // just update with U·R·V'
                    //
                    if ( is_null( matrix ) )
                        matrix = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(),
                                                                                 std::move( blas::prod( alpha, U, R ) ),
                                                                                 std::move( blas::copy( V ) ) );
                    else
                    {
                        auto  US = blas::prod( U, R );
                        auto  T  = Hpro::TRkMatrix< value_t >( M.row_is(), M.col_is(), US, V );
                
                        hlr::add( alpha, T, *matrix, acc, approx );
                    }// else
                }// else
            }// if

            if ( Z.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                {
                    blas::scale( alpha, Z );
                    matrix = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( blas::copy( U ) ), std::move( Z ) );
                }// if
                else
                {
                    auto  T = Hpro::TRkMatrix< value_t >( M.row_is(), M.col_is(), U, Z );
                    
                    hlr::add( alpha, T, *matrix, acc, approx );
                }// else
            }// if
            
            if ( Y.ncols() > 0 )
            {
                if ( is_null( matrix ) )
                {
                    blas::scale( alpha, Y );
                    matrix = std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( Y ), std::move( blas::copy( V ) ) );
                }// if
                else
                {
                    auto  T = Hpro::TRkMatrix< value_t >( M.row_is(), M.col_is(), Y, V );
                    
                    hlr::add( alpha, T, *matrix, acc, approx );
                }// else
            }// if
        }// if

        //
        // handle remaining computable updates, i.e., one factor is a leaf block
        //
        
        auto  pending_comp = std::vector< update >();
        
        for ( auto  it = pending.begin(); it != pending.end(); )
        {
            // HLR_ASSERT( ! hlr::matrix::is_uniform_lowrank_any( (*it).A, (*it).B ) );
            
            if ( ! is_blocked_all( (*it).A, (*it).B ) )
            {
                pending_comp.push_back( *it );
                it = pending.erase( it );
            }// if
            else
                ++it;
        }// for
        
        if ( ! pending_comp.empty() )
        {
            auto  T = std::move( sum_pending( 0, pending_comp.size(), pending_comp, handle_dense, acc, approx ) );

            if ( is_null( matrix ) )
            {
                matrix = std::move( T );
            }// if
            else if ( ! hlr::matrix::is_dense( *matrix ) && hlr::matrix::is_dense( *T ) )
            {
                // prefer dense format to avoid unnecessary truncations
                hlr::add( value_t(1), *matrix, *T, acc, approx );
                matrix = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *matrix, acc, approx );
            }// else
        }// if
        
        //
        // test if leaf = blocked × blocked is in remaining updates and set up
        // temporary block matrix for recursion
        //

        auto  BC = std::unique_ptr< Hpro::TBlockMatrix< value_t > >(); // for recursive handling

        for ( auto  [ op_A, A, op_B, B ] : pending )
        {
            if ( is_blocked_all( *A, *B, M ) )
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
                if ( ! ( hlr::matrix::is_lowrank( M ) || hlr::matrix::is_lowrankS( M ) || hlr::matrix::is_uniform_lowrank( M ) ) )
                    HLR_ERROR( "unsupported matrix type: " + M.typestr() );
                
                auto  BA = ptrcast( A, Hpro::TBlockMatrix< value_t > );
                auto  BB = ptrcast( B, Hpro::TBlockMatrix< value_t > );
                
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

                break;
            }// if
        }// for
        
        //
        // now handle recursive updates if M is a leaf block
        //
    
        if ( ! is_null( BC ) )
        {
            //
            // first, split update matrix into subblock updates
            // (to release matrix before recursion)
            //

            auto  sub_accu = restrict( *BC );

            matrix.reset( nullptr );
        
            //
            // apply recursive updates
            //

            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,alpha] ( const auto &  r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            sub_accu(i,j).eval( alpha, *BC->block(i,j), acc, approx );

                            // replace block in BC by accumulator matrix for agglomeration below
                            BC->delete_block( i, j );
                            BC->set_block( i, j, sub_accu(i,j).release_matrix() );
                        }// for
                    }// for
                } );

            //
            // finally convert subblocks to single low-rank matrix for new accumulated updates
            //

            if ( handle_dense )
                matrix = seq::matrix::convert_to_dense( *BC );
            else
                matrix = seq::matrix::convert_to_lowrank( *BC, acc, approx );
        }// if
    }

    //
    // compute sum of uniform × uniform updates
    //
    blas::matrix< value_t >
    sum_uni_uni ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // for uniform x uniform only coefficients need to be added without
            // truncation due to shared bases:
            //
            //   U ( Σ_i S_i X_i' W_i T_i ) V' = U R V'  with R = Σ_i S_i X_i' W_i T_i
            //

            auto  [ op_A, A, op_B, B ] = updates[ lb ];

            auto  RA  = ptrcast( A, uniform_lrmatrix< value_t > );
            auto  RB  = ptrcast( B, uniform_lrmatrix< value_t > );

            auto  lock = std::scoped_lock( A->mutex(),
                                           B->mutex(),
                                           RA->col_cb( op_A ).mutex(),
                                           RB->row_cb( op_B ).mutex() );
                                           
            auto  S   = RA->coeff();
            auto  X   = RA->col_basis( op_A );
            auto  W   = RB->row_basis( op_B );
            auto  T   = RB->coeff();
            auto  XW  = blas::prod( blas::adjoint( X ), W );
            auto  SXW = blas::prod( blas::mat_view( op_A, S ), XW );

            return std::move( blas::prod( SXW, blas::mat_view( op_B, T ) ) );
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       R1  = blas::matrix< value_t >();
            auto       R2  = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&,lb,mid] () { R1 = sum_uni_uni( lb, mid, updates ); },
                                    [&,mid,ub] () { R2 = sum_uni_uni( mid, ub, updates ); } );
            
            blas::add( value_t(1), R1, R2 );

            return std::move( R2 );
        }// else
    }

    //
    // compute sum of uniform × anything updates
    //
    blas::matrix< value_t >
    sum_uni_any ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // U ( Σ_i S_i X_i' ) × B_i = U ( Σ_i S_i ( X_i' × B_i ) )
            //                          = U ( Σ_i S_i Z_i' )   with Z_i = B_i' × X_i
            //                          = U ( Σ_i (Z_i B_i'))'
            //                          = U Z'                 with Z   = Σ_i (Z_i S_i')
            //
            
            auto  [ op_A, A, op_B, B ] = updates[ lb ];

            auto  RA  = ptrcast( A, hlr::matrix::uniform_lrmatrix< value_t > );

            auto  lock = std::scoped_lock( RA->mutex(),
                                           RA->col_cb( op_A ).mutex() );
                                           
            auto  S   = RA->coeff();
            auto  X   = RA->col_basis( op_A );
            auto  Z_i = std::move( blas::matrix< value_t >( B->ncols(), X.ncols() ) );
                        
            hlr::multiply( value_t(1), blas::adjoint( op_B ), *B, X, Z_i );

            return blas::prod( Z_i, blas::adjoint( S ) );
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       Z1  = blas::matrix< value_t >();
            auto       Z2  = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&,lb,mid] () { Z1 = sum_uni_any( lb, mid, updates ); },
                                    [&,mid,ub] () { Z2 = sum_uni_any( mid, ub, updates ); } );

            blas::add( value_t(1), Z1, Z2 );

            return std::move( Z2 );
        }// else
    }
    
    //
    // compute sum of anything × uniform updates
    //
    blas::matrix< value_t >
    sum_any_uni ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates )
    {
        if ( ub - lb <= 1 )
        {
            //
            // A_i × ( Σ_i W_i S_i ) V' = ( Σ_i ( A_i × W_i ) S_i ) V'
            //                          = ( Σ_i Y_i S_i ) V' with Y_i = A_i × W_i
            //                          = ( Σ_i (Y_i S_i) ) V'
            //                          = Y V'
            //

            auto  [ op_A, A, op_B, B ] = updates[ lb ];

            auto  RB  = ptrcast( B, hlr::matrix::uniform_lrmatrix< value_t > );

            auto  lock = std::scoped_lock( RB->mutex(),
                                           RB->row_cb( op_B ).mutex() );
                                           
            auto  W   = RB->row_basis( op_B );
            auto  S   = RB->coeff();
            auto  Y_i = blas::matrix< value_t >( A->nrows(), W.ncols() );
            
            hlr::multiply( value_t(1), op_A, *A, W, Y_i );
            
            return blas::prod( Y_i, S );
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       Y1  = blas::matrix< value_t >();
            auto       Y2  = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&,lb,mid] () { Y1 = sum_any_uni( lb, mid, updates ); },
                                    [&,mid,ub] () { Y2 = sum_any_uni( mid, ub, updates ); } );

            blas::add( value_t(1), Y1, Y2 );

            return std::move( Y2 );
        }// else
    }
    
    //
    // compute sum of general updates with non-leaf types
    // for at least one factor
    //
    template < typename approx_t >
    std::unique_ptr< Hpro::TMatrix< value_t > >
    sum_pending ( const int                lb,
                  const int                ub,
                  std::vector< update > &  updates,
                  const bool               handle_dense,
                  const Hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
    {
        if ( ub - lb <= 1 )
        {
            auto  [ op_A, A, op_B, B ] = updates[ lb ];
            auto  T                    = std::unique_ptr< Hpro::TMatrix< value_t > >();

            if ( handle_dense ||
                 hlr::matrix::is_dense_all( A, B ) ||
                 ( is_blocked( A ) && hlr::matrix::is_dense(   B ) ) ||
                 ( hlr::matrix::is_dense(   A ) && is_blocked( B ) ))
                T = std::make_unique< Hpro::TDenseMatrix< value_t > >( A->row_is( op_A ), B->col_is( op_B ) );
            else
            {
                std::cout << "!!! : " << A->typestr() << " x " << B->typestr() << std::endl;
                T = std::make_unique< Hpro::TRkMatrix< value_t > >( A->row_is( op_A ), B->col_is( op_B ) );
            }// else

            //
            // in case of handle_dense, A or B may be uniform_lowrank and therefore need to be
            // locked before multiplication to prevent a race condition with basis updates
            //

            if ( hlr::matrix::is_uniform_lowrank_all( A, B ) )
            {
                auto  RA   = ptrcast( A, matrix::uniform_lrmatrix< value_t > );
                auto  RB   = ptrcast( B, matrix::uniform_lrmatrix< value_t > );
                auto  lock = std::scoped_lock( RA->mutex(),
                                               RB->mutex(),
                                               RA->col_cb( op_A ).mutex(),
                                               RB->row_cb( op_B ).mutex() );

                hlr::multiply< value_t >( value_t(1), op_A, *A, op_B, *B, *T, acc, approx );
            }// if
            else if ( hlr::matrix::is_uniform_lowrank( A ) )
            {
                auto  RA   = ptrcast( A, matrix::uniform_lrmatrix< value_t > );
                auto  lock = std::scoped_lock( RA->mutex(),
                                               RA->col_cb( op_A ).mutex() );

                hlr::multiply< value_t >( value_t(1), op_A, *A, op_B, *B, *T, acc, approx );
            }// if
            else if ( hlr::matrix::is_uniform_lowrank( B ) )
            {
                auto  RB   = ptrcast( B, matrix::uniform_lrmatrix< value_t > );
                auto  lock = std::scoped_lock( RB->mutex(),
                                               RB->row_cb( op_B ).mutex() );

                hlr::multiply< value_t >( value_t(1), op_A, *A, op_B, *B, *T, acc, approx );
            }// if
            else
                hlr::multiply< value_t >( value_t(1), op_A, *A, op_B, *B, *T, acc, approx );

            return T;
        }// if
        else
        {
            const int  mid = ( ub + lb ) / 2;
            auto       T1  = std::unique_ptr< Hpro::TMatrix< value_t > >();
            auto       T2  = std::unique_ptr< Hpro::TMatrix< value_t > >();

            ::tbb::parallel_invoke( [&,lb,mid,handle_dense] () { T1 = sum_pending( lb, mid, updates, handle_dense, acc, approx ); },
                                    [&,mid,ub,handle_dense] () { T2 = sum_pending( mid, ub, updates, handle_dense, acc, approx ); } );

            if ( hlr::matrix::is_dense( *T1 ) )
            {
                hlr::add( value_t(1), *T2, *T1 );

                return T1;
            }// if
            else if ( hlr::matrix::is_dense( *T2 ) )
            {
                hlr::add( value_t(1), *T1, *T2 );

                return T2;
            }// if
            else
            {
                hlr::add( value_t(1), *T2, *T1, acc, approx );

                return T1;
            }// else
        }// else
    }
    
    //
    // return true if given matrix is dense
    // - for structured matrices, only next level is tested
    //
    bool
    check_dense ( const Hpro::TMatrix< value_t > &  M ) const
    {
        // return false;
        if ( hlr::matrix::is_dense( M ) )
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
                    if ( ! is_null( B->block( i, j ) ) && ! hlr::matrix::is_dense( B->block( i, j ) ) )
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
// build block mappings (indexset to matrix block)
//
template < typename value_t >
void
build_block_maps ( Hpro::TMatrix< value_t > &    A,
                   is_matrix_map_t< value_t > &  rowmap,
                   is_matrix_map_t< value_t > &  colmap )
{
    auto  blocks = std::list< Hpro::TMatrix< value_t > * >{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = decltype( blocks )();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = ptrcast( M, Hpro::TBlockMatrix< value_t > );

                for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        if ( ! is_null( BM->block( i, j ) ) )
                            subblocks.push_back( BM->block( i, j ) );
            }// if
            else if ( is_uniform_lowrank( M ) )
            {
                rowmap[ M->row_is() ].push_back( M );
                colmap[ M->col_is() ].push_back( M );
            }// if
        }// for

        blocks = std::move( subblocks );
    }// while
}

//
// implements recursive LU factorization with accumulator
//
template < typename value_t >
struct rec_lu_factorization
{
    // maps indexsets to set of uniform matrices sharing corresponding cluster basis
    is_matrix_map_t< value_t >   rowmap_L, colmap_L;
    is_matrix_map_t< value_t >   rowmap_U, colmap_U;

    // mutices for mappings
    std::mutex                   rowmap_L_mtx, colmap_L_mtx;
    std::mutex                   rowmap_U_mtx, colmap_U_mtx;
    
    //
    // ctor
    //
    rec_lu_factorization ( Hpro::TMatrix< value_t > &  L,
                           Hpro::TMatrix< value_t > &  U )
    {}

    template < typename approx_t >
    void
    solve_lower_tri ( const eval_side_t                  side,
                      const diag_type_t                  diag,
                      Hpro::TMatrix< value_t > &         L,
                      Hpro::TMatrix< value_t > &         M,
                      accumulator< value_t > &           accu,
                      const Hpro::TTruncAcc &            acc,
                      const approx_t &                   approx,
                      shared_cluster_basis< value_t > &  rowcb,   // new cluster bases for M
                      shared_cluster_basis< value_t > &  colcb )
    {
        //
        // evaluate all computable updates to M
        //

        trace::region_start( "eval" );

        accu.eval( value_t(1), M, acc, approx );

        trace::region_end( "eval" );
    
        if ( is_blocked_all( L, M ) )
        {
            auto  BL = ptrcast( &L, Hpro::TBlockMatrix< value_t > );
            auto  BM = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
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
            
                    HLR_ASSERT( ! is_null( rowcb.son(i) ) );

                    ::tbb::parallel_for< uint >(
                        0, BM->nblock_cols(),
                        [&,side,diag,L_ii,BM,i] ( const uint  j )
                    {
                        HLR_ASSERT( ! is_null( colcb.son(j) ) );
                    
                        solve_lower_tri( side, diag, *L_ii, *BM->block(i,j),
                                         sub_accu(i,j), acc, approx,
                                         *rowcb.son(i), *colcb.son(j) );
                    } );

                    for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                        for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                            sub_accu(k,j).add_update( *BL->block(k,i), *BM->block(i,j) );
                }// for
            }// if
            else
            {
                HLR_ASSERT( false );
            }// else
        }// if
        else if ( hlr::matrix::is_uniform_lowrank( M ) )
        {
            // std::cout << M.id() << "  " << M.block_is().to_string() << std::endl;
            
            //
            // update and solve local matrix
            //
            // Remark: local basis may be changed in parallel but the numerical data of M
            //         is not (this task is repsonsible for this). So, when converting to
            //         standard low-rank format, all subsequent operations are only working
            //         with task-local data.
            //

            auto  UM = ptrcast( &M, hlr::matrix::uniform_lrmatrix< value_t > );
            auto  R  = Hpro::TRkMatrix< value_t >( M.row_is(), M.col_is() );

            {
                std::scoped_lock  lock( M.mutex(), rowcb.mutex(), colcb.mutex() );

                R.set_lrmat( std::move( blas::prod( UM->row_basis(), UM->coeff() ) ),
                             std::move( blas::copy( UM->col_basis() ) ) );
            }
            
            trace::region_start( "apply" );
            
            // no recursive updates left, apply accumulated updates and solve
            accu.apply( value_t(-1), R, acc, approx );
        
            trace::region_end( "apply" );
            
            hlr::solve_lower_tri( side, diag, L, R, acc, approx );

            //
            // now replace M by R and update row/column bases
            //

            trace::region_start( "basis" );
        
            auto  W  = std::move( blas::mat_U< value_t >( R ) );
            auto  X  = std::move( blas::mat_V< value_t >( R ) );
            auto  RW = blas::matrix< value_t >();
            auto  RX = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&] () { blas::qr( W, RW ); },
                                    [&] () { blas::qr( X, RX ); } );

            auto  T  = blas::prod( RW, blas::adjoint( RX ) );

            // Remark: now we leave the pure-local part and need to synchonise with 
            //         parallel tasks possibly changing the row/column bases
            
            ::tbb::this_task_arena::isolate( [&] ()
            {
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                ::tbb::parallel_invoke(
                    [&] ()
                    {
                        auto  Un = compute_extended_basis( rowcb, W, T, acc, approx, rowmap_U, rowmap_U_mtx, apply_adjoint );

                        update_coupling( rowcb, Un, rowmap_U, rowmap_U_mtx, false );
                        rowcb.set_basis( std::move( Un ) );
                    },
                        
                    [&] ()
                    {
                        auto  Vn = compute_extended_basis( colcb, X, T, acc, approx, colmap_U, colmap_U_mtx, apply_normal );
            
                        update_coupling( colcb, Vn, colmap_U, colmap_U_mtx, true );
                        colcb.set_basis( std::move( Vn ) );
                    }
                );

                //
                // update new basis and replace data in M
                //
        
                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  Sn = blas::prod( TS, blas::adjoint( TV ) );

                UM->set_matrix_data( rowcb, std::move( Sn ), colcb );

                // M now also part of matrices sharing rowcb/colcb
                {
                    auto  lock_is = std::scoped_lock( rowmap_U_mtx, colmap_U_mtx );
                    
                    rowmap_U[ rowcb.is() ].push_back( UM );
                    colmap_U[ colcb.is() ].push_back( UM );
                }
                
                trace::region_end( "basis" );
            } );
        }// if
        else
        {
            accu.apply( value_t(-1), M, acc, approx );

            hlr::solve_lower_tri( side, diag, L, M, acc, approx );
        }// else
    }

    template < typename approx_t >
    void
    solve_upper_tri ( const eval_side_t                  side,
                      const diag_type_t                  diag,
                      Hpro::TMatrix< value_t > &         U,
                      Hpro::TMatrix< value_t > &         M,
                      accumulator< value_t > &           accu,
                      const Hpro::TTruncAcc &            acc,
                      const approx_t &                   approx,
                      shared_cluster_basis< value_t > &  rowcb,   // new cluster bases for M
                      shared_cluster_basis< value_t > &  colcb )
    {
        //
        // evaluate all computable updates to M
        //

        trace::region_start( "eval" );
    
        accu.eval( value_t(1), M, acc, approx );
    
        trace::region_end( "eval" );
        
        if ( is_blocked_all( U, M ) )
        {
            auto  BU = ptrcast( &U, Hpro::TBlockMatrix< value_t > );
            auto  BM = ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
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
                    
                    HLR_ASSERT( ! is_null_any( U_jj, colcb.son(j) ) );

                    ::tbb::parallel_for< uint >(
                        0, BM->nblock_rows(),
                        [&,side,diag,U_jj,BM,j] ( const uint  i )
                    {
                        HLR_ASSERT( ! is_null( rowcb.son(i) ) );
                    
                        solve_upper_tri( side, diag, *U_jj, *BM->block( i, j ),
                                         sub_accu(i,j), acc, approx, 
                                         *rowcb.son(i), *colcb.son(j) );
                    } );
            
                    for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                            sub_accu(i,k).add_update( *BM->block(i,j), *BU->block(j,k) );
                }// for
            }// else
        }// if
        else if ( hlr::matrix::is_uniform_lowrank( M ) )
        {
            // std::cout << M.id() << "  " << M.block_is().to_string() << std::endl;
            
            //
            // update and solve local matrix
            //
            // Remark: local basis may be changed in parallel but the numerical data of M
            //         is not (this task is repsonsible for this). So, when converting to
            //         standard low-rank format, all subsequent operations are only working
            //         with task-local data.
            //

            auto  UM = ptrcast( &M, uniform_lrmatrix< value_t > );
            auto  R  = Hpro::TRkMatrix< value_t >( M.row_is(), M.col_is() );
                                             
            {
                std::scoped_lock  lock( M.mutex(), rowcb.mutex(), colcb.mutex() );

                R.set_lrmat( std::move( blas::prod( UM->row_basis(), UM->coeff() ) ),
                             std::move( blas::copy( UM->col_basis() ) ) );
            }

            trace::region_start( "apply" );
        
            // no recursive updates left, apply accumulated updates and solve
            accu.apply( value_t(-1), R, acc, approx );

            trace::region_end( "apply" );

            hlr::solve_upper_tri( side, diag, U, R, acc, approx );

            //
            // now replace M by R and update row/column bases
            //

            trace::region_start( "basis" );
        
            auto  W  = std::move( blas::mat_U< value_t >( R ) );
            auto  X  = std::move( blas::mat_V< value_t >( R ) );
            auto  RW = blas::matrix< value_t >();
            auto  RX = blas::matrix< value_t >();

            ::tbb::parallel_invoke( [&] () { blas::qr( W, RW ); },
                                    [&] () { blas::qr( X, RX ); } );

            auto  T = blas::prod( RW, blas::adjoint( RX ) );

            // Remark: now we leave the pure-local part and need to synchonise with 
            //         parallel tasks possibly changing the row/column bases

            ::tbb::this_task_arena::isolate( [&] ()
            {
                auto  lock_cb = std::scoped_lock( rowcb.mutex(), colcb.mutex() );
                
                ::tbb::parallel_invoke(
                    [&] ()
                    {
                        auto  Un = compute_extended_basis( rowcb, W, T, acc, approx, rowmap_L, rowmap_L_mtx, apply_adjoint );

                        update_coupling( rowcb, Un, rowmap_L, rowmap_L_mtx, false );
                        rowcb.set_basis( std::move( Un ) );
                    },
                        
                    [&] ()
                    {
                        auto  Vn = compute_extended_basis( colcb, X, T, acc, approx, colmap_L, colmap_L_mtx, apply_normal );
            
                        update_coupling( colcb, Vn, colmap_L, colmap_L_mtx, true );
                        colcb.set_basis( std::move( Vn ) );
                    }
                );

                //
                // update new basis and replace data in M
                //
                
                auto  TU = blas::prod( blas::adjoint( rowcb.basis() ), W );
                auto  TV = blas::prod( blas::adjoint( colcb.basis() ), X );
                auto  TS = blas::prod( TU, T );
                auto  Sn = blas::prod( TS, blas::adjoint( TV ) );
                
                UM->set_matrix_data( rowcb, std::move( Sn ), colcb );

                // M now also part of matrices sharing rowcb/colcb
                {
                    auto  lock_is = std::scoped_lock( rowmap_L_mtx, colmap_L_mtx );
                    
                    rowmap_L[ rowcb.is() ].push_back( UM );
                    colmap_L[ colcb.is() ].push_back( UM );
                }

                trace::region_end( "basis" );
            } );
        }// if
        else
        {
            accu.apply( value_t(-1), M, acc, approx );
            
            hlr::solve_upper_tri( side, diag, U, M, acc, approx );
        }// else
    }
                      
    //
    // recursive LU factorization
    //
    template < typename approx_t >
    void
    lu ( Hpro::TMatrix< value_t > &         A,
         Hpro::TMatrix< value_t > &         L,
         Hpro::TMatrix< value_t > &         U,
         accumulator< value_t > &           accu,
         const Hpro::TTruncAcc &            acc,
         const approx_t &                   approx,
         shared_cluster_basis< value_t > &  rowcb_L, // new cluster bases for L
         shared_cluster_basis< value_t > &  colcb_L,
         shared_cluster_basis< value_t > &  rowcb_U, // new cluster bases for U
         shared_cluster_basis< value_t > &  colcb_U )
    {
        //
        // evaluate all computable updates to M
        //

        trace::region_start( "eval" );
    
        accu.eval( value_t(1), A, acc, approx );
    
        trace::region_end( "eval" );

        //
        // (recursive) LU factorization
        //

        if ( is_blocked( A ) )
        {
            auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
            auto  BL = ptrcast( &L, Hpro::TBlockMatrix< value_t > );
            auto  BU = ptrcast( &U, Hpro::TBlockMatrix< value_t > );

            //
            // first, split accumulated updates U and recursive updates upd_rec
            // into subblock updates
            // - to release U before recursion and by that avoid memory
            //   consumption dependent on hierarchy depth
            //

            auto  sub_accu = accu.restrict( *BA );

            accu.clear_matrix();

            //
            // recursive LU factorization but add updates to accumulator
            // instead of applying them
            //
        
            for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
            {
                HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
                lu( * BA->block( i, i ), * BL->block( i, i ), * BU->block( i, i ),
                    sub_accu(i,i), acc, approx,
                    *rowcb_L.son(i), *colcb_L.son(i),
                    *rowcb_U.son(i), *colcb_U.son(i) );

                ::tbb::parallel_invoke( 
                    [&,i,BA,BU,BL] ()
                    {
                        ::tbb::parallel_for< uint >(
                            i+1, BA->nblock_rows(),
                            [&,i,BA,BU,BL] ( const uint  j )
                            {
                                if ( ! is_null( BA->block( j, i ) ) )
                                    solve_upper_tri( from_right, general_diag,
                                                     *BU->block( i, i ), *BL->block( j, i ),
                                                     sub_accu(j,i), acc, approx,
                                                     *rowcb_L.son(j), *colcb_L.son(i) );
                            } );
                    },
                    
                    [&,i,BA,BU,BL] ()
                    {
                        ::tbb::parallel_for< uint >(
                            i+1, BA->nblock_cols(),
                            [&,i,BA,BU,BL] ( const uint  j )
                            {
                                if ( ! is_null( BA->block( i, j ) ) )
                                    solve_lower_tri( from_left, unit_diag,
                                                     *BL->block( i, i ), *BU->block( i, j ),
                                                     sub_accu(i,j), acc, approx,
                                                     *rowcb_U.son(i), *colcb_U.son(j) );
                            } );
                    }
                );

                for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
                    for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                        if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                            sub_accu(j,l).add_update( *BL->block( j, i ), *BU->block( i, l ) );
            }// for
        }// if
        else if ( hlr::matrix::is_dense( A ) )
        {
            auto  DA = ptrcast( &A, matrix::dense_matrix< value_t > );
            auto  DU = ptrcast( &U, matrix::dense_matrix< value_t > );

            accu.apply( value_t(-1), A, acc, approx );

            auto  Am = DA->mat();
            auto  Um = DU->mat();
        
            blas::copy( Am, Um );
            blas::invert( Um );

            if ( DU->is_compressed() )
                DU->set_matrix( std::move( Um ), acc );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }
};

}}}}}// namespace hlr::tbb::uniform::accu::detail

#endif // __HLR_TBB_DETAIL_UNIFORM_ACCU_LU_HH
