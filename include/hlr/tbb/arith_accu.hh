#ifndef __HLR_TBB_ARITH_ACCU_HH
#define __HLR_TBB_ARITH_ACCU_HH
//
// Project     : HLib
// File        : arith.hh
// Description : arithmetic functions using accumulators
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/algebra/mat_add.hh>
#include <hpro/algebra/mat_conv.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/arith/blas.hh"
#include "hlr/arith/multiply.hh"
#include "hlr/arith/add.hh"
#include "hlr/tbb/accumulator.hh"

namespace hlr { namespace tbb { namespace accu {

namespace hpro = HLIB;

using hlr::tbb::matrix::accumulator;

////////////////////////////////////////////////////////////////////////////////
//
// accumulator based matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

using  upd_list_t = std::list< std::pair< const hpro::TMatrix *, const hpro::TMatrix * > >;

//
// forward decl.
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TMatrix &                     C,
           upd_list_t &                        upd_C,
           std::unique_ptr< hpro::TMatrix > &  upd_accu,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx );

//
// special case : C is low-rank matrix
// - construct sub-blocks of C for all blocked updates with corresponding accumulators
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TRkMatrix &                   C,
           upd_list_t &                        upd_C,
           std::unique_ptr< hpro::TMatrix > &  upd_accu,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx )
{
    //
    // first handle updates to C with at least one leaf block
    //

    std::unique_ptr< hpro::TMatrix >       U( std::move( upd_accu ) );
    std::unique_ptr< hpro::TBlockMatrix >  BC;
    ::tbb::affinity_partitioner            ap; // to ensure (???) allocated data is used in equal threads
    
    for ( auto  [ A, B ] : upd_C )
    {
        if ( is_blocked_all( *A, *B ) )
        {
            //
            // set up block matrix with low-rank sub-blocks
            //

            if ( is_null( BC ) )
            {
                auto  BA = cptrcast( A, hpro::TBlockMatrix );
                auto  BB = cptrcast( B, hpro::TBlockMatrix );
                
                BC = std::make_unique< hpro::TBlockMatrix >( C.row_is(), C.col_is() );

                BC->set_block_struct( BA->nblock_rows( op_A ), BB->nblock_cols( op_B ) );

                ::tbb::parallel_for(
                    ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                    0, BC->nblock_cols() ),
                    [&,alpha,op_A,op_B,BA,BB] ( const auto & r )
                    {
                        for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                            for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                            {
                                HLR_ASSERT( ! is_null_any( BA->block( i, 0, op_A ), BB->block( 0, j, op_B ) ) );
                                
                                BC->set_block( i, j, new hpro::TRkMatrix( BA->block( i, 0, op_A )->row_is( op_A ),
                                                                          BB->block( 0, j, op_B )->col_is( op_B ),
                                                                          C.value_type() ) );
                            }// for
                    },
                    ap );
            }// if
        }// if
        else
        {
            auto  T = multiply< value_t >( alpha, op_A, A, op_B, B );
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else if ( is_dense( T.get() ) )
            {
                hlr::add( value_t(1), *U, *T, acc, approx );
                U = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *U, acc, approx );
            }// else
        }// else
    }// for

    //
    // now handle recursive updates
    //
    
    if ( ! is_null( BC ) )
    {
        //
        // first, split update U into subblock updates
        // (to release U before recursion)
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_U( BC->nblock_rows(), BC->nblock_cols() );
        
        if ( ! is_null( U ) )
        {
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,alpha,op_A,op_B] ( const auto & r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                        for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                            sub_U(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );
                },
                ap );
            
            U.reset( nullptr );
        }// if

        //
        // apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                            0, BC->nblock_cols() ),
            [&,alpha,op_A,op_B] ( const auto & r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        upd_list_t  upd_ij;
                
                        for ( auto  [ A, B ] : upd_C )
                        {
                            if ( ! is_blocked_all( A, B ) )
                                continue;
                            
                            auto  BA = cptrcast( A, hpro::TBlockMatrix );
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );
                        
                            for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                            {
                                auto  A_il = BA->block( i, l, op_A );
                                auto  B_lj = BB->block( l, j, op_B );
                                
                                if ( is_null_any( A_il, B_lj ) )
                                    continue;
                                
                                upd_ij.push_back( { A_il, B_lj } );
                            }// for
                        }// for

                        HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                        multiply< value_t >( alpha, op_A, op_B, *BC->block( i, j ), upd_ij, sub_U( i, j ), acc, approx );
                    }// for
                }// for
            },
            ap );

        //
        // finally convert subblocks to single low-rank matrix for new accumulated updates
        //

        if ( ! is_null( U ) )
            HLR_ERROR( "accumulator non-null" );
        
        U = to_rank( BC.get(), acc );
    }// if

    //
    // apply all accumulated updates
    //

    if ( ! is_null( U ) )
        hlr::add( alpha, *U, C, acc, approx );
}

//
// general version to compute C = C + α op( A ) op( B )
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                       alpha,
           const hpro::matop_t                 op_A,
           const hpro::matop_t                 op_B,
           hpro::TMatrix &                     C,
           upd_list_t &                        upd_C,
           std::unique_ptr< hpro::TMatrix > &  upd_accu,
           const hpro::TTruncAcc &             acc,
           const approx_t &                    approx )
{
    if ( is_lowrank( C ) )
    {
        multiply< value_t >( alpha, op_A, op_B, *ptrcast( &C, hpro::TRkMatrix ), upd_C, upd_accu, acc, approx );
        return;
    }// if
    
    //
    // first handle computable updates to C, including non-blocked C
    //

    std::unique_ptr< hpro::TMatrix >  U( std::move( upd_accu ) );
    
    for ( auto  [ A, B ] : upd_C )
    {
        if ( ! is_blocked_all( *A, *B, C ) )
        {
            auto  T = std::unique_ptr< hpro::TMatrix >();
            
            if ( is_blocked_all( *A, *B ) )
            {
                HLR_ERROR( C.typestr() + " += blocked × blocked" );
                
                T = std::make_unique< hpro::TRkMatrix >( C.row_is(), C.col_is(), hpro::value_type< value_t >::value );

                hpro::multiply( value_t(1), op_A, A, op_B, B, value_t(0), T.get(), acc );
            }// if
            else
            {
                // either A or B is low-rank or dense
                T = multiply< value_t >( alpha, op_A, A, op_B, B );
            }// else
            
            if ( is_null( U ) )
            {
                U = std::move( T );
            }// if
            else if ( is_dense( T.get() ) )
            {
                hlr::add( value_t(1), *U, *T, acc, approx );
                U = std::move( T );
            }// if
            else
            {
                hlr::add( value_t(1), *T, *U, acc, approx );
            }// else
        }// if
    }// for

    //
    // now handle recursive updates
    //
    
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

        //
        // first, split update U into subblock updates
        // (to release U before recursion)
        //

        tensor2< std::unique_ptr< hpro::TMatrix > >  sub_U( BC->nblock_rows(), BC->nblock_cols() );
        ::tbb::affinity_partitioner            ap;
        
        if ( ! is_null( U ) )
        {
            ::tbb::parallel_for(
                ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                                0, BC->nblock_cols() ),
                [&,BC,alpha,op_A,op_B] ( const auto & r )
                {
                    for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    {
                        for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                        {
                            if ( ! is_null( BC->block( i, j ) ) )
                                sub_U(i,j) = hlr::matrix::restrict( *U, BC->block( i, j )->block_is() );
                        }// for
                    }// for
                },
                ap );
            
            U.reset( nullptr );
        }// if

        //
        // now apply recursive multiplications, e.g.,
        // collect all sub-products and recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                            0, BC->nblock_cols() ),
            [&,BC,alpha,op_A,op_B] ( const auto & r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  C_ij = BC->block(i,j);

                        HLR_ASSERT( ! is_null( C_ij ) );

                        upd_list_t  upd_ij;
                
                        for ( auto  [ A, B ] : upd_C )
                        {
                            if ( ! is_blocked_all( A, B ) )
                                continue;
                            
                            auto  BA = cptrcast( A, hpro::TBlockMatrix );
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );
                                
                            for ( uint  l = 0; l < BA->nblock_cols( op_A ); ++l )
                            {
                                auto  A_il = BA->block( i, l, op_A );
                                auto  B_lj = BB->block( l, j, op_B );
                                    
                                if ( is_null_any( A_il, B_lj ) )
                                    continue;
                                    
                                upd_ij.push_back( { A_il, B_lj } );
                            }// for
                        }// for
                        
                        multiply< value_t >( alpha, op_A, op_B, *C_ij, upd_ij, sub_U( i, j ), acc, approx );
                    }// for
                }// for
            },
            ap );
    }// if
    else
    {
        // apply accumulated updates
        hlr::add( alpha, *U, C, acc, approx );
    }// else
}

//
// compute C = C + α op( A ) op( B ) where A and B are provided as accumulated updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           hpro::TMatrix &          C,
           accumulator &            accu,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
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
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

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
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    std::unique_ptr< hpro::TMatrix >  U;

    #if 0
    
    detail::upd_list_t                upd{ { &A, &B } };
    
    detail::multiply< value_t >( alpha, op_A, op_B, C, upd, U, acc, approx );

    #else

    accumulator::update_list   upd{ { op_A, &A, op_B, &B } };
    accumulator                accu( std::move( U ), std::move( upd ) );
                       
    detail::multiply< value_t >( alpha, C, accu, acc, approx );
    
    #endif
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
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  accumulator &            accu,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );
    
    if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        
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

                ::tbb::parallel_for< uint >( 0, BM->nblock_cols(),
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
                  const hpro::TMatrix &               U,
                  hpro::TMatrix &                     M,
                  accumulator &                       accu,
                  const hpro::TTruncAcc &             acc,
                  const approx_t &                    approx )
{
    // apply computable updates
    accu.eval( value_t(1), M, acc, approx );
    
    if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        
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
            
                ::tbb::parallel_for< uint >( 0, BM->nblock_rows(),
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
lu ( hpro::TMatrix &          M,
     accumulator &            accu,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
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
        auto  BM = ptrcast( &M, hpro::TBlockMatrix );

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
                    ::tbb::parallel_for< uint >( i+1, BM->nblock_rows(),
                                                 [=,&sub_accu,&acc,&approx] ( const uint j )
                                                 {
                                                     solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );
                                                 } );
                },

                [=,&sub_accu,&acc,&approx]
                {
                    ::tbb::parallel_for< uint >( i+1, BM->nblock_cols(),
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
        
        if ( is_dense( M ) )
        {
            auto  D = ptrcast( &M, hpro::TDenseMatrix );

            invert< value_t >( *D );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          M,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    accumulator  accu;
    
    detail::lu< value_t >( M, accu, acc, approx );
}

}}}// namespace hlr::tbb::accu

#endif // __HLR_TBB_ARITH_ACCU_HH
