#ifndef __HLR_TF_ARITH_LAZY_HH
#define __HLR_TF_ARITH_LAZY_HH
//
// Project     : HLib
// Module      : tf/arith_lazy
// Description : arithmetic functions for lazy evaluation using TF
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <taskflow/taskflow.hpp>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TMatrixProduct.hh>
#include <hpro/matrix/structure.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/utils/checks.hh"

namespace hlr { namespace tf { namespace lazy {

namespace hpro = HLIB;

////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

//////////////////////////////////////////////////////////////////////
//
// operator wrapper for sum of linear operators
//
//////////////////////////////////////////////////////////////////////

template < typename T_value >
struct linopsum_operator
{
    using  value_t = T_value;

    const std::list< const hpro::TLinearOperator * > &  ops;

    linopsum_operator ( const std::list< const hpro::TLinearOperator * > &  aops )
            : ops( aops )
    {
        HLR_ASSERT( ! aops.empty() );
    }
    
    size_t  nrows () const { return ops.front()->range_dim(); }
    size_t  ncols () const { return ops.front()->domain_dim(); }
    
    blas::vector< value_t >
    get_row ( const size_t  j ) const
    {
        auto  e_j = blas::vector< value_t >( nrows() );
        auto  row = blas::vector< value_t >( ncols() );

        e_j(j) = value_t(1);
        
        for ( auto  op : ops )
            op->apply_add( value_t(1), e_j, row, apply_transposed );
        
        return row;
    }

    blas::vector< value_t >
    get_column ( const size_t  j ) const
    {
        auto  e_j = blas::vector< value_t >( ncols() );
        auto  col = blas::vector< value_t >( nrows() );

        e_j(j) = value_t(1);
        
        for ( auto  op : ops )
            op->apply_add( value_t(1), e_j, col, apply_normal );
        
        return col;
    }
        
    void
    prod ( const value_t                    alpha,
           const matop_t                    op_M,
           const blas::vector< value_t > &  x,
           blas::vector< value_t > &        y ) const
    {
        for ( auto  op : ops )
            op->apply_add( alpha, x, y, op_M );
    }

    void
    prod ( const value_t                    alpha,
           const matop_t                    op_M,
           const blas::matrix< value_t > &  X,
           blas::matrix< value_t > &        Y ) const
    {
        for ( auto  op : ops )
            op->apply_add( alpha, X, Y, op_M );
    }
};

template < typename value_t > size_t  nrows ( const linopsum_operator< value_t > &  op ) { return op.nrows(); }
template < typename value_t > size_t  ncols ( const linopsum_operator< value_t > &  op ) { return op.ncols(); }

template < typename value_t >
blas::vector< value_t >
get_row ( const linopsum_operator< value_t > &  op,
          const size_t                          i )
{
    return op.get_row( i );
}

template < typename value_t >
blas::vector< value_t >
get_column ( const linopsum_operator< value_t > &  op,
             const size_t                          j )
{
    return op.get_column( j );
}

template < typename value_t >
void
prod ( const value_t                         alpha,
       const matop_t                         op_M,
       const linopsum_operator< value_t > &  M,
       const blas::vector< value_t > &       x,
       blas::vector< value_t > &             y )
{
    M.prod( alpha, op_M, x, y );
}

template < typename value_t >
void
prod ( const value_t                         alpha,
       const matop_t                         op_M,
       const linopsum_operator< value_t > &  M,
       const blas::matrix< value_t > &       X,
       blas::matrix< value_t > &             Y )
{
    M.prod( alpha, op_M, X, Y );
}

template < typename value_t >
linopsum_operator< value_t >
operator_wrapper ( const std::list< const hpro::TLinearOperator * > &  M )
{
    return linopsum_operator< value_t > ( M );
}

//
// accumulator for lazy updates
//
struct lazy_accumulator
{
    using  update_t      = std::pair< const hpro::TMatrix *, const hpro::TMatrix * >;
    using  update_list_t = std::list< update_t >;
    using  create_list_t = std::list< std::unique_ptr< hpro::TMatrix > >;

    // list of collected updates
    update_list_t  updates;

    // list of created wrapper matrices
    create_list_t  created;

    // signals dense handling of updates
    bool           handle_dense = false;

    //
    // add A×B to updates
    //
    template < typename value_t >
    void
    add_update ( const hpro::TMatrix *  A,
                 const hpro::TMatrix *  B )
    {
        HLR_ASSERT( ! is_null( A ) );
        
        if ( is_lowrank_all( A, B ) )
        {
            //
            // create U·T·V' representation with T=V(A)'·U(B)
            //

            auto  RA = cptrcast( A, hpro::TRkMatrix );
            auto  RB = cptrcast( B, hpro::TRkMatrix );

            auto  U  = blas::mat_U< value_t >( RA );
            auto  T  = blas::prod( blas::adjoint( blas::mat_V< value_t >( RA ) ), blas::mat_U< value_t >( RB ) );
            auto  V  = blas::mat_V< value_t >( RB );
            
            auto  AxB = std::make_unique< hlr::matrix::lrsmatrix< value_t > >( A->row_is(), B->col_is(), std::move( U ), std::move( T ), std::move( V ) );
            
            updates.push_back( { AxB.get(), nullptr } );
            created.push_back( std::move( AxB ) );
        }// if
        else
        {
            if ( is_dense_all( A, B ) )
                handle_dense = true;
            
            updates.push_back( { A, B } );
        }// else
    }

    template < typename value_t >
    void
    add_update ( const hpro::TMatrix *    A )
    {
        add_update< value_t >( A, nullptr );
    }
    
    //
    // split given set of updates to set of updates for sub blocks
    //
    template < typename value_t >
    tensor2< lazy_accumulator >
    split ( const hpro::TBlockMatrix &  M )
    {
        using  hlr::matrix::lrsmatrix;
        using  hlr::matrix::is_lowrankS;
        
        auto  sub_accu = tensor2< lazy_accumulator >( M.nblock_rows(), M.nblock_cols() );

        for ( uint  i = 0; i < M.nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < M.nblock_cols(); ++j )
            {
                auto  M_ij  = M.block( i, j );
                auto  row_i = M_ij->row_is();
                auto  col_j = M_ij->col_is();

                for ( auto  [ A, B ] : updates )
                {
                    //
                    // extract sub-update for M_ij based on types of A/B
                    //

                    if ( is_blocked( A ) )
                    {
                        auto  BA = cptrcast( A, hpro::TBlockMatrix );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null_all( BA->block( i, k ), BB->block( k, j ) ) );

                                sub_accu(i,j).add_update< value_t >( BA->block( i, k ), BB->block( k, j ) );
                            }// for
                        }// if
                        else if ( is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, hpro::TRkMatrix );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  U_k = blas::matrix< value_t >( blas::mat_U< value_t >( RB ), col_k - B->row_ofs(), blas::range::all );
                                auto  V_j = blas::matrix< value_t >( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                                auto  B_kj = std::make_unique< hpro::TRkMatrix >( col_k, col_j, U_k, V_j );
                                
                                sub_accu(i,j).add_update< value_t >( A_ik, B_kj.get() );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else if ( is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, hpro::TDenseMatrix );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  D_kj = blas::matrix< value_t >( blas::mat< value_t >( DB ), col_k - B->row_ofs(), col_j - B->col_ofs() );
                                auto  B_kj = std::make_unique< hpro::TDenseMatrix >( col_k, col_j, D_kj );
                                
                                sub_accu(i,j).add_update< value_t >( A_ik, B_kj.get() );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( is_lowrank( A ) )
                    {
                        auto  RA = cptrcast( A, hpro::TRkMatrix );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  U_i  = blas::matrix< value_t >( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                                auto  V_k  = blas::matrix< value_t >( blas::mat_V< value_t >( RA ), col_k - A->col_ofs(), blas::range::all );
                                auto  A_ik = std::make_unique< hpro::TRkMatrix >( row_i, col_k, U_i, V_k );
                                
                                sub_accu(i,j).add_update< value_t >( A_ik.get(), B_kj );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, hpro::TRkMatrix );

                            // restrict A to row_i
                            auto  U_i = blas::matrix< value_t >( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TRkMatrix >( row_i, A->col_is(), U_i, blas::mat_V< value_t >( RA ) );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix< value_t >( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< hpro::TRkMatrix >( B->row_is(), col_j, blas::mat_U< value_t >( RB ), V_j );
                                
                            sub_accu(i,j).add_update< value_t >( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, hpro::TDenseMatrix );

                            // restrict A to row_i
                            auto  U_i = blas::matrix< value_t >( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TRkMatrix >( row_i, A->col_is(), U_i, blas::mat_V< value_t >( RA ) );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix< value_t >( blas::mat< value_t >( DB ), blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< hpro::TDenseMatrix >( B->row_is(), col_j, D_j );
                                
                            sub_accu(i,j).add_update< value_t >( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( is_lowrankS( A ) )
                    {
                        auto  RA = cptrcast( A, lrsmatrix< value_t > );

                        // A already holds full product
                        HLR_ASSERT( is_null( B ) );

                        // A already holds product of original A and B so just restrict to M_ij
                        auto  U_i  = blas::matrix< value_t >( RA->U(), row_i - A->row_ofs(), blas::range::all );
                        auto  V_j  = blas::matrix< value_t >( RA->V(), col_j - A->col_ofs(), blas::range::all );
                        auto  A_ij = std::make_unique< lrsmatrix< value_t > >( row_i, col_j, U_i, RA->S(), V_j );

                        sub_accu(i,j).add_update< value_t >( A_ij.get() );
                        created.push_back( std::move( A_ij ) );
                    }// if
                    else if ( is_dense( A ) )
                    {
                        auto  DA = cptrcast( A, hpro::TDenseMatrix );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, hpro::TBlockMatrix );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  D_ik = blas::matrix< value_t >( blas::mat< value_t >( DA ), row_i - A->row_ofs(), col_k - A->col_ofs() );
                                auto  A_ik = std::make_unique< hpro::TDenseMatrix >( row_i, col_k, D_ik );
                                
                                sub_accu(i,j).add_update< value_t >( A_ik.get(), B_kj );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, hpro::TRkMatrix );
                            
                            // restrict A to row_i
                            auto  D_i = blas::matrix< value_t >( blas::mat< value_t >( DA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TDenseMatrix >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix< value_t >( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< hpro::TRkMatrix >( B->row_is(), col_j, blas::mat_U< value_t >( RB ), V_j );
                                
                            sub_accu(i,j).add_update< value_t >( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, hpro::TDenseMatrix );

                            // restrict A to row_i
                            auto  D_i = blas::matrix< value_t >( blas::mat< value_t >( DA ), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< hpro::TDenseMatrix >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix< value_t >( blas::mat< value_t >( DB ), blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< hpro::TDenseMatrix >( B->row_is(), col_j, D_j );
                                
                            sub_accu(i,j).add_update< value_t >( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type: " + A->typestr() );
                }// for
            }// for
        }// for
    
        return sub_accu;
    }

    //
    // apply updates to leaf matrix
    //
    template < typename value_t,
               typename approx_t >
    void
    apply_leaf ( const value_t            alpha,
                 hpro::TMatrix &          M,
                 const hpro::TTruncAcc &  acc,
                 const approx_t &         approx )
    {
        using  hlr::matrix::lrsmatrix;
        using  hlr::matrix::is_lowrankS;
        
        if ( is_lowrank( M ) )
        {
            auto  R = ptrcast( &M, hpro::TRkMatrix );

            //
            // an update resulted in a dense matrix, so handle all in dense format
            //
            
            if ( handle_dense )
            {
                auto  D = hlr::matrix::convert_to_dense< value_t >( *R );

                apply_leaf( alpha, *D, acc, approx );
                
                auto  [ U, V ] = approx( blas::mat< value_t >( D ), acc );
                
                R->set_lrmat( std::move( U ), std::move( V ) );
            }// if
            else
            {
                //
                // otherwise handle updates in low-rank format
                //
            
                if ( approx_t::supports_general_operator )
                {
                    //
                    // set up operator for sum of matrix products plus C
                    //
        
                    auto  op_list = std::list< const hpro::TLinearOperator * >();
                    auto  deleted = std::list< const hpro::TLinearOperator * >();

                    for ( auto  [ A, B ] : updates )
                    {
                        if ( is_lowrankS( A ) )
                        {
                            HLR_ASSERT( is_null( B ) );
                        
                            auto  op_AxB = hpro::matrix_product( alpha, A );
                        
                            deleted.push_back( op_AxB.get() );
                            op_list.push_back( op_AxB.release() );
                        }// if
                        else
                        {
                            auto  op_AxB = hpro::matrix_product( alpha, A, value_t(1), B );
                        
                            deleted.push_back( op_AxB.get() );
                            op_list.push_back( op_AxB.release() );
                        }// else
                    }// for
            
                    op_list.push_back( R );

                    auto  sumop = operator_wrapper< value_t >( op_list );
                
                    //
                    // apply lowrank approximation and update C
                    //
                
                    auto  [ U, V ] = approx( sumop, acc );
                
                    R->set_lrmat( std::move( U ), std::move( V ) );
                
                    //
                    // clean up
                    //
                
                    for ( auto  op : deleted )
                        delete op;
                }// if
                else
                {
                    //
                    // add each update individually
                    //
                
                    for ( auto  [ A, B ] : updates )
                    {
                        if ( is_lowrankS( A ) )
                        {
                            HLR_ASSERT( is_null( B ) );
                    
                            auto  RA = cptrcast( A, lrsmatrix< value_t > );
                            auto  US = blas::prod( alpha, RA->U(), RA->S() );

                            auto  [ U, V ] = approx( { blas::mat_U< value_t >( R ), US },
                                                     { blas::mat_V< value_t >( R ), RA->V() },
                                                     acc );
            
                            R->set_lrmat( std::move( U ), std::move( V ) );
                        }// if
                        else
                        {
                            hlr::multiply( alpha, apply_normal, *A, apply_normal, *B, M, acc, approx );
                        }// else
                    }// for
                }// else
            }// if
        }// if
        else if ( is_dense( M ) )
        {
            auto  D = ptrcast( &M, hpro::TDenseMatrix );

            for ( auto  [ A, B ] : updates )
            {
                if ( is_lowrankS( A ) )
                {
                    HLR_ASSERT( is_null( B ) );
                    
                    auto  RA = cptrcast( A, lrsmatrix< value_t > );
                    auto  US = blas::prod( RA->U(), RA->S() );

                    blas::prod( alpha, US, blas::adjoint( RA->V() ), value_t(1), blas::mat< value_t >( D ) );
                }// if
                else
                {
                    hlr::multiply( alpha, apply_normal, *A, apply_normal, *B, M );
                }// else
            }// for
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + M.typestr() );

        //
        // clean up update list
        //
        
        updates.clear();
    }
};

//
// compute C = C + α op( A ) op( B ) where A and B are provided as list of updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( ::tf::Subflow &          tf,
           const value_t            alpha,
           hpro::TMatrix &          C,
           lazy_accumulator &       accu,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split< value_t >( *BC );
        
        //
        // recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                tf.emplace(
                    [=,&sub_accu,&acc,&approx] ( ::tf::Subflow &  sf )
                    {
                        multiply< value_t >( sf, alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
                    } );
            }// for
        }// for
    }// if
    else
    {
        accu.apply_leaf( alpha, C, acc, approx );
    }// if
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      /* op_A */,
           const hpro::TMatrix &    A,
           const hpro::matop_t      /* op_B */,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    detail::lazy_accumulator  accu;

    accu.add_update< value_t >( &A, &B );
    
    ::tf::Taskflow  tf;
    
    tf.emplace( [=,&A,&B,&C,&acc,&approx,&accu] ( ::tf::Subflow &  sf ) { detail::multiply< value_t >( sf, alpha, C, accu, acc, approx ); } );

    ::tf::Executor  executor;
    
    executor.run( tf ).wait();
}

}}}// namespace hlr::tf::lazy

#endif // __HLR_TF_ARITH_LAZY_HH
