#ifndef __HLR_TBB_ARITH_LAZY_HH
#define __HLR_TBB_ARITH_LAZY_HH
//
// Project     : HLR
// Module      : tbb/arith_lazy.hh
// Description : arithmetic functions for lazy evaluation using TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/matrix/convert.hh"
#include "hlr/matrix/product.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/io.hh"

namespace hlr { namespace tbb { namespace lazy {

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

    const std::list< const Hpro::TLinearOperator< value_t > * > &  ops;

    linopsum_operator ( const std::list< const Hpro::TLinearOperator< value_t > * > &  aops )
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
operator_wrapper ( const std::list< const Hpro::TLinearOperator< value_t > * > &  M )
{
    return linopsum_operator< value_t > ( M );
}

//
// accumulator for lazy updates
//
template < typename value_t >
struct lazy_accumulator
{
    using  update_t      = std::pair< const Hpro::TMatrix< value_t > *, const Hpro::TMatrix< value_t > * >;
    using  update_list_t = std::list< update_t >;
    using  create_list_t = std::list< std::unique_ptr< Hpro::TMatrix< value_t > > >;

    // list of collected updates
    update_list_t  updates;

    // list of created wrapper matrices
    create_list_t  created;

    // signals dense handling of updates
    bool           handle_dense = false;

    //
    // add A×B to updates
    // - α is applied _only_ if possible
    //
    void
    add_update ( const Hpro::TMatrix< value_t > *  A,
                 const Hpro::TMatrix< value_t > *  B )
    {
        HLR_ASSERT( ! is_null( A ) );
        
        if ( matrix::is_lowrank_all( A, B ) )
        {
            //
            // create U·T·V' representation with T=V(A)'·U(B)
            //

            auto  RA = cptrcast( A, matrix::lrmatrix< value_t > );
            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );

            auto  U  = RA->U();
            auto  T  = blas::prod( blas::adjoint( RA->V() ), RB->U() );
            auto  V  = RB->V();
            
            auto  AxB = std::make_unique< matrix::lrsmatrix< value_t > >( A->row_is(), B->col_is(), std::move( U ), std::move( T ), std::move( V ) );
            
            updates.push_back( { AxB.get(), nullptr } );
            created.push_back( std::move( AxB ) );
        }// if
        else
        {
            if ( matrix::is_dense_all( A, B ) )
                handle_dense = true;
            
            updates.push_back( { A, B } );
        }// else
    }

    void
    add_update ( const Hpro::TMatrix< value_t > *  A )
    {
        add_update( A, nullptr );
    }
    
    //
    // split given set of updates to set of updates for sub blocks
    //
    tensor2< lazy_accumulator >
    split ( const Hpro::TBlockMatrix< value_t > &  M )
    {
        using  hlr::matrix::lrsmatrix;
        
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
                        auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null_all( BA->block( i, k ), BB->block( k, j ) ) );

                                sub_accu(i,j).add_update( BA->block( i, k ), BB->block( k, j ) );
                            }// for
                        }// if
                        else if ( matrix::is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );

                            HLR_ASSERT( ! RB->is_compressed() );
                            
                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  U_k  = blas::matrix< value_t >( RB->U(), col_k - B->row_ofs(), blas::range::all );
                                auto  V_j  = blas::matrix< value_t >( RB->V(), col_j - B->col_ofs(), blas::range::all );
                                auto  B_kj = std::make_unique< matrix::lrmatrix< value_t > >( col_k, col_j, U_k, V_j );
                                
                                sub_accu(i,j).add_update( A_ik, B_kj.get() );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else if ( matrix::is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, matrix::dense_matrix< value_t > );

                            HLR_ASSERT( ! DB->is_compressed() );
                            
                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  D_kj = blas::matrix< value_t >( DB->mat(), col_k - B->row_ofs(), col_j - B->col_ofs() );
                                auto  B_kj = std::make_unique< matrix::dense_matrix< value_t > >( col_k, col_j, D_kj );
                                
                                sub_accu(i,j).add_update( A_ik, B_kj.get() );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( matrix::is_lowrank( A ) )
                    {
                        auto  RA = cptrcast( A, matrix::lrmatrix< value_t > );
                        auto  AU = RA->U();
                        auto  AV = RA->V();

                        HLR_ASSERT( ! RA->is_compressed() );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  U_i  = blas::matrix< value_t >( RA->U(), row_i - A->row_ofs(), blas::range::all );
                                auto  V_k  = blas::matrix< value_t >( RA->V(), col_k - A->col_ofs(), blas::range::all );
                                auto  A_ik = std::make_unique< matrix::lrmatrix< value_t > >( row_i, col_k, U_i, V_k );
                                
                                sub_accu(i,j).add_update( A_ik.get(), B_kj );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( matrix::is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );
                            auto  BU = RB->U();
                            auto  BV = RB->V();

                            HLR_ASSERT( ! RB->is_compressed() );
                            
                            // restrict A to row_i
                            auto  U_i = blas::matrix< value_t >( AU, row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::lrmatrix< value_t > >( row_i, A->col_is(), U_i, AV );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix< value_t >( BV, col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< matrix::lrmatrix< value_t > >( B->row_is(), col_j, BU, V_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( matrix::is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, matrix::dense_matrix< value_t > );
                            auto  BD = DB->mat();

                            HLR_ASSERT( ! DB->is_compressed() );

                            // restrict A to row_i
                            auto  U_i = blas::matrix< value_t >( AU, row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::lrmatrix< value_t > >( row_i, A->col_is(), U_i, AV );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix< value_t >( BD, blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< matrix::dense_matrix< value_t > >( B->row_is(), col_j, D_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( matrix::is_lowrankS( A ) )
                    {
                        auto  RA = cptrcast( A, lrsmatrix< value_t > );

                        // A already holds full product
                        HLR_ASSERT( is_null( B ) );

                        // A already holds product of original A and B so just restrict to M_ij
                        auto  U_i  = blas::matrix< value_t >( RA->U(), row_i - A->row_ofs(), blas::range::all );
                        auto  V_j  = blas::matrix< value_t >( RA->V(), col_j - A->col_ofs(), blas::range::all );
                        auto  A_ij = std::make_unique< lrsmatrix< value_t > >( row_i, col_j, U_i, RA->S(), V_j );

                        sub_accu(i,j).add_update( A_ij.get() );
                        created.push_back( std::move( A_ij ) );
                    }// if
                    else if ( matrix::is_dense( A ) )
                    {
                        auto  DA = cptrcast( A, matrix::dense_matrix< value_t > );
                        auto  AD = DA->mat();

                        HLR_ASSERT( ! DA->is_compressed() );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  D_ik = blas::matrix< value_t >( AD, row_i - A->row_ofs(), col_k - A->col_ofs() );
                                auto  A_ik = std::make_unique< matrix::dense_matrix< value_t > >( row_i, col_k, D_ik );
                                
                                sub_accu(i,j).add_update( A_ik.get(), B_kj );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( matrix::is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );
                            auto  BU = RB->U();
                            auto  BV = RB->V();
                            
                            HLR_ASSERT( ! RB->is_compressed() );
                            
                            // restrict A to row_i
                            auto  D_i = blas::matrix< value_t >( AD, row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::dense_matrix< value_t > >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix< value_t >( BV, col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< matrix::lrmatrix< value_t > >( B->row_is(), col_j, BU, V_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get() );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( matrix::is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, matrix::dense_matrix< value_t > );
                            auto  BD = DB->mat();

                            HLR_ASSERT( ! DB->is_compressed() );

                            // restrict A to row_i
                            auto  D_i = blas::matrix< value_t >( AD, row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::dense_matrix< value_t > >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix< value_t >( BD, blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< matrix::dense_matrix< value_t > >( B->row_is(), col_j, D_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get() );

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
    template < typename approx_t >
    void
    apply_leaf ( const value_t               alpha,
                 Hpro::TMatrix< value_t > &  M,
                 const accuracy &            acc,
                 const approx_t &            approx )
    {
        using  hlr::matrix::lrsmatrix;
        
        if ( matrix::is_lowrank( M ) )
        {
            auto  R = ptrcast( &M, matrix::lrmatrix< value_t > );

            //
            // an update resulted in a dense matrix, so handle all in dense format
            //
            
            if ( handle_dense )
            {
                auto  D = hlr::matrix::convert_to_dense< value_t >( *R );

                apply_leaf( alpha, *D, acc, approx );

                auto  DD       = D->mat();
                auto  [ U, V ] = approx( DD, acc );
                    
                R->set_lrmat( std::move( U ), std::move( V ) );
            }// if
            else
            {
                //
                // otherwise handle updates in low-rank format
                //
            
                if constexpr( approx_t::supports_general_operator )
                {
                    //
                    // set up operator for sum of matrix products plus C
                    //
        
                    auto  op_list = std::list< const Hpro::TLinearOperator< value_t > * >();
                    auto  deleted = std::list< const Hpro::TLinearOperator< value_t > * >();

                    for ( auto  [ A, B ] : updates )
                    {
                        if ( matrix::is_lowrankS( A ) )
                        {
                            HLR_ASSERT( is_null( B ) );
                        
                            auto  op_AxB = matrix::product( alpha, *A );
                        
                            deleted.push_back( op_AxB.get() );
                            op_list.push_back( op_AxB.release() );
                        }// if
                        else
                        {
                            auto  op_AxB = matrix::product( alpha, *A, 1, *B );
                        
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
                        if ( matrix::is_lowrankS( A ) )
                        {
                            HLR_ASSERT( is_null( B ) );
                    
                            auto  RA = cptrcast( A, matrix::lrsmatrix< value_t > );
                            auto  US = blas::prod( alpha, RA->U(), RA->S() );

                            auto  [ U, V ] = approx( { R->U(), US },
                                                     { R->V(), RA->V() },
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
        else if ( matrix::is_dense( M ) )
        {
            auto  D = ptrcast( &M, matrix::dense_matrix< value_t > );

            for ( auto  [ A, B ] : updates )
            {
                if ( matrix::is_lowrankS( A ) )
                {
                    HLR_ASSERT( is_null( B ) );
                    
                    auto  RA = cptrcast( A, lrsmatrix< value_t > );
                    auto  US = blas::prod( RA->U(), RA->S() );
                    auto  DD = D->mat();

                    blas::prod( alpha, US, blas::adjoint( RA->V() ), value_t(1), DD );
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
multiply ( const value_t                  alpha,
           Hpro::TMatrix< value_t > &     C,
           lazy_accumulator< value_t > &  accu,
           const accuracy &               acc,
           const approx_t &               approx )
{
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast( &C, Hpro::TBlockMatrix< value_t > );

        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BC );
        
        //
        // recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BC->nblock_rows(),
                                            0, BC->nblock_cols() ),
            [&,BC,alpha] ( const auto & r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                    for ( uint  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                        multiply< value_t >( alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
                    }// for
            } );
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
multiply ( const value_t                       alpha,
           const Hpro::matop_t                 /* op_A */,
           const Hpro::TMatrix< value_t > &    A,
           const Hpro::matop_t                 /* op_B */,
           const Hpro::TMatrix< value_t > &    B,
           Hpro::TMatrix< value_t > &          C,
           const accuracy &                    acc,
           const approx_t &                    approx )
{
    detail::lazy_accumulator< value_t >  accu;

    accu.add_update( &A, &B );
    
    detail::multiply( alpha, C, accu, acc, approx );
}

////////////////////////////////////////////////////////////////////////////////
//
// lazy LU factorization
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
                  lazy_accumulator< value_t > &     accu,
                  const accuracy &                  acc,
                  const approx_t &                  approx )
{
    if ( hlr::is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BM );

        //
        // recurse
        //
        
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
                        sub_accu(k,j).add_update( BL->block(k,i), BM->block(i,j) );
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else
    {
        accu.apply_leaf( value_t(-1), M, acc, approx );
        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t                 side,
                  const diag_type_t                 diag,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TMatrix< value_t > &        M,
                  lazy_accumulator< value_t > &     accu,
                  const accuracy &                  acc,
                  const approx_t &                  approx )
{
    if ( hlr::is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BM );

        //
        // recurse
        //
        
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
                        sub_accu(i,k).add_update( BM->block(i,j), BU->block(j,k) );
            }// for
        }// else
    }// if
    else
    {
        accu.apply_leaf( value_t(-1), M, acc, approx );
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// else
}

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &     M,
     lazy_accumulator< value_t > &  accu,
     const accuracy &               acc,
     const approx_t &               approx )
{
    //
    // (recursive) LU factorization
    //
    
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BM );

        //
        // recursive factorization
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
                    sub_accu(j,l).add_update( BM->block( j, i ), BM->block( i, l ) );
        }// for
    }// if
    else
    {
        if ( matrix::is_dense( M ) )
        {
            accu.apply_leaf( value_t(-1), M, acc, approx );
        
            auto  D  = ptrcast( &M, matrix::dense_matrix< value_t > );
            auto  DD = D->mat();

            blas::invert( DD );
        
            if ( D->is_compressed() )
                D->set_matrix( std::move( DD ), acc );
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
     const accuracy &            acc,
     const approx_t &            approx )
{
    detail::lazy_accumulator< value_t >  accu;
    
    detail::lu< value_t >( M, accu, acc, approx );
}

}}}// namespace hlr::tbb::lazy

#endif // __HLR_TBB_ARITH_LAZY_HH
