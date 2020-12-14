#ifndef __HLR_SEQ_ARITH_LAZY_HH
#define __HLR_SEQ_ARITH_LAZY_HH
//
// Project     : HLib
// Module      : seq/arith_lazy.hh
// Description : sequential arithmetic functions using lazy evaluation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>
#include <hpro/matrix/TDenseMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/arith/multiply.hh"

namespace hlr { namespace seq { namespace lazy {

namespace hpro = HLIB;

////////////////////////////////////////////////////////////////////////////////
//
// matrix multiplication
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

//
// auxiliary types
//

using  update_t      = std::pair< const hpro::TMatrix *, const hpro::TMatrix * >;
using  update_list_t = std::list< update_t >;

//
// operator wrapper for sum of linear operators
//

template < typename T_value >
struct linopsum_operator
{
    using  value_t = T_value;

    const std::list< hpro::TLinearOperator * > &  ops;

    linopsum_operator ( const std::list< hpro::TLinearOperator * > &  aops )
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
operator_wrapper ( const std::list< hpro::TLinearOperator * > &  M )
{
    return linopsum_operator< value_t > ( M );
}

//
// split given set of updates to set of updates for sub blocks
//
template < typename value_t >
std::pair< tensor2< update_list_t >,
           std::list< std::unique_ptr< hpro::TMatrix > > >
split_updates( const hpro::TBlockMatrix &  M,
               const update_list_t &       updates )
{
    auto  sub_updates = tensor2< update_list_t >( M.nblock_rows(), M.nblock_cols() );
    auto  created     = std::list< std::unique_ptr< hpro::TMatrix > >();

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

                            sub_updates(i,j).push_back( std::make_pair( BA->block( i, k ), BB->block( k, j ) ) );
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
                            auto  U_k = blas::matrix( blas::mat_U< value_t >( RB ), col_k - B->row_ofs(), blas::range::all );
                            auto  V_j = blas::matrix( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                            auto  B_kj = std::make_unique< hpro::TRkMatrix >( col_k, col_j, U_k, V_j );
                                
                            sub_updates(i,j).push_back( std::make_pair( A_ik, B_kj.get() ) );

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
                            auto  D_kj = blas::matrix( blas::mat< value_t >( DB ), col_k - B->row_ofs(), col_j - B->col_ofs() );
                            auto  B_kj = std::make_unique< hpro::TDenseMatrix >( col_k, col_j, D_kj );
                                
                            sub_updates(i,j).push_back( std::make_pair( A_ik, B_kj.get() ) );

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
                            auto  U_i  = blas::matrix( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                            auto  V_k  = blas::matrix( blas::mat_V< value_t >( RA ), col_k - A->col_ofs(), blas::range::all );
                            auto  A_ik = std::make_unique< hpro::TRkMatrix >( row_i, col_k, U_i, V_k );
                                
                            sub_updates(i,j).push_back( std::make_pair( A_ik.get(), B_kj ) );

                            created.push_back( std::move( A_ik ) );
                        }// for
                    }// if
                    else if ( is_lowrank( B ) )
                    {
                        auto  RB = cptrcast( B, hpro::TRkMatrix );

                        // restrict A to row_i
                        auto  U_i = blas::matrix( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                        auto  A_i = std::make_unique< hpro::TRkMatrix >( row_i, A->col_is(), U_i, blas::mat_V< value_t >( RA ) );
                            
                        // restrict B to col_j
                        auto  V_j = blas::matrix( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                        auto  B_j = std::make_unique< hpro::TRkMatrix >( B->row_is(), col_j, blas::mat_U< value_t >( RB ), V_j );
                                
                        sub_updates(i,j).push_back( std::make_pair( A_i.get(), B_j.get() ) );

                        created.push_back( std::move( A_i ) );
                        created.push_back( std::move( B_j ) );
                    }// if
                    else if ( is_dense( B ) )
                    {
                        auto  DB = cptrcast( B, hpro::TDenseMatrix );

                        // restrict A to row_i
                        auto  U_i = blas::matrix( blas::mat_U< value_t >( RA ), row_i - A->row_ofs(), blas::range::all );
                        auto  A_i = std::make_unique< hpro::TRkMatrix >( row_i, A->col_is(), U_i, blas::mat_V< value_t >( RA ) );
                            
                        // restrict B to col_j
                        auto  D_j = blas::matrix( blas::mat< value_t >( DB ), blas::range::all, col_j - B->col_ofs() );
                        auto  B_j = std::make_unique< hpro::TDenseMatrix >( B->row_is(), col_j, D_j );
                                
                        sub_updates(i,j).push_back( std::make_pair( A_i.get(), B_j.get() ) );

                        created.push_back( std::move( A_i ) );
                        created.push_back( std::move( B_j ) );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type: " + B->typestr() );
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
                            auto  D_ik = blas::matrix( blas::mat< value_t >( DA ), row_i - A->row_ofs(), col_k - A->col_ofs() );
                            auto  A_ik = std::make_unique< hpro::TDenseMatrix >( row_i, col_k, D_ik );
                                
                            sub_updates(i,j).push_back( std::make_pair( A_ik.get(), B_kj ) );

                            created.push_back( std::move( A_ik ) );
                        }// for
                    }// if
                    else if ( is_lowrank( B ) )
                    {
                        auto  RB = cptrcast( B, hpro::TRkMatrix );
                            
                        // restrict A to row_i
                        auto  D_i = blas::matrix( blas::mat< value_t >( DA ), row_i - A->row_ofs(), blas::range::all );
                        auto  A_i = std::make_unique< hpro::TDenseMatrix >( row_i, A->col_is(), D_i );
                            
                        // restrict B to col_j
                        auto  V_j = blas::matrix( blas::mat_V< value_t >( RB ), col_j - B->col_ofs(), blas::range::all );
                        auto  B_j = std::make_unique< hpro::TRkMatrix >( B->row_is(), col_j, blas::mat_U< value_t >( RB ), V_j );
                                
                        sub_updates(i,j).push_back( std::make_pair( A_i.get(), B_j.get() ) );

                        created.push_back( std::move( A_i ) );
                        created.push_back( std::move( B_j ) );
                    }// if
                    else if ( is_dense( B ) )
                    {
                        auto  DB = cptrcast( B, hpro::TDenseMatrix );

                        // restrict A to row_i
                        auto  D_i = blas::matrix( blas::mat< value_t >( DA ), row_i - A->row_ofs(), blas::range::all );
                        auto  A_i = std::make_unique< hpro::TDenseMatrix >( row_i, A->col_is(), D_i );
                            
                        // restrict B to col_j
                        auto  D_j = blas::matrix( blas::mat< value_t >( DB ), blas::range::all, col_j - B->col_ofs() );
                        auto  B_j = std::make_unique< hpro::TDenseMatrix >( B->row_is(), col_j, D_j );
                                
                        sub_updates(i,j).push_back( std::make_pair( A_i.get(), B_j.get() ) );

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
    
    return { std::move( sub_updates ), std::move( created ) };
}

//
// apply updates to lowrank matrix
//
template < typename value_t,
           typename approx_t >
void
apply_to_lowrank ( const value_t            alpha,
                   hpro::TRkMatrix &        M,
                   const update_list_t &    updates,
                   const hpro::TTruncAcc &  acc,
                   const approx_t &         approx )
{
    //
    // set up operator for sum of matrix products plus C
    //
        
    auto  op_list = std::list< hpro::TLinearOperator * >();

    for ( auto  [ A, B ] : updates )
    {
        auto  op_AxB = hpro::matrix_product( alpha, A, value_t(1), B );

        op_list.push_back( op_AxB.release() );
    }// for

    op_list.push_back( &M );

    auto  sumop = operator_wrapper< value_t >( op_list );

    // std::cout << C.id() << " : " << op_list.size() << std::endl;
        
    //
    // apply lowrank approximation and update C
    //
        
    auto  [ U, V ] = approx( sumop, acc );

    M.set_lrmat( std::move( U ), std::move( V ) );
    
    //
    // clean up
    //
        
    for ( auto  op : op_list )
    {
        if ( op != &M )
            delete op;
    }// for
}

//
// compute C = C + α op( A ) op( B ) where A and B are provided as list of updates
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           hpro::TMatrix &          C,
           update_list_t &          updates,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, hpro::TBlockMatrix );

        //
        // restrict set of updates for all subblocks
        //

        auto  [ sub_updates, created ] = split_updates< value_t >( *BC, updates );
        
        //
        // recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                multiply< value_t >( alpha, *BC->block(i,j), sub_updates(i,j), acc, approx );
            }// for
        }// for
    }// if
    else if ( is_lowrank( C ) )
    {
        auto  R = ptrcast( &C, hpro::TRkMatrix );
        
        apply_to_lowrank( alpha, *R, updates, acc, approx );
    }// if
    else if ( is_dense( C ) )
    {
        // std::cout << C.id() << std::endl;

        // evaluate all updates and apply to C
        for ( auto  [ A, B ] : updates )
            hlr::multiply( alpha, apply_normal, *A, apply_normal, *B, C, acc, approx );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + C.typestr() );
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
    detail::update_list_t  updates{ std::make_pair( &A, &B ) };
    
    detail::multiply< value_t >( alpha, C, updates, acc, approx );
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
solve_lower_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    L,
                  hpro::TMatrix &          M,
                  update_list_t &          updates,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        
        //
        // restrict set of updates for all subblocks
        //

        auto  [ sub_updates, created ] = split_updates< value_t >( *BM, updates );

        //
        // recurse
        //
        
        if ( side == from_left )
        {
            for ( uint i = 0; i < BM->nblock_rows(); ++i )
            {
                const auto  L_ii = BL->block( i, i );
            
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block(i,j), sub_updates(i,j), acc, approx );

                for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        sub_updates(k,j).push_back( { BL->block(k,i), BM->block(i,j) } );
            }// for
        }// if
        else
        {
            HLR_ASSERT( false );
        }// else
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = ptrcast( &M, hpro::TRkMatrix );
        
        apply_to_lowrank( value_t(-1), *R, updates, acc, approx );
        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// if
    else if ( is_dense( M ) )
    {
        for ( auto  [ A, B ] : updates )
            hlr::multiply( value_t(-1), apply_normal, *A, apply_normal, *B, M, acc, approx );
        
        hlr::solve_lower_tri< value_t >( side, diag, L, M, acc, approx );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

template < typename value_t,
           typename approx_t >
void
solve_upper_tri ( const eval_side_t        side,
                  const diag_type_t        diag,
                  const hpro::TMatrix &    U,
                  hpro::TMatrix &          M,
                  update_list_t &          updates,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, hpro::TBlockMatrix );
        auto  BM =  ptrcast( &M, hpro::TBlockMatrix );
        
        //
        // restrict set of updates for all subblocks
        //

        auto  [ sub_updates, created ] = split_updates< value_t >( *BM, updates );

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
            
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                    solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ), sub_updates(i,j), acc, approx );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        sub_updates(i,k).push_back( { BM->block(i,j), BU->block(j,k) } );
            }// for
        }// else
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = ptrcast( &M, hpro::TRkMatrix );
        
        apply_to_lowrank( value_t(-1), *R, updates, acc, approx );
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// if
    else if ( is_dense( M ) )
    {
        for ( auto  [ A, B ] : updates )
            hlr::multiply( value_t(-1), apply_normal, *A, apply_normal, *B, M, acc, approx );
        
        hlr::solve_upper_tri< value_t >( side, diag, U, M, acc, approx );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          M,
     update_list_t &          updates,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    //
    // (recursive) LU factorization
    //
    
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( &M, hpro::TBlockMatrix );

        //
        // restrict set of updates for all subblocks
        //

        auto  [ sub_updates, created ] = split_updates< value_t >( *BM, updates );

        //
        // recursive factorization
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            lu< value_t >( *B_ii, sub_updates(i,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_updates(j,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_cols(); ++j )
                solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_updates(i,j), acc, approx );

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    sub_updates(j,l).push_back( { BM->block( j, i ), BM->block( i, l ) } );
        }// for
    }// if
    else
    {
        if ( is_dense( M ) )
        {
            for ( auto  [ A, B ] : updates )
                hlr::multiply( value_t(-1), apply_normal, *A, apply_normal, *B, M, acc, approx );
        
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
    detail::update_list_t  updates;
    
    detail::lu< value_t >( M, updates, acc, approx );
}

}}}// namespace hlr::seq::lazy

#endif // __HLR_SEQ_ARITH_LAZY_HH
