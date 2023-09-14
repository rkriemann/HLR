#ifndef __HLR_SEQ_ARITH_LAZY_HH
#define __HLR_SEQ_ARITH_LAZY_HH
//
// Project     : HLR
// Module      : seq/arith_lazy.hh
// Description : sequential arithmetic functions using lazy evaluation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "hlr/arith/multiply.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/log.hh"
#include "hlr/utils/io.hh"

namespace hlr { namespace seq { namespace lazy {

// define to enable some statistics output
#define HLR_ARITH_LAZY_STAT( msg ) // std::cout << msg << std::endl

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
    using  accumulator_t = std::unique_ptr< Hpro::TMatrix< value_t > >;

    // list of collected updates
    update_list_t  updates;

    // list of created wrapper matrices
    create_list_t  created;

    // optional matrix for computed updates
    accumulator_t  accumulator;

    // signals dense handling of updates
    bool           handle_dense = false;

    //
    // add A×B to updates
    //
    template < typename approx_t >
    void
    add_update ( const Hpro::TMatrix< value_t > *    A,
                 const Hpro::TMatrix< value_t > *    B,
                 const Hpro::TTruncAcc &  acc,
                 const approx_t           approx )
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

        if ( false && ( updates.size() > 100 ))
        {
            if ( is_null( accumulator ) )
            {
                const indexset  rowis = A->row_is();
                const indexset  colis = ( is_null( B ) ? A->col_is() : B->col_is() );
                
                if ( handle_dense )
                    accumulator = std::make_unique< matrix::dense_matrix< value_t > >( rowis, colis );
                else
                    accumulator = std::make_unique< matrix::lrmatrix< value_t > >( rowis, colis );
            }// if

            apply_leaf( value_t(1), *accumulator, acc, approx );
        }// if
    }

    template < typename approx_t >
    void
    add_update ( const Hpro::TMatrix< value_t > *  A,
                 const Hpro::TTruncAcc &           acc,
                 const approx_t                    approx )
    {
        add_update( A, nullptr, acc, approx );
    }
    
    //
    // split given set of updates to set of updates for sub blocks
    //
    template < typename approx_t >
    tensor2< lazy_accumulator >
    split ( const Hpro::TBlockMatrix< value_t > &  M,
            const Hpro::TTruncAcc &     acc,
            const approx_t              approx )
    {
        auto  sub_accu = tensor2< lazy_accumulator >( M.nblock_rows(), M.nblock_cols() );

        if ( ! is_null( accumulator ) )
        {
            for ( uint  i = 0; i < M.nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < M.nblock_cols(); ++j )
                {
                    sub_accu(i,j).accumulator = std::move( hlr::matrix::restrict( *accumulator, M.block( i, j )->block_is() ) );
                }// for
            }// for
        }// if
        
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

                                sub_accu(i,j).add_update( BA->block( i, k ), BB->block( k, j ), acc, approx );
                            }// for
                        }// if
                        else if ( matrix::is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  U_k = blas::matrix< value_t >( RB->U(), col_k - B->row_ofs(), blas::range::all );
                                auto  V_j = blas::matrix< value_t >( RB->V(), col_j - B->col_ofs(), blas::range::all );
                                auto  B_kj = std::make_unique< matrix::lrmatrix< value_t > >( col_k, col_j, U_k, V_j );
                                
                                sub_accu(i,j).add_update( A_ik, B_kj.get(), acc, approx );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else if ( matrix::is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, matrix::dense_matrix< value_t > );

                            for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BA->block( i, k ) ) );

                                auto  A_ik  = BA->block( i, k );
                                auto  col_k = A_ik->col_is();

                                // restrict B to col_k × col_j
                                auto  D_kj = blas::matrix< value_t >( DB->mat(), col_k - B->row_ofs(), col_j - B->col_ofs() );
                                auto  B_kj = std::make_unique< matrix::dense_matrix< value_t > >( col_k, col_j, D_kj );
                                
                                sub_accu(i,j).add_update( A_ik, B_kj.get(), acc, approx );

                                created.push_back( std::move( B_kj ) );
                            }// for
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( matrix::is_lowrank( A ) )
                    {
                        auto  RA = cptrcast( A, matrix::lrmatrix< value_t > );
                        
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
                                
                                sub_accu(i,j).add_update( A_ik.get(), B_kj, acc, approx );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( matrix::is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );

                            // restrict A to row_i
                            auto  U_i = blas::matrix< value_t >( RA->U(), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::lrmatrix< value_t > >( row_i, A->col_is(), U_i, RA->V() );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix< value_t >( RB->V(), col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< matrix::lrmatrix< value_t > >( B->row_is(), col_j, RB->U(), V_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get(), acc, approx );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( matrix::is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, matrix::dense_matrix< value_t > );

                            // restrict A to row_i
                            auto  U_i = blas::matrix< value_t >( RA->U(), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::lrmatrix< value_t > >( row_i, A->col_is(), U_i, RA->V() );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix< value_t >( DB->mat(), blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< matrix::dense_matrix< value_t > >( B->row_is(), col_j, D_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get(), acc, approx );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type: " + B->typestr() );
                    }// if
                    else if ( matrix::is_lowrankS( A ) )
                    {
                        auto  RA = cptrcast( A, matrix::lrsmatrix< value_t > );

                        // A already holds full product
                        HLR_ASSERT( is_null( B ) );

                        // A already holds product of original A and B so just restrict to M_ij
                        auto  U_i  = blas::matrix< value_t >( RA->U(), row_i - A->row_ofs(), blas::range::all );
                        auto  V_j  = blas::matrix< value_t >( RA->V(), col_j - A->col_ofs(), blas::range::all );
                        auto  A_ij = std::make_unique< matrix::lrsmatrix< value_t > >( row_i, col_j, U_i, RA->S(), V_j );

                        sub_accu(i,j).add_update( A_ij.get(), acc, approx );
                        created.push_back( std::move( A_ij ) );
                    }// if
                    else if ( matrix::is_dense( A ) )
                    {
                        auto  DA = cptrcast( A, matrix::dense_matrix< value_t > );
                        
                        if ( is_blocked( B ) )
                        {
                            auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );

                            for ( uint  k = 0; k < BB->nblock_rows(); ++k )
                            {
                                HLR_ASSERT( ! is_null( BB->block( k, j ) ) );

                                auto  B_kj  = BB->block( k, j );
                                auto  col_k = B_kj->row_is();
                                
                                // restrict A to row_i × col_k
                                auto  D_ik = blas::matrix< value_t >( DA->mat(), row_i - A->row_ofs(), col_k - A->col_ofs() );
                                auto  A_ik = std::make_unique< matrix::dense_matrix< value_t > >( row_i, col_k, D_ik );
                                
                                sub_accu(i,j).add_update( A_ik.get(), B_kj, acc, approx );

                                created.push_back( std::move( A_ik ) );
                            }// for
                        }// if
                        else if ( matrix::is_lowrank( B ) )
                        {
                            auto  RB = cptrcast( B, matrix::lrmatrix< value_t > );
                            
                            // restrict A to row_i
                            auto  D_i = blas::matrix< value_t >( DA->mat(), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::dense_matrix< value_t > >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  V_j = blas::matrix< value_t >( RB->V(), col_j - B->col_ofs(), blas::range::all );
                            auto  B_j = std::make_unique< matrix::lrmatrix< value_t > >( B->row_is(), col_j, RB->U(), V_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get(), acc, approx );

                            created.push_back( std::move( A_i ) );
                            created.push_back( std::move( B_j ) );
                        }// if
                        else if ( matrix::is_dense( B ) )
                        {
                            auto  DB = cptrcast( B, matrix::dense_matrix< value_t > );

                            // restrict A to row_i
                            auto  D_i = blas::matrix< value_t >( DA->mat(), row_i - A->row_ofs(), blas::range::all );
                            auto  A_i = std::make_unique< matrix::dense_matrix< value_t > >( row_i, A->col_is(), D_i );
                            
                            // restrict B to col_j
                            auto  D_j = blas::matrix< value_t >( DB->mat(), blas::range::all, col_j - B->col_ofs() );
                            auto  B_j = std::make_unique< matrix::dense_matrix< value_t > >( B->row_is(), col_j, D_j );
                                
                            sub_accu(i,j).add_update( A_i.get(), B_j.get(), acc, approx );

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
                 const Hpro::TTruncAcc &     acc,
                 const approx_t &            approx )
    {
        const bool  apply_to_accu = ( & M == accumulator.get() );
        
        if ( matrix::is_lowrank( M ) )
        {
            auto  R = ptrcast( &M, matrix::lrmatrix< value_t > );

            //
            // an update resulted in a dense matrix, so handle all in dense format
            //
            
            if ( handle_dense )
            {
                auto  D = matrix::convert_to_dense< value_t >( *R );

                if ( apply_to_accu )
                {
                    accumulator = std::move( D );

                    apply_leaf( alpha, *accumulator, acc, approx );
                }// if
                else
                {    
                    apply_leaf( alpha, *D, acc, approx );

                    auto  [ U, V ] = approx( D->mat(), acc );
                    
                    R->set_lrmat( std::move( U ), std::move( V ) );
                }// 
            }// if
            else
            {
                //
                // otherwise handle updates in low-rank format
                //
            
                HLR_ARITH_LAZY_STAT( "#updates lowrank " << std::min( M.nrows(), M.ncols() ) << " " << updates.size() );
        
                if constexpr( approx_t::supports_general_operator )
                {
                    //
                    // set up operator for sum of matrix products plus C
                    //
        
                    auto  op_list = std::list< const Hpro::TLinearOperator< value_t > * >();
                    auto  deleted = std::list< const Hpro::TLinearOperator< value_t > * >();

                    if ( ! apply_to_accu && ! is_null( accumulator ) )
                    {
                        auto  op_AxB = Hpro::matrix_product( alpha, accumulator.get() );
                        
                        deleted.push_back( op_AxB.get() );
                        op_list.push_back( op_AxB.release() );
                    }// if
                
                    for ( auto  [ A, B ] : updates )
                    {
                        if ( matrix::is_lowrankS( A ) )
                        {
                            HLR_ASSERT( is_null( B ) );
                        
                            auto  op_AxB = Hpro::matrix_product( alpha, A );
                        
                            deleted.push_back( op_AxB.get() );
                            op_list.push_back( op_AxB.release() );
                        }// if
                        else
                        {
                            auto  op_AxB = Hpro::matrix_product( alpha, A, value_t(1), B );
                        
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
                    // add already computed updates
                    //
                
                    if ( ! apply_to_accu && ! is_null( accumulator ) )
                    {
                        hlr::add( alpha, *accumulator, M, acc, approx );
                    }// if
                
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
            HLR_ARITH_LAZY_STAT( "#updates dense " << std::min( M.nrows(), M.ncols() ) << " " << updates.size() );
        
            auto  D = ptrcast( &M, matrix::dense_matrix< value_t > );

            // add already computed updates
            if ( ! apply_to_accu && ! is_null( accumulator ) )
            {
                hlr::add( alpha, *accumulator, M );
            }// if
            
            for ( auto  [ A, B ] : updates )
            {
                if ( matrix::is_lowrankS( A ) )
                {
                    HLR_ASSERT( is_null( B ) );
                    
                    auto  RA = cptrcast( A, matrix::lrsmatrix< value_t > );
                    auto  US = blas::prod( RA->U(), RA->S() );

                    blas::prod( alpha, US, blas::adjoint( RA->V() ), value_t(1), D->mat() );
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
        // clean up update list and accumulator
        //
        
        updates.clear();

        if ( ! apply_to_accu )
            accumulator.reset( nullptr );
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
           const Hpro::TTruncAcc &        acc,
           const approx_t &               approx )
{
    if ( is_blocked( C ) )
    {
        auto  BC = ptrcast(  &C, Hpro::TBlockMatrix< value_t > );

        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BC, acc, approx );
        
        //
        // recurse
        //
        
        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                HLR_ASSERT( ! is_null( BC->block( i, j ) ) );
                
                multiply< value_t >( alpha, *BC->block(i,j), sub_accu(i,j), acc, approx );
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
           const Hpro::matop_t      /* op_A */,
           const Hpro::TMatrix< value_t > &    A,
           const Hpro::matop_t      /* op_B */,
           const Hpro::TMatrix< value_t > &    B,
           Hpro::TMatrix< value_t > &          C,
           const Hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    detail::lazy_accumulator< value_t >  accu;

    accu.add_update( &A, &B, acc, approx );
    
    detail::multiply< value_t >( alpha, C, accu, acc, approx );
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
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    if ( is_blocked_all( L, M ) )
    {
        auto  BL = cptrcast( &L, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BM, acc, approx );

        //
        // recurse
        //
        
        if ( side == from_left )
        {
            for ( uint i = 0; i < BM->nblock_rows(); ++i )
            {
                const auto  L_ii = BL->block( i, i );

                HLR_ASSERT( ! is_null( L_ii ) );
                
                for ( uint j = 0; j < BM->nblock_cols(); ++j )
                    if ( ! is_null( BM->block( i, j ) ) )
                        solve_lower_tri< value_t >( side, diag, *L_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );

                for ( uint  k = i+1; k < BM->nblock_rows(); ++k )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        if ( ! is_null_any( BL->block( k, i ), BM->block( i, j ) ) )
                            sub_accu(k,j).add_update( BL->block( k, i ), BM->block( i, j ), acc, approx );
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
                  const Hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    if ( is_blocked_all( U, M ) )
    {
        auto  BU = cptrcast( &U, Hpro::TBlockMatrix< value_t > );
        auto  BM =  ptrcast( &M, Hpro::TBlockMatrix< value_t > );
        
        //
        // restrict set of updates for all subblocks
        //

        auto  sub_accu = accu.split( *BM, acc, approx );

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
            
                HLR_ASSERT( ! is_null( U_jj ) );
                
                for ( uint i = 0; i < BM->nblock_rows(); ++i )
                    if ( ! is_null( BM->block( i, j ) ) )
                        solve_upper_tri< value_t >( side, diag, *U_jj, *BM->block( i, j ), sub_accu(i,j), acc, approx );
            
                for ( uint  k = j+1; k < BM->nblock_cols(); ++k )
                    for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                        if ( ! is_null_any( BM->block( i, j ), BU->block( j, k ) ) )
                            sub_accu(i,k).add_update( BM->block(i,j), BU->block(j,k), acc, approx );
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
     const Hpro::TTruncAcc &        acc,
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

        auto  sub_accu = accu.split( *BM, acc, approx );

        //
        // recursive factorization
        //
        
        for ( uint  i = 0; i < std::min( BM->nblock_rows(), BM->nblock_cols() ); ++i )
        {
            auto  B_ii = BM->block( i, i );

            HLR_ASSERT( ! is_null( B_ii ) );
            
            lu< value_t >( *B_ii, sub_accu(i,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                if ( ! is_null( BM->block( j, i ) ) )
                    solve_upper_tri< value_t >( from_right, general_diag, *B_ii, *BM->block( j, i ), sub_accu(j,i), acc, approx );

            for ( uint  j = i+1; j < BM->nblock_cols(); ++j )
                if ( ! is_null( BM->block( i, j ) ) )
                    solve_lower_tri< value_t >( from_left, unit_diag, *B_ii, *BM->block( i, j ), sub_accu(i,j), acc, approx );

            // add updates to sub lists
            for ( uint  j = i+1; j < BM->nblock_rows(); ++j )
                for ( uint  l = i+1; l < BM->nblock_cols(); ++l )
                    if ( ! is_null_any( BM->block( j, i ), BM->block( i, l ) ) )
                        sub_accu(j,l).add_update( BM->block( j, i ), BM->block( i, l ), acc, approx );
        }// for
    }// if
    else if ( matrix::is_dense( M ) )
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
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &          M,
     const Hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    detail::lazy_accumulator< value_t >  accu;
    
    detail::lu< value_t >( M, accu, acc, approx );
}

}}}// namespace hlr::seq::lazy

#endif // __HLR_SEQ_ARITH_LAZY_HH
