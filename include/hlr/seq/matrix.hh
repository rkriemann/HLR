#ifndef __HLR_SEQ_MATRIX_HH
#define __HLR_SEQ_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>
#include <type_traits>

#include <matrix/TMatrix.hh>
#include <matrix/TBlockMatrix.hh>
#include <matrix/structure.hh>
#include <base/TTruncAcc.hh>

#include "hlr/utils/checks.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"

namespace hlr { namespace seq { namespace matrix {

using namespace HLIB;
    
//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< TMatrix >
build ( const TBlockCluster *  bct,
        const coeff_t &        coeff,
        const lrapx_t &        lrapx,
        const TTruncAcc &      acc )
{
    static_assert( std::is_same< typename coeff_t::value_t,
                                 typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    assert( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< TMatrix >( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
        }// else
    }// if
    else
    {
        M = std::make_unique< TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), TBlockMatrix );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( bct->son( i, j ) != nullptr )
                {
                    auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
    }// else

    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// return copy of matrix
//
inline
std::unique_ptr< TMatrix >
copy ( const TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, TBlockMatrix );
        auto  N  = std::make_unique< TBlockMatrix >();
        auto  B  = ptrcast( N.get(), TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

template < typename value_t >
std::unique_ptr< TMatrix >
copy_tiled ( const TMatrix &  M,
             const size_t     ntile )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, TBlockMatrix );
        auto  N  = std::make_unique< TBlockMatrix >();
        auto  B  = ptrcast( N.get(), TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_tiled< value_t >( * BM->block( i, j ), ntile );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_lowrank( & M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, TRkMatrix );

        return std::make_unique< hlr::matrix::tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                           RM->col_is(),
                                                                           ntile,
                                                                           blas_mat_A< value_t >( RM ),
                                                                           blas_mat_B< value_t >( RM ) );
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of (block-wise) lower-left part of matrix
//
inline
std::unique_ptr< TMatrix >
copy_ll ( const TMatrix &    M,
          const diag_type_t  diag = general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, TBlockMatrix );
        auto  N  = std::make_unique< TBlockMatrix >();
        auto  B  = ptrcast( N.get(), TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = ( i == j ? copy_ll( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        auto  T = M.copy();

        if ( diag == unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                assert( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = BLAS::identity< HLIB::complex >( D->nrows() );
                else
                    D->blas_rmat() = BLAS::identity< HLIB::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// return copy of (block-wise) upper-right part of matrix
//
inline
std::unique_ptr< TMatrix >
copy_ur ( const TMatrix &    M,
          const diag_type_t  diag = general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, TBlockMatrix );
        auto  N  = std::make_unique< TBlockMatrix >();
        auto  B  = ptrcast( N.get(), TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = ( i == j ? copy_ur( * BM->block( i, j ), diag ) : copy( * BM->block( i, j ) ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else
    {
        // assuming non-structured block
        auto  T = M.copy();

        if ( diag == unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                assert( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = BLAS::identity< HLIB::complex >( D->nrows() );
                else
                    D->blas_rmat() = BLAS::identity< HLIB::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// copy data of A to matrix B
// - ASSUMPTION: identical matrix structure
//
inline
void
copy_to ( const TMatrix &  A,
          TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, TBlockMatrix );
        auto  BB = ptrcast(  &B, TBlockMatrix );

        assert( BA->nblock_rows() == BB->nblock_rows() );
        assert( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    assert( ! is_null( BB->block( i, j ) ) );

                    copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    assert( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// copy lower-left data of A to matrix B
// - ASSUMPTION: identical matrix structure in lower-left part
//
inline
void
copy_to_ll ( const TMatrix &  A,
             TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, TBlockMatrix );
        auto  BB = ptrcast(  &B, TBlockMatrix );

        assert( BA->nblock_rows() == BB->nblock_rows() );
        assert( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j <= i; ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    assert( ! is_null( BB->block( i, j ) ) );

                    if ( i == j )
                        copy_to_ll( * BA->block( i, j ), * BB->block( i, j ) );
                    else
                        copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    assert( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// copy upper-right data of A to matrix B
// - ASSUMPTION: identical matrix structure in upper-right part
//
inline
void
copy_to_ur ( const TMatrix &  A,
             TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, TBlockMatrix );
        auto  BB = ptrcast(  &B, TBlockMatrix );

        assert( BA->nblock_rows() == BB->nblock_rows() );
        assert( BA->nblock_cols() == BB->nblock_cols() );
        
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = i; j < BA->nblock_cols(); ++j )
            {
                if ( BA->block( i, j ) != nullptr )
                {
                    assert( ! is_null( BB->block( i, j ) ) );

                    if ( i == j )
                        copy_to_ur( * BA->block( i, j ), * BB->block( i, j ) );
                    else
                        copy_to( * BA->block( i, j ), * BB->block( i, j ) );
                }// if
                else
                    assert( is_null( BB->block( i, j ) ) );
            }// for
        }// for
    }// if
    else
    {
        A.copy_to( & B );
    }// else
}

//
// reallocate matrix blocks
// - frees old data
// - local operation thereby limiting extra memory usage
//
inline
std::unique_ptr< TMatrix >
realloc ( TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, TBlockMatrix );
        auto  C  = std::make_unique< TBlockMatrix >();
        auto  BC = ptrcast( C.get(), TBlockMatrix );

        C->copy_struct_from( B );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  C_ij = realloc( B->block( i, j ) );

                BC->set_block( i, j, C_ij.release() );
                B->set_block( i, j, nullptr );
            }// for
        }// for

        delete B;

        return C;
    }// if
    else
    {
        auto  C = A->copy();

        delete A;

        return C;
    }// else
}

//
// nullify data in matrix, e.g., M := 0
//
inline
void
clear ( TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( & M, TBlockMatrix );

        for ( uint  i = 0; i < BM->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BM->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    clear( * BM->block( i, j ) );
                }// if
            }// for
        }// for
    }// if
    else if ( is_lowrank( & M ) )
    {
        auto  R = ptrcast( & M, TRkMatrix );

        R->set_rank( 0 );
    }// if
    else if ( is_dense( & M ) )
    {
        auto  D = ptrcast( & M, TDenseMatrix );

        if ( D->is_complex() )
            BLAS::fill( HLIB::complex(0), blas_mat< HLIB::complex >( D ) );
        else
            BLAS::fill( HLIB::real(0), blas_mat< HLIB::real >( D ) );
    }// if
    else
        assert( false );
}

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_MATRIX_HH
