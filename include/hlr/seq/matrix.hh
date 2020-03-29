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

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>
#include <hpro/base/TTruncAcc.hh>

#include "hlr/utils/checks.hh"
#include "hlr/arith/blas.hh"
#include "hlr/matrix/cluster_basis.hh"
#include "hlr/matrix/uniform_lrmatrix.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"

namespace hlr { namespace seq { namespace matrix {

namespace hpro = HLIB;
    
//
// build representation of dense matrix with
// matrix structure defined by <bct>,
// matrix coefficients defined by <coeff> and
// low-rank blocks computed by <lrapx>
//
template < typename coeff_t,
           typename lrapx_t >
std::unique_ptr< hpro::TMatrix >
build ( const hpro::TBlockCluster *  bct,
        const coeff_t &              coeff,
        const lrapx_t &              lrapx,
        const hpro::TTruncAcc &      acc,
        const size_t                 nseq = hpro::CFG::Arith::max_seq_size ) // ignored
{
    static_assert( std::is_same< typename coeff_t::value_t,
                                 typename lrapx_t::value_t >::value,
                   "coefficient function and low-rank approximation must have equal value type" );
    
    assert( bct != nullptr );
    
    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc ) );
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
        }// else
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

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
                    auto  B_ij = build( bct->son( i, j ), coeff, lrapx, acc, nseq );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

//
// return copy of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

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

//
// return copy of matrix with TRkMatrix replaced by tiled_lrmatrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_tiled ( const hpro::TMatrix &  M,
             const size_t           ntile )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

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
    else if ( is_lowrank( M ) )
    {
        //
        // copy low-rank data into tiled form
        //

        auto  RM = cptrcast( & M, hpro::TRkMatrix );
        auto  R  = std::make_unique< hlr::matrix::tiled_lrmatrix< value_t > >( RM->row_is(),
                                                                               RM->col_is(),
                                                                               ntile,
                                                                               hpro::blas_mat_A< value_t >( RM ),
                                                                               hpro::blas_mat_B< value_t >( RM ) );

        R->set_id( RM->id() );

        return R;
    }// if
    else
    {
        // assuming non-structured block
        return M.copy();
    }// else
}

//
// return copy of matrix with tiled_lrmatrix replaced by TRkMatrix
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_nontiled ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_nontiled< value_t >( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( IS_TYPE( & M, tiled_lrmatrix ) )
    {
        //
        // copy low-rank data into tiled form
        //

        assert( M.is_real() );
        
        auto  RM = cptrcast( & M, hlr::matrix::tiled_lrmatrix< real > );
        auto  R  = std::make_unique< hpro::TRkMatrix >( RM->row_is(), RM->col_is() );
        auto  U  = hlr::matrix::to_dense( RM->U() );
        auto  V  = hlr::matrix::to_dense( RM->V() );

        R->set_lrmat( U, V );
        R->set_id( RM->id() );

        return R;
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
std::unique_ptr< hpro::TMatrix >
copy_ll ( const hpro::TMatrix &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

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

        if ( diag == hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                assert( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = blas::identity< hpro::complex >( D->nrows() );
                else
                    D->blas_rmat() = blas::identity< hpro::real >( D->nrows() );
            }// if
        }// if

        return T;
    }// else
}

//
// return copy of (block-wise) upper-right part of matrix
//
inline
std::unique_ptr< hpro::TMatrix >
copy_ur ( const hpro::TMatrix &    M,
          const hpro::diag_type_t  diag = hpro::general_diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

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

        if ( diag == hpro::unit_diag )
        {
            if ( T->row_is() == T->col_is() )
            {
                assert( is_dense( T.get() ) );

                auto  D = ptrcast( T.get(), hpro::TDenseMatrix );

                if ( D->is_complex() )
                    D->blas_cmat() = blas::identity< hpro::complex >( D->nrows() );
                else
                    D->blas_rmat() = blas::identity< hpro::real >( D->nrows() );
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
copy_to ( const hpro::TMatrix &  A,
          hpro::TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

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
copy_to_ll ( const hpro::TMatrix &  A,
             hpro::TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

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
copy_to_ur ( const hpro::TMatrix &  A,
             hpro::TMatrix &        B )
{
    assert( A.type()     == B.type() );
    assert( A.block_is() == B.block_is() );
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, hpro::TBlockMatrix );
        auto  BB = ptrcast(  &B, hpro::TBlockMatrix );

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
std::unique_ptr< hpro::TMatrix >
realloc ( hpro::TMatrix *  A )
{
    if ( is_null( A ) )
        return nullptr;
    
    if ( is_blocked( A ) )
    {
        auto  B  = ptrcast( A, hpro::TBlockMatrix );
        auto  C  = std::make_unique< hpro::TBlockMatrix >();
        auto  BC = ptrcast( C.get(), hpro::TBlockMatrix );

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
clear ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = ptrcast( & M, hpro::TBlockMatrix );

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
        auto  R = ptrcast( & M, hpro::TRkMatrix );

        R->set_rank( 0 );
    }// if
    else if ( is_dense( & M ) )
    {
        auto  D = ptrcast( & M, hpro::TDenseMatrix );

        if ( D->is_complex() )
            blas::fill( hpro::blas_mat< hpro::complex >( D ), hpro::complex(0) );
        else
            blas::fill( hpro::blas_mat< hpro::real >( D ), hpro::real(0) );
    }// if
    else
        assert( false );
}

//
// return copy of matrix with uniform low-rank matrices
// - TODO: add cluster basis as template argument to allow
//         different bases
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
copy_uniform ( const hpro::TMatrix &                          M,
               const hlr::matrix::cluster_basis< value_t > &  rowcb,
               const hlr::matrix::cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );

        HLR_ASSERT( B->nblock_rows() == rowcb.nsons() );
        HLR_ASSERT( B->nblock_cols() == colcb.nsons() );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = copy_uniform( * BM->block( i, j ), * rowcb.son(i), * colcb.son(j) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_lowrank( M ) )
    {
        //
        // project into row/column cluster basis:
        //
        //   M = A·B^H = (U·U^H·A) (V·V^H·B)^H
        //             = U · (U^H·A)·(V^H·B)^H · V^H
        //             = U · S · V^H   with  S = (U^H·A)·(V^H·B)^H
        //
        
        auto  R  = cptrcast( &M, hpro::TRkMatrix );

        auto  UA = rowcb.transform_forward( hpro::blas_mat_A< value_t >( R ) );
        auto  VB = colcb.transform_forward( hpro::blas_mat_B< value_t >( R ) );
        auto  S  = blas::prod( value_t(1), UA, blas::adjoint( VB ) );

        // {
        //     auto  M1 = blas::prod( value_t(1), hpro::blas_mat_A< value_t >( R ), blas::adjoint( hpro::blas_mat_B< value_t >( R ) ) );
        //     auto  T  = blas::prod( value_t(1), rowcb.basis(), S );
        //     auto  M2 = blas::prod( value_t(1), T, blas::adjoint( colcb.basis() ) );
            
        //     blas::add( value_t(-1), M2, M1 );
            
        //     std::cout << hpro::to_string( "%.6e", blas::norm_F( M1 ) ) << std::endl;
        // }
        
        return std::make_unique< hlr::matrix::uniform_lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                             rowcb, colcb,
                                                                             std::move( S ) );
    }// if
    else
    {
        // assuming dense block (no low-rank)
        return M.copy();
    }// else
}

//
// convert given matrix into lowrank format
//
template < typename approx_t >
std::unique_ptr< hpro::TRkMatrix >
convert_to_lowrank ( const hpro::TMatrix &    M,
                     const hpro::TTruncAcc &  acc,
                     const approx_t &         approx )
{
    using  value_t = typename approx_t::value_t;
    
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block into low-rank format and 
        // enlarge to size of M (pad with zeroes)
        //

        auto  B  = cptrcast( &M, hpro::TBlockMatrix );
        auto  Us = std::list< blas::matrix< value_t > >();
        auto  Vs = std::list< blas::matrix< value_t > >();

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( is_null( B_ij ) )
                    continue;

                auto  R_ij = convert_to_lowrank( *B_ij, acc, approx );
                auto  U    = blas::matrix< value_t >( M.nrows(), R_ij->rank() );
                auto  V    = blas::matrix< value_t >( M.ncols(), R_ij->rank() );
                auto  U_i  = blas::matrix< value_t >( U, R_ij->row_is() - M.row_ofs(), blas::range::all );
                auto  V_j  = blas::matrix< value_t >( V, R_ij->col_is() - M.col_ofs(), blas::range::all );

                blas::copy( hpro::blas_mat_A< value_t >( R_ij ), U_i );
                blas::copy( hpro::blas_mat_B< value_t >( R_ij ), V_j );

                Us.push_back( std::move( U ) );
                Vs.push_back( std::move( V ) );
            }// for
        }// for

        auto  [ U, V ] = approx( Us, Vs, acc );

        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_dense( M ) )
    {
        auto  D        = cptrcast( &M, hpro::TDenseMatrix );
        auto  T        = std::move( blas::copy( hpro::blas_mat< value_t >( D ) ) );
        auto  [ U, V ] = approx( T, acc );

        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, hpro::TRkMatrix );
        auto  [ U, V ] = approx( hpro::blas_mat_A< value_t >( R ),
                                 hpro::blas_mat_B< value_t >( R ),
                                 acc );
        
        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_MATRIX_HH
