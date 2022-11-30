#ifndef __HLR_ARITH_LU_HH
#define __HLR_ARITH_LU_HH
//
// Project     : HLib
// Module      : arith/lu
// Description : LU factorization functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hlr/arith/multiply.hh>
#include <hlr/arith/solve.hh>
#include <hlr/arith/invert.hh>
#include <hlr/matrix/level_matrix.hh>
#include <hlr/seq/matrix.hh>
#include <hlr/utils/checks.hh>
#include <hlr/utils/io.hh> // DEBUG

namespace hlr {

////////////////////////////////////////////////////////////////////////////////
//
// general H-LU factorization
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &  A,
     const Hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            
            lu( * BA->block( i, i ), acc, approx );

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                    solve_upper_tri( from_right, general_diag, *BA->block( i, i ), *BA->block( j, i ), acc, approx );
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                    solve_lower_tri( from_left, unit_diag, *BA->block( i, i ), *BA->block( i, j ), acc, approx );
            }// for

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                {
                    if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                    {
                        HLR_ASSERT( ! is_null( BA->block( j, l ) ) );
                    
                        multiply( value_t(-1),
                                  apply_normal, *BA->block( j, i ),
                                  apply_normal, *BA->block( i, l ),
                                  *BA->block( j, l ), acc, approx );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else if ( is_dense( A ) )
    {
        auto  D = ptrcast( &A, Hpro::TDenseMatrix< value_t > );

        invert< value_t >( *D );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// level-wise LU factorization
//
////////////////////////////////////////////////////////////////////////////////

//
// compute LU factorization of A within L
//
template < typename value_t,
           typename approx_t >
void
lu ( Hpro::TMatrix< value_t > &         A,
     matrix::level_matrix< value_t > &  L,
     const Hpro::TTruncAcc &            acc,
     const approx_t &                   approx )
{
    // DBG::printf( "lu( %d )", A.id() );

    ///////////////////////////////////////////////////////////////
    //
    // either factorise diagonal block or recurse to visit all
    // diagonal blocks
    //
    ///////////////////////////////////////////////////////////////

    const bool  A_is_leaf = ( is_leaf( A ) || is_small( A ) );
    
    if ( A_is_leaf )
    {
        lu( & A, acc, approx );
    }// if
    else
    {
        auto        B      = ptrcast( & A, Hpro::TBlockMatrix< value_t > );
        const uint  nbrows = B->nblock_rows();
        const uint  nbcols = B->nblock_cols();

        for ( uint  i = 0; i < std::min( nbrows, nbcols ); ++i )
        {
            auto  A_ii = B->block( i, i );
            
            assert( ! is_null( A_ii ) );
            
            lu( * A_ii, *( L.below() ), acc, approx );
        }// for
    }// if

    ///////////////////////////////////////////////////////////////
    //
    // actual factorisation of A, solving in block row/column of A
    // and update of trailing sub matrix with respect to A (all on L)
    //
    ///////////////////////////////////////////////////////////////
    
    //
    // get block row/column of A in L
    //

    const auto  [ bi, bj ] = L.get_index( A );
    const auto  nbrows     = L.nblock_rows();
    const auto  nbcols     = L.nblock_cols();

    // check if found in L
    assert(( bi != L.nblock_rows() ) && ( bj != L.nblock_cols() ));
    
    // should be on diagonal
    assert( bi == bj );
    
    //
    // off-diagonal solves in current block row/column
    //
    
    for ( uint  j = bi+1; j < nbcols; ++j )
    {
        auto  L_ij = L.block( bi, j );
            
        if ( ! is_null( L_ij ) && ( is_leaf( L_ij ) || A_is_leaf ))
        {
            // DBG::printf( "solve_lower_left( %d, %d )", A.id(), L_ij->id() );

            solve_lower_tri( from_left, unit_diag, A, *L_ij, acc, approx );
        }// if
    }// for
        
    for ( uint  j = bi+1; j < nbrows; ++j )
    {
        auto  L_ji = L.block( j, bi );
            
        if ( ! is_null( L_ji ) && ( is_leaf( L_ji ) || A_is_leaf ))
        {
            // DBG::printf( "solve_upper_right( %d, %d )", A.id(), L_ji->id() );

            solve_upper_tri( from_right, general_diag, A, *L_ji, acc, approx );
        }// if
    }// for

    //
    // update of trailing sub matrix
    //

    for ( uint  j = bi+1; j < nbrows; ++j )
    {
        auto  L_ji = L.block( j, bi );
            
        for ( uint  l = bi+1; l < nbcols; ++l )
        {
            auto  L_il = L.block( bi, l );
            auto  L_jl = L.block(  j, l );
            
            if ( ! is_null_any( L_ji, L_il, L_jl ) && ( is_leaf_any( L_ji, L_il, L_jl ) || A_is_leaf ))
            {
                // DBG::printf( "update( %d, %d, %d )", L_ji->id(), L_il->id(), L_jl->id() );
                
                multiply( value_t(-1),
                          apply_normal, *L_ji,
                          apply_normal, *L_il,
                          *L_jl, acc, approx );
            }// if
        }// for
    }// for
}

//
// factorize given level matrix
// - only calls actual LU algorithm for all diagonal blocks
//
template < typename value_t,
           typename approx_t >
void
lu ( matrix::level_matrix< value_t > &  A,
     const Hpro::TTruncAcc &            acc,
     const approx_t &                   approx )
{
    const uint  nbrows = A.nblock_rows();
    const uint  nbcols = A.nblock_cols();

    for ( uint  i = 0; i < std::min( nbrows, nbcols ); ++i )
    {
        auto  A_ii = A.block( i, i );

        assert( ! is_null( A_ii ) );
        
        lu( * A_ii, A, acc, approx );
    }// for
}

////////////////////////////////////////////////////////////////////////////////
//
// general H-LDU factorization
//
////////////////////////////////////////////////////////////////////////////////

namespace detail
{

template < typename value_t,
           typename approx_t >
void
ldu ( Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &  D,
      const Hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    if ( is_blocked( A ) )
    {
        HLR_ASSERT( is_blocked( D ) );
        
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        auto  BD = ptrcast( &D, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT(( BA->nblock_rows() == BD->nblock_rows() ) &&
                   ( BA->nblock_cols() == BD->nblock_cols() ));
        
        for ( uint  i = 0; i < std::min( BA->nblock_rows(), BA->nblock_cols() ); ++i )
        {
            HLR_ASSERT( ! is_null( BA->block( i, i ) ) );
            HLR_ASSERT( ! is_null( BD->block( i, i ) ) );
            
            ldu< value_t >( * BA->block( i, i ), *BD->block( i, i ), acc, approx );

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                if ( ! is_null( BA->block( j, i ) ) )
                {
                    solve_upper_tri< value_t >( from_right, unit_diag, *BA->block( i, i ), *BA->block( j, i ), acc, approx );
                    solve_diag< value_t >(      from_right, general_diag, apply_normal, *BA->block( i, i ), *BA->block( j, i ), acc, approx );
                }// if
            }// for

            for ( uint  j = i+1; j < BA->nblock_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    solve_lower_tri< value_t >( from_left, unit_diag, *BA->block( i, i ), *BA->block( i, j ), acc, approx );
                    solve_diag< value_t >(      from_left, general_diag, apply_normal, *BA->block( i, i ), *BA->block( i, j ), acc, approx );
                }// if
            }// for

            for ( uint  j = i+1; j < BA->nblock_rows(); ++j )
            {
                for ( uint  l = i+1; l < BA->nblock_cols(); ++l )
                {
                    if ( ! is_null_any( BA->block( j, i ), BA->block( i, l ) ) )
                    {
                        HLR_ASSERT( ! is_null( BA->block( j, l ) ) );
                    
                        multiply_diag( value_t(-1),
                                       apply_normal, *BA->block( j, i ),
                                       apply_normal, *BD->block( i, i ),
                                       apply_normal, *BA->block( i, l ),
                                       *BA->block( j, l ), acc, approx );

                        auto  T = matrix::convert_to_dense< value_t >( *BA->block( j, l ) );

                        io::matlab::write( *T, "T" );
                    }// if
                }// for
            }// for
        }// for
    }// if
    else if ( is_dense( A ) )
    {
        auto  DA = ptrcast( &A, Hpro::TDenseMatrix< value_t > );

        A.copy_to( &D );
        invert< value_t >( *DA );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );
}

}// namespace detail

template < typename value_t,
           typename approx_t >
void
ldu ( Hpro::TMatrix< value_t > &  A,
      const Hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    auto  D = seq::matrix::copy_diag( A );

    detail::ldu< value_t, approx_t >( A, *D, acc, approx );
}
    
}// namespace hlr

#endif // __HLR_ARITH_LU_HH
