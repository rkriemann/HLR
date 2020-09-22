#ifndef __HLR_SEQ_ARITH_UNIFORM_HH
#define __HLR_SEQ_ARITH_UNIFORM_HH
//
// Project     : HLib
// Module      : seq/arith_uniform.hh
// Description : sequential arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hlr/arith/blas.hh>
#include <hlr/matrix/cluster_basis.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/vector/scalar_vector.hh>
#include <hlr/vector/uniform_vector.hh>

namespace hlr { namespace seq { namespace uniform {

namespace hpro = HLIB;

///////////////////////////////////////////////////////////////////////
//
// general arithmetic functions
//
///////////////////////////////////////////////////////////////////////

//
// compute y = y + α op( M ) x
//
namespace detail
{

using matrix::cluster_basis;
using vector::scalar_vector;
using vector::uniform_vector;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
template < typename value_t >
void
mul_vec ( const value_t                                       alpha,
          const hpro::matop_t                                 op_M,
          const hpro::TMatrix &                               M,
          const uniform_vector< cluster_basis< value_t > > &  x,
          uniform_vector< cluster_basis< value_t > > &        y,
          const scalar_vector< value_t > &                    sx,
          scalar_vector< value_t > &                          sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

        if ( ! (( B->nblock_rows( op_M ) == y.nblocks() ) &&
                ( B->nblock_cols( op_M ) == x.nblocks() )) )
            HLR_ERROR( "matrix/vector block structure incompatible" );
            
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            auto  y_i = y.block( i );
            
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                auto  B_ij = B->block( i, j, op_M );
                auto  x_j  = x.block( j );
            
                if ( ! is_null( B_ij ) )
                {
                    mul_vec( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy );
                }// if
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D   = cptrcast( &M, hpro::TDenseMatrix );
        auto  x_i = blas::vector< value_t >( hpro::blas_vec< value_t >( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( hpro::blas_vec< value_t >( sy ), M.row_is( op_M ) - sy.ofs() );
        
        blas::mulvec( alpha, blas::mat_view( op_M, hpro::blas_mat< value_t >( D ) ), x_i, value_t(1), y_j );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        
        if ( op_M == hpro::apply_normal )
        {
            blas::mulvec( value_t(1), R->coeff(), x.coeffs(), value_t(1), y.coeffs() );
        }// if
        else if ( op_M == hpro::apply_transposed )
        {
            HLR_ASSERT( false );
        }// if
        else if ( op_M == hpro::apply_adjoint )
        {
            blas::mulvec( value_t(1), blas::adjoint(R->coeff()), x.coeffs(), value_t(1), y.coeffs() );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// copy given scalar vector into uniform vector format
//
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
scalar_to_uniform ( const cluster_basis< value_t > &  cb,
                    const scalar_vector< value_t > &  v )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.cluster(), cb );

    if ( cb.rank() > 0 )
    {
        auto  v_cb = blas::vector< value_t >( hpro::blas_vec< value_t >( v ), cb.cluster() - v.ofs() );
        auto  s    = cb.transform_forward( v_cb );

        u->set_coeffs( std::move( s ) );
    }// if

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            u->set_block( i, scalar_to_uniform( *cb.son(i), v ).release() );
    }// if

    return u;
}

//
// create empty uniform vector for given cluster basis
//
template < typename value_t >
std::unique_ptr< uniform_vector< cluster_basis< value_t > > >
make_uniform ( const cluster_basis< value_t > &  cb )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis< value_t > > >( cb.cluster(), cb );

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
            u->set_block( i, make_uniform( *cb.son(i) ).release() );
    }// if

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t >
void
add_uniform_to_scalar ( const uniform_vector< cluster_basis< value_t > > &  u,
                        scalar_vector< value_t > &                          v )
{
    if ( u.basis().rank() > 0 )
    {
        auto  x   = u.basis().transform_backward( u.coeffs() );
        auto  v_u = blas::vector< value_t >( hpro::blas_vec< value_t >( v ), u.is() - v.ofs() );
            
        blas::add( value_t(1), x, v_u );
    }// if

    if ( u.nblocks() > 0 )
    {
        for ( uint  i = 0; i < u.nblocks(); ++i )
            add_uniform_to_scalar( *u.block(i), v );
    }// if
}
    
}// namespace detail

template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const hpro::matop_t                       op_M,
          const hpro::TMatrix &                     M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y,
          matrix::cluster_basis< value_t > &        rowcb,
          matrix::cluster_basis< value_t > &        colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb );

    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y );
}

//////////////////////////////////////////////////////////////////////
//
// TLR versions
//
//////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// matrix multiplication
//
template < typename value_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    aA,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    aB,
           hpro::TMatrix &          aC,
           const hpro::TTruncAcc &  acc )
{
    HLR_ASSERT( is_blocked_all( aA, aB, aC ) );

    auto  A = cptrcast( &aA, hpro::TBlockMatrix );
    auto  B = cptrcast( &aB, hpro::TBlockMatrix );
    auto  C = cptrcast( &aC, hpro::TBlockMatrix );

    HLR_ASSERT( C->nblock_rows()       == A->nblock_rows( op_A ) );
    HLR_ASSERT( C->nblock_cols()       == B->nblock_cols( op_B ) );
    HLR_ASSERT( A->nblock_cols( op_A ) == B->nblock_rows( op_B ) );

    for ( uint  i = 0; i < C->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C->nblock_cols(); ++j )
        {
            auto  C_ij = C->block( i, j );

            HLR_ASSERT( ! is_null( C_ij ) );

            for ( uint  k = 0; k < A->nblock_cols( op_A ); ++k )
            {
                auto  A_ik = A->block( i, k, op_A );
                auto  B_kj = B->block( k, j, op_B );

                HLR_ASSERT( ! is_null_any( A_ik, B_kj ) );

                //
                // due to TLR format, C_ij, A_ik and B_kj can only be dense or uniform-lowrank
                // hence, handle all combinations
                //

                if ( matrix::is_uniform_lowrank( C_ij ) )
                {
                    auto  RC_ij = ptrcast( C_ij, matrix::uniform_lrmatrix< value_t > );
                    
                    if ( matrix::is_uniform_lowrank( A_ik ) )
                    {
                        auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );

                            blas::prod( alpha,
                                        blas::mat_view( op_A, RA_ik->coeff() ),
                                        blas::mat_view( op_B, RB_kj->coeff() ),
                                        value_t(1),
                                        RC_ij->coeff() );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else if ( is_dense( A_ik ) )
                    {
                        auto  DA_ik = cptrcast( A_ik, hpro::TDenseMatrix );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
                }// if
                else if ( is_dense( C_ij ) )
                {
                    auto  DC_ij = ptrcast( C_ij, hpro::TDenseMatrix );
                    
                    if ( matrix::is_uniform_lowrank( A_ik ) )
                    {
                        auto  RA_ik = cptrcast( A_ik, matrix::uniform_lrmatrix< value_t > );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            // C = C + U_A S_A V_A^T · U_B S_B V_B^T
                            //   = C + U_A ( S_A S_B ) V_B^T
                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                            auto  S     = blas::prod( alpha,
                                                      blas::mat_view( op_A, RA_ik->coeff() ),
                                                      blas::mat_view( op_B, RB_kj->coeff() ) );
                            auto  UT    = blas::prod( value_t(1),
                                                      RA_ik->row_cb( op_A ).basis(),
                                                      S );

                            blas::prod( value_t(1), UT, RB_kj->col_cb( op_B ).basis(),
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            // C = C + U_A ( S_A ( V_A^T · B ) )
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );
                            auto  VB    = blas::prod( value_t(1),
                                                      RA_ik->col_cb( op_A ).basis(),
                                                      blas::mat_view( op_B, hpro::blas_mat< value_t >( DB_kj ) ) );
                            auto  SVB   = blas::prod( alpha,
                                                      blas::mat_view( op_A, RA_ik->coeff() ),
                                                      VB );

                            blas::prod( value_t(1), RA_ik->row_cb( op_A ).basis(), SVB,
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else if ( is_dense( A_ik ) )
                    {
                        auto  DA_ik = cptrcast( A_ik, hpro::TDenseMatrix );
                        
                        if ( matrix::is_uniform_lowrank( B_kj ) )
                        {
                            // C = C + ( ( A · U_B ) S_B ) V_B^T
                            auto  RB_kj = cptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                            auto  AU    = blas::prod( value_t(1),
                                                      blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                                      RB_kj->row_cb( op_B ).basis() );
                            auto  AUS   = blas::prod( alpha,
                                                      AU,
                                                      blas::mat_view( op_B, RB_kj->coeff() ) );

                            blas::prod( value_t(1), AUS, RB_kj->col_cb( op_B ).basis(),
                                        value_t(1), hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else if ( is_dense( B_kj ) )
                        {
                            // C = C + A · B
                            auto  DB_kj = cptrcast( B_kj, hpro::TDenseMatrix );

                            blas::prod( alpha,
                                        blas::mat_view( op_A, hpro::blas_mat< value_t >( DA_ik ) ),
                                        blas::mat_view( op_B, hpro::blas_mat< value_t >( DB_kj ) ),
                                        value_t(1),
                                        hpro::blas_mat< value_t >( DC_ij ) );
                        }// if
                        else
                            HLR_ERROR( "unsupported matrix type : " + B_kj->typestr() );
                    }// if
                    else
                        HLR_ERROR( "unsupported matrix type : " + A_ik->typestr() );
                }// if
                else
                    HLR_ERROR( "unsupported matrix type : " + C_ij->typestr() );
            }// for
        }// for
    }// for
}

}// namespace tlr

}}}// namespace hlr::seq::uniform

#endif // __HLR_SEQ_ARITH_UNIFORM_HH
