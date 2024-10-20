#ifndef __HLR_MATRIX_CONVERT_HH
#define __HLR_MATRIX_CONVERT_HH
//
// Project     : HLR
// Module      : matrix/convert
// Description : matrix conversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include <hlr/approx/traits.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/uniform_lr2matrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/h2_lr2matrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>

namespace hlr { namespace matrix {

//
// convert given matrix into lowrank format with truncation
//
template < typename                   value_t,
           approx::approximation_type approx_t >
std::unique_ptr< lrmatrix< value_t > >
convert_to_lowrank ( const Hpro::TMatrix< value_t > &  M,
                     const Hpro::TTruncAcc &           acc,
                     const approx_t &                  approx )
{
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block into low-rank format and 
        // enlarge to size of M (pad with zeroes)
        //

        auto  B  = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
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

                blas::copy( R_ij->U(), U_i );
                blas::copy( R_ij->V(), V_j );

                Us.push_back( std::move( U ) );
                Vs.push_back( std::move( V ) );
            }// for
        }// for

        auto  [ U, V ] = approx( Us, Vs, acc );

        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, lrmatrix< value_t > );
        auto  U        = R->U();
        auto  V        = R->V();
        auto  [ W, X ] = approx( U, V, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  R        = cptrcast( &M, lrsvmatrix< value_t > );
        auto  U        = blas::prod_diag( R->U(), R->S() );
        auto  V        = R->V();
        auto  [ W, X ] = approx( U, V, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( W ), std::move( X ) );
    }// if
    else if ( Hpro::is_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, lrmatrix< value_t > );
        auto  [ U, V ] = approx( R->U(), R->V(), acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  US       = blas::prod( R->row_basis(), R->coupling() );
        auto  [ U, V ] = approx( US, R->col_basis(), acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_uniform_lowrank2( M ) )
    {
        auto  R        = cptrcast( &M, uniform_lr2matrix< value_t > );
        auto  US       = blas::prod( R->row_basis(), R->row_coupling() );
        auto  VS       = blas::prod( R->col_basis(), R->col_coupling() );
        auto  [ U, V ] = approx( US, VS, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, h2_lrmatrix< value_t > );
        auto  U        = R->row_cb().transform_backward( R->coupling() );
        auto  I        = blas::identity< value_t >( R->col_rank() );
        auto  V        = R->col_cb().transform_backward( I );
        auto  [ W, X ] = approx( U, V, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_h2_lowrank2( M ) )
    {
        auto  R        = cptrcast( &M, h2_lr2matrix< value_t > );
        auto  U        = R->row_cb().transform_backward( R->row_coupling() );
        auto  V        = R->col_cb().transform_backward( R->col_coupling() );
        auto  [ W, X ] = approx( U, V, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( W ), std::move( X ) );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D        = cptrcast( &M, matrix::dense_matrix< value_t > );
        auto  T        = blas::copy( D->mat() );
        auto  [ U, V ] = approx( T, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( Hpro::is_dense( M ) )
    {
        auto  D        = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  T        = blas::copy( D->blas_mat() );
        auto  [ U, V ] = approx( T, acc );
        
        return std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_sparse( M ) )
    {
        auto  S = cptrcast( &M, Hpro::TSparseMatrix< value_t > );

        // use each non-zero row of S as a column vector and
        // set the corresponding row vector to the unit vector

        const auto  nrows = S->nrows();
        const auto  ncols = S->ncols();
        uint        k     = 0;

        // determine rank (number of non-zero rows)
        for ( size_t  i = 0; i < nrows; ++i )
            k += ( S->rowptr(i) != S->rowptr(i+1) ? 1 : 0 );

        // copy coefficients
        auto  U = blas::matrix< value_t >( nrows, k );
        auto  V = blas::matrix< value_t >( ncols, k );

        // reuse as counter
        k = 0;
        
        for ( size_t  i = 0; i < nrows; ++i )
        {
            auto  lb = S->rowptr(i);
            auto  ub = S->rowptr(i+1);

            if ( lb == ub )
                continue;

            U(i,k) = value_t(1);
            
            for ( auto  l = lb; l < ub; ++l )
                V( S->colind(l), k ) = math::conj( Hpro::coeff< value_t >( S, l ) );

            ++k;
        }// for

        auto  [ W, X ] = approx( U, V, acc );
        
        return std::make_unique< lrmatrix< value_t > >( S->row_is(), S->col_is(), std::move( W ), std::move( X ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// convert given matrix into lowrank U·S·V' format with truncation
//
template < typename                   value_t,
           approx::approximation_type approx_t >
std::unique_ptr< lrsvmatrix< value_t > >
convert_to_lowrank_sv ( const Hpro::TMatrix< value_t > &  M,
                        const Hpro::TTruncAcc &           acc,
                        const approx_t &                  approx )
{
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block into low-rank format and 
        // enlarge to size of M (pad with zeroes)
        //

        auto  B  = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  Us = std::list< blas::matrix< value_t > >();
        auto  Vs = std::list< blas::matrix< value_t > >();

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( is_null( B_ij ) )
                    continue;

                auto  R_ij = convert_to_lowrank_sv( *B_ij, acc, approx );
                auto  U    = blas::matrix< value_t >( M.nrows(), R_ij->rank() );
                auto  V    = blas::matrix< value_t >( M.ncols(), R_ij->rank() );
                auto  U_i  = blas::matrix< value_t >( U, R_ij->row_is() - M.row_ofs(), blas::range::all );
                auto  V_j  = blas::matrix< value_t >( V, R_ij->col_is() - M.col_ofs(), blas::range::all );

                blas::copy( R_ij->U(), U_i );
                blas::prod_diag_ip( U_i, R_ij->S() );
                blas::copy( R_ij->V(), V_j );

                Us.push_back( std::move( U ) );
                Vs.push_back( std::move( V ) );
            }// for
        }// for

        auto  [ U, S, V ] = approx.approx_ortho( Us, Vs, acc );

        return std::make_unique< matrix::lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  R           = cptrcast( &M, lrmatrix< value_t > );
        auto  [ U, S, V ] = approx.approx_ortho( R->U(), R->V(), acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  R           = cptrcast( &M, lrsvmatrix< value_t > );
        auto  U           = blas::prod_diag( R->U(), R->S() );
        auto  V           = R->V();
        auto  [ W, S, X ] = approx.approx_ortho( U, V, acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( W ), std::move( S ), std::move( X ) );
    }// if
    else if ( Hpro::is_lowrank( M ) )
    {
        auto  R           = cptrcast( &M, lrmatrix< value_t > );
        auto  [ U, S, V ] = approx.approx_ortho( R->U(), R->V(), acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R           = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  US          = blas::prod( R->row_basis(), R->coupling() );
        auto  [ U, S, V ] = approx.approx_ortho( US, R->col_basis(), acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( matrix::is_uniform_lowrank2( M ) )
    {
        auto  R           = cptrcast( &M, uniform_lr2matrix< value_t > );
        auto  Ur          = blas::prod( R->row_basis(), R->row_coupling() );
        auto  Vc          = blas::prod( R->col_basis(), R->col_coupling() );
        auto  [ U, S, V ] = approx.approx_ortho( Ur, Vc, acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R           = cptrcast( &M, h2_lrmatrix< value_t > );
        auto  U           = R->row_cb().transform_backward( R->coupling() );
        auto  I           = blas::identity< value_t >( R->col_rank() );
        auto  V           = R->col_cb().transform_backward( I );
        auto  [ W, S, X ] = approx.approx_ortho( U, V, acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( W ), std::move( S ), std::move( X ) );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  D           = cptrcast( &M, matrix::dense_matrix< value_t > );
        auto  T           = blas::copy( D->mat() );
        auto  [ U, S, V ] = approx.approx_ortho( T, acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( Hpro::is_dense( M ) )
    {
        auto  D           = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  T           = blas::copy( D->blas_mat() );
        auto  [ U, S, V ] = approx.approx_ortho( T, acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( S ), std::move( V ) );
    }// if
    else if ( is_sparse( M ) )
    {
        auto  S = cptrcast( &M, Hpro::TSparseMatrix< value_t > );

        // use each non-zero row of S as a column vector and
        // set the corresponding row vector to the unit vector

        const auto  nrows = S->nrows();
        const auto  ncols = S->ncols();
        uint        k     = 0;

        // determine rank (number of non-zero rows)
        for ( size_t  i = 0; i < nrows; ++i )
            k += ( S->rowptr(i) != S->rowptr(i+1) ? 1 : 0 );

        // copy coefficients
        auto  U = blas::matrix< value_t >( nrows, k );
        auto  V = blas::matrix< value_t >( ncols, k );

        // reuse as counter
        k = 0;
        
        for ( size_t  i = 0; i < nrows; ++i )
        {
            auto  lb = S->rowptr(i);
            auto  ub = S->rowptr(i+1);

            if ( lb == ub )
                continue;

            U(i,k) = value_t(1);
            
            for ( auto  l = lb; l < ub; ++l )
                V( S->colind(l), k ) = math::conj( Hpro::coeff< value_t >( S, l ) );

            ++k;
        }// for

        auto  [ W, sv, X ] = approx.approx_ortho( U, V, acc );
        
        return std::make_unique< lrsvmatrix< value_t > >( S->row_is(), S->col_is(), std::move( W ), std::move( sv ), std::move( X ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// convert given matrix into lowrank format without truncation
// (only implemented for lowrank compatible formats)
//
template < typename value_t >
std::unique_ptr< lrmatrix< value_t > >
convert_to_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    if ( matrix::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::lrmatrix< value_t > );
        
        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(),
                                                                std::move( blas::copy( R->U() ) ),
                                                                std::move( blas::copy( R->V() ) ) );
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  U = blas::prod( R->row_basis(), R->coupling() );
        auto  V = blas::copy( R->col_basis() );
        
        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_uniform_lowrank2( M ) )
    {
        auto  R = cptrcast( &M, uniform_lr2matrix< value_t > );
        auto  U = blas::prod( R->row_basis(), R->row_coupling() );
        auto  V = blas::prod( R->col_basis(), R->col_coupling() );
        
        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R = cptrcast( &M, h2_lrmatrix< value_t > );
        auto  U = R->row_cb().transform_backward( R->coupling() );
        auto  I = blas::identity< value_t >( R->col_rank() );
        auto  V = R->col_cb().transform_backward( I );
        
        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( matrix::is_h2_lowrank2( M ) )
    {
        auto  R = cptrcast( &M, h2_lr2matrix< value_t > );
        auto  U = R->row_cb().transform_backward( R->row_coupling() );
        auto  V = R->col_cb().transform_backward( R->col_coupling() );
        
        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// convert given matrix into dense format
//
template < typename value_t >
std::unique_ptr< dense_matrix< value_t > >
convert_to_dense ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block and copy to dense version
        //

        auto  B  = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  D  = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is() );
        auto  DD = D->mat();

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( is_null( B_ij ) )
                    continue;

                auto  D_ij  = convert_to_dense< value_t >( *B_ij );
                auto  DD_ij = D_ij->mat();
                auto  TD    = blas::matrix< value_t >( DD,
                                                       D_ij->row_is() - M.row_ofs(),
                                                       D_ij->col_is() - M.col_ofs() );

                blas::copy( DD_ij, TD );
            }// for
        }// for

        return D;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  R  = cptrcast( &M, lrmatrix< value_t > );
        auto  D  = blas::prod( R->U(), blas::adjoint( R->V() ) );
        auto  DM = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is(), std::move( D ) );
        
        return DM;
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  R  = cptrcast( &M, lrsvmatrix< value_t > );
        auto  US = blas::prod_diag( R->U(), R->S() );
        auto  D  = blas::prod( US, blas::adjoint( R->V() ) );
        auto  DM = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is(), std::move( D ) );
        
        return DM;
    }// if
    else if ( matrix::is_lowrankS( M ) )
    {
        auto  R  = cptrcast( &M, matrix::lrsmatrix< value_t > );
        auto  US = blas::prod( R->U(), R->S() );
        auto  D  = blas::prod( US, blas::adjoint( R->V() ) );
        auto  DM = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is(), std::move( D ) );
        
        return DM;
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R   = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  UxS = blas::prod( R->row_cb().basis(), R->coupling() );
        auto  D   = blas::prod( UxS, blas::adjoint( R->col_cb().basis() ) );
        auto  DM  = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is(), std::move( D ) );
        
        return DM;
    }// if
    else if ( matrix::is_uniform_lowrank2( M ) )
    {
        auto  R   = cptrcast( &M, uniform_lr2matrix< value_t > );
        auto  Ur  = blas::prod( R->row_basis(), R->row_coupling() );
        auto  Vc  = blas::prod( R->col_basis(), R->col_coupling() );
        auto  D   = blas::prod( Ur, blas::adjoint( Vc ) );
        auto  DM  = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is(), std::move( D ) );
        
        return DM;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        return std::unique_ptr< dense_matrix< value_t > >( ptrcast( M.copy().release(), dense_matrix< value_t > ) );
    }// if
    else if ( is_sparse( M ) )
    {
        auto  S  = cptrcast( &M, Hpro::TSparseMatrix< value_t > );
        auto  D  = std::make_unique< dense_matrix< value_t > >( M.row_is(), M.col_is() );
        auto  DD = D->mat();

        const auto  nrows = S->nrows();

        for ( size_t  i = 0; i < nrows; ++i )
        {
            auto  lb = S->rowptr(i);
            auto  ub = S->rowptr(i+1);

            for ( auto  l = lb; l < ub; ++l )
                DD( i, S->colind(l) ) = Hpro::coeff< value_t >( S, l );
        }// for
        
        return D;
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// return compressible version of M by replacing blocks
// (reusing data)
//
template < typename value_t >
Hpro::TMatrix< value_t > *
convert_to_compressible ( Hpro::TMatrix< value_t > *  M )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( M, Hpro::TBlockMatrix< value_t > );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( ! is_null( B_ij ) )
                {
                    auto  N_ij = convert_to_compressible( B_ij );

                    if ( B_ij != N_ij )
                    {
                        N_ij->set_parent( B );
                        B->set_block( i, j, N_ij );
                        delete B_ij;
                    }// if
                }// if
            }// for
        }// for

        return B;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        return  M;
    }// if
    else if ( Hpro::is_lowrank( M ) )
    {
        auto  R = ptrcast( M, Hpro::TRkMatrix< value_t > );
        auto  N = std::make_unique< matrix::lrmatrix< value_t > >( R->row_is(), R->col_is(),
                                                                   std::move( blas::mat_U( R ) ),
                                                                   std::move( blas::mat_V( R ) ) );
            
        N->set_id( M->id() );
        
        return N.release();
    }// if
    else if ( matrix::is_dense( M ) )
    {
        return  M;
    }// if
    else if ( Hpro::is_dense( M ) )
    {
        auto  D = ptrcast( M, Hpro::TDenseMatrix< value_t > );
        auto  N = std::make_unique< matrix::dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( blas::mat( D ) ) );
        
        N->set_id( M->id() );
        
        return N.release();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M->typestr() );

    return 0;
}

//
// convert given matrix into H-matrix, i.e., with standard lowrank and
// dense leaf blocks
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
convert_to_h ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = convert_to_h( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  U  = blas::prod( RM->row_basis(), RM->coupling() );
        auto  V  = RM->col_basis();
        auto  R  = std::make_unique< lrmatrix< value_t > >( RM->row_is(), RM->col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else if ( is_h2_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, h2_lrmatrix< value_t > );
        auto  U  = RM->row_cb().transform_backward( RM->coupling() );
        auto  I  = blas::identity< value_t >( RM->col_rank() );
        auto  V  = RM->col_cb().transform_backward( I );
        auto  R  = std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else if ( is_lowrank_sv( M ) )
    {
        auto  RM = cptrcast( &M, lrsvmatrix< value_t > );
        auto  U  = blas::prod_diag( RM->U(), RM->S() );
        auto  V  = RM->V();
        auto  R  = std::make_unique< lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else
    {
        return M.copy();
    }// if
}

//
// convert given matrix into HLIBpro matrix types, e.g., TRkMatrix and TDenseMatrix
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
convert_to_hpro ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = convert_to_hpro( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, matrix::lrmatrix< value_t > );
        auto  U  = blas::copy( RM->U() );
        auto  V  = blas::copy( RM->V() );
        auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( RM->row_is(), RM->col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  RM = cptrcast( &M, matrix::lrsvmatrix< value_t > );
        auto  U  = blas::copy( RM->U() );
        auto  S  = RM->S();
        auto  V  = blas::copy( RM->V() );

        blas::prod_diag( U, S );
        
        auto  R  = std::make_unique< Hpro::TRkMatrix< value_t > >( RM->row_is(), RM->col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  DM = cptrcast( &M, matrix::dense_matrix< value_t > );
        auto  DD = blas::copy( DM->mat() );
        auto  D  = std::make_unique< Hpro::TDenseMatrix< value_t > >( M.row_is(), M.col_is(), std::move( DD ) );

        D->set_id( M.id() );

        return D;
    }// if
    else
    {
        HLR_ERROR( "unsupported matrix type: " + M.typestr() );
    }// if
}

//
// convert given matrix into H matrix with off-diagonal lowrank blocks
// (very similar to matrix::coarsen but without storage size check)
//
template < typename                    value_t,
           approx::approximation_type  approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
convert_to_hodlr ( const Hpro::TMatrix< value_t > &  M,
                   const accuracy &                  acc,
                   const approx_t &                  approx )
{
    if ( is_blocked( M ) )
    {
        auto  BM          = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N           = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B           = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );
        bool  all_lowrank = true;
        uint  k_sum       = 0;

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( BM->block( i, j ) ) )
                {
                    auto  B_ij = convert_to_hodlr( * BM->block( i, j ), acc, approx );
                    
                    if ( matrix::is_lowrank( *B_ij ) )
                        k_sum += cptrcast( B_ij.get(), matrix::lrmatrix< value_t > )->rank();
                    else
                        all_lowrank = false;
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // convert off-diagonal block
        if ( M.row_is() != M.col_is() )
        {
            HLR_ASSERT( all_lowrank );
            
            auto  U_sum = blas::matrix< value_t >( M.nrows(), k_sum );
            auto  V_sum = blas::matrix< value_t >( M.ncols(), k_sum );
            uint  pos   = 0;
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  R_ij = cptrcast( B->block( i, j ), matrix::lrmatrix< value_t > );

                    if ( is_null( R_ij ) )
                        continue;

                    auto  RU   = R_ij->U();
                    auto  RV   = R_ij->V();
                    auto  U_i  = blas::matrix< value_t >( U_sum, R_ij->row_is() - M.row_ofs(), blas::range( pos, pos + R_ij->rank() - 1 ) );
                    auto  V_j  = blas::matrix< value_t >( V_sum, R_ij->col_is() - M.col_ofs(), blas::range( pos, pos + R_ij->rank() - 1 ) );

                    blas::copy( RU, U_i );
                    blas::copy( RV, V_j );
                    pos += R_ij->rank();
                }// for
            }// for

            auto  [ U, V ] = approx( U_sum, V_sum, acc );

            return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
        }// if
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        return M.copy();
    }// if
    else if ( matrix::is_dense( M ) )
    {
        // convert off-diagonal block
        if ( M.row_is() != M.col_is() )
            return convert_to_lowrank( M, acc, approx );
        else
            return M.copy();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_CONVERT_HH
