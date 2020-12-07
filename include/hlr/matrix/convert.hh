#ifndef __HLR_MATRIX_CONVERT_HH
#define __HLR_MATRIX_CONVERT_HH
//
// Project     : HLib
// Module      : matrix/convert
// Description : matrix conversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>
#include <hpro/matrix/structure.hh>

#include <hlr/matrix/uniform_lrmatrix.hh>

namespace hlr { namespace matrix {

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

                blas::copy( blas::mat_U< value_t >( R_ij ), U_i );
                blas::copy( blas::mat_V< value_t >( R_ij ), V_j );

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
        auto  [ U, V ] = approx( blas::mat_U< value_t >( R ),
                                 blas::mat_V< value_t >( R ),
                                 acc );
        
        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  US       = blas::prod( R->row_cb().basis(), R->coeff() );
        auto  [ U, V ] = approx( US, R->col_cb().basis(), acc );
        
        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// convert given matrix into dense format
//
template < typename value_t >
std::unique_ptr< hpro::TDenseMatrix >
convert_to_dense ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block and copy to dense version
        //

        auto  B  = cptrcast( &M, hpro::TBlockMatrix );
        auto  D  = std::make_unique< hpro::TDenseMatrix >( M.row_is(), M.col_is(), hpro::value_type< value_t >::value );
        auto  DD = blas::mat< value_t >( *D );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
                
                if ( is_null( B_ij ) )
                    continue;

                auto  D_ij  = convert_to_dense< value_t >( *B_ij );
                auto  DD_ij = blas::mat< value_t >( *D_ij );
                auto  TD    = blas::matrix( DD,
                                            D_ij->row_is() - M.row_ofs(),
                                            D_ij->col_is() - M.col_ofs() );

                blas::copy( DD_ij, TD );
            }// for
        }// for

        return D;
    }// if
    else if ( is_dense( M ) )
    {
        auto  D   = cptrcast( &M, hpro::TDenseMatrix );
        auto  DD  = blas::mat< value_t >( *D );
        auto  DC  = std::make_unique< hpro::TDenseMatrix >( M.row_is(), M.col_is(), hpro::value_type< value_t >::value );
        auto  DDC = blas::mat< value_t >( *DC );

        blas::copy( DD, DDC );
            
        return DC;
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R  = cptrcast( &M, hpro::TRkMatrix );
        auto  D  = std::make_unique< hpro::TDenseMatrix >( M.row_is(), M.col_is(), hpro::value_type< value_t >::value );
        auto  DD = blas::mat< value_t >( *D );

        blas::prod( value_t(1), blas::mat_U< value_t >( R ), blas::adjoint( blas::mat_V< value_t >( R ) ),
                    value_t(0), DD );
        
        return D;
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R   = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  D   = std::make_unique< hpro::TDenseMatrix >( M.row_is(), M.col_is(), hpro::value_type< value_t >::value );
        auto  DD  = blas::mat< value_t >( *D );
        auto  UxS = blas::prod( value_t(1), R->row_cb().basis(), R->coeff() );

        blas::prod( value_t(1), UxS, blas::adjoint( R->col_cb().basis() ),
                    value_t(0), DD );
        
        return D;
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_CONVERT_HH