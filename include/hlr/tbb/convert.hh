#ifndef __HLR_TBB_CONVERT_HH
#define __HLR_TBB_CONVERT_HH
//
// Project     : HLR
// Module      : convert
// Description : matrix conversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <hpro/config.h>

#include <hlr/matrix/convert.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/compression.hh>

namespace hlr { namespace tbb { namespace matrix {

using namespace hlr::matrix;

//
// convert given matrix into dense format
//
template < typename value_t >
std::unique_ptr< Hpro::TDenseMatrix< value_t > >
convert_to_dense ( const Hpro::TMatrix< value_t > &  M )
{
    return hlr::matrix::convert_to_dense< value_t >( M );
}

//
// convert given matrix into lowrank format
//
template < typename value_t,
           typename approx_t >
std::unique_ptr< Hpro::TRkMatrix< value_t > >
convert_to_lowrank ( const Hpro::TMatrix< value_t > &    M,
                     const Hpro::TTruncAcc &  acc,
                     const approx_t &         approx )
{
    if ( is_blocked( M ) )
    {
        //
        // convert each sub block into low-rank format and 
        // enlarge to size of M (pad with zeroes)
        //

        auto        B  = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto        Us = std::list< blas::matrix< value_t > >();
        auto        Vs = std::list< blas::matrix< value_t > >();
        std::mutex  mtx;

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  B_ij = B->block( i, j );
                
                        if ( is_null( B_ij ) )
                            continue;

                        auto  R_ij = convert_to_lowrank( *B_ij, acc, approx );
                        auto  U    = blas::matrix< value_t >( M.nrows(), R_ij->rank() );
                        auto  V    = blas::matrix< value_t >( M.ncols(), R_ij->rank() );
                        auto  U_i  = blas::matrix< value_t >( U, R_ij->row_is() - M.row_ofs(), blas::range::all );
                        auto  V_j  = blas::matrix< value_t >( V, R_ij->col_is() - M.col_ofs(), blas::range::all );

                        blas::copy( Hpro::blas_mat_A< value_t >( R_ij ), U_i );
                        blas::copy( Hpro::blas_mat_B< value_t >( R_ij ), V_j );

                        std::scoped_lock  lock( mtx );
                            
                        Us.push_back( std::move( U ) );
                        Vs.push_back( std::move( V ) );
                    }// for
                }// for
            } );

        auto  [ U, V ] = approx( Us, Vs, acc );

        return std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_dense( M ) )
    {
        auto  D        = cptrcast( &M, Hpro::TDenseMatrix< value_t > );
        auto  T        = std::move( blas::copy( Hpro::blas_mat< value_t >( D ) ) );
        auto  [ U, V ] = approx( T, acc );

        return std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R        = cptrcast( &M, Hpro::TRkMatrix< value_t > );
        auto  [ U, V ] = approx( Hpro::blas_mat_A< value_t >( R ),
                                 Hpro::blas_mat_B< value_t >( R ),
                                 acc );
        
        return std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// convert given matrix into lowrank format without truncation
// (only implemented for lowrank compatible formats)
//
template < typename value_t >
std::unique_ptr< Hpro::TRkMatrix< value_t > >
convert_to_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, Hpro::TRkMatrix< value_t > );
        
        return std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(),
                                                    std::move( blas::copy( blas::mat_U< value_t >( R ) ) ),
                                                    std::move( blas::copy( blas::mat_V< value_t >( R ) ) ) );
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  R = cptrcast( &M, uniform_lrmatrix< value_t > );
        auto  U = blas::prod( R->row_basis(), R->coeff() );
        auto  V = blas::copy( R->col_basis() );
        
        return std::make_unique< Hpro::TRkMatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// convert matrix between different floating point precisions
// - return storage used with destination precision
//
template < typename T_value_dest,
           typename T_value_src >
size_t
convert_prec ( Hpro::TMatrix< T_value_src > &  M )
{
    if constexpr( std::is_same_v< T_value_dest, T_value_src > )
        return M.byte_size();
    
    if ( is_blocked( M ) )
    {
        auto    B   = ptrcast( &M, Hpro::TBlockMatrix< T_value_src > );
        size_t  s   = sizeof(Hpro::TBlockMatrix< T_value_src >);
        auto    mtx = std::mutex();

        s += B->nblock_rows() * B->nblock_cols() * sizeof(Hpro::TMatrix< T_value_dest > *);

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&s,&mtx,B] ( auto  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( B->block( i, j ) ) )
                        {
                            auto  s_ij = convert_prec< T_value_dest, T_value_src >( * B->block( i, j ) );

                            {
                                auto  lock = std::scoped_lock( mtx );

                                s += s_ij;
                            }
                        }// if
                    }// for
                }// for
            } );

        return s;
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = ptrcast( &M, Hpro::TRkMatrix< T_value_src > );
        auto  U = blas::copy< T_value_dest >( blas::mat_U< T_value_src >( R ) );
        auto  V = blas::copy< T_value_dest >( blas::mat_V< T_value_src >( R ) );

        blas::copy< T_value_dest, T_value_src >( U, blas::mat_U< T_value_src >( R ) );
        blas::copy< T_value_dest, T_value_src >( V, blas::mat_V< T_value_src >( R ) );

        return R->byte_size() - sizeof(T_value_src) * R->rank() * ( R->nrows() + R->ncols() ) + sizeof(T_value_dest) * R->rank() * ( R->nrows() + R->ncols() ); 
    }// if
    else if ( is_uniform_lowrank( M ) )
    {
        auto  U = ptrcast( &M, matrix::uniform_lrmatrix< T_value_src > );
        auto  S = blas::copy< T_value_dest >( U->coeff() );

        blas::copy< T_value_dest, T_value_src >( S, U->coeff() );

        return U->byte_size() - sizeof(T_value_src) * S.nrows() * S.ncols() + sizeof(T_value_dest) * S.nrows() * S.ncols(); 
    }// if
    else if ( is_dense( M ) )
    {
        auto  D  = ptrcast( &M, Hpro::TDenseMatrix< T_value_src > );
        auto  DD = blas::copy< T_value_dest >( blas::mat< T_value_src >( D ) );

        blas::copy< T_value_dest, T_value_src >( DD, blas::mat< T_value_src >( D ) );

        return D->byte_size() - sizeof(T_value_src) * D->nrows() * D->ncols() + sizeof(T_value_dest) * D->nrows() * D->ncols();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );

    return 0;
}

//
// return copy of matrix in given value type
//
template < typename dest_value_t,
           typename src_value_t >
std::unique_ptr< Hpro::TMatrix< dest_value_t > >
convert ( const Hpro::TMatrix< src_value_t > &  A )
{
    // if types are equal, just perform standard copy
    if constexpr ( std::is_same< dest_value_t, src_value_t >::value )
        return A.copy();

    // to copy basic properties
    auto  copy_struct = [] ( const auto &  A, auto &  B )
    {
        B.set_id( A.id() );
        B.set_form( A.form() );
        B.set_ofs( A.row_ofs(), A.col_ofs() );
        B.set_size( A.rows(), A.cols() );
        B.set_procs( A.procs() );
    };
    
    if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< src_value_t > );
        auto  BC = std::make_unique< Hpro::TBlockMatrix< dest_value_t > >( BA->row_is(), BA->col_is() );

        copy_struct( *BA, *BC );
        BC->set_block_struct( BA->nblock_rows(), BA->nblock_cols() );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, BA->nblock_rows(),
                                            0, BA->nblock_cols() ),
            [BA,&BC] ( auto  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( ! is_null( BA->block( i, j ) ) )
                        {
                            auto  BC_ij = convert< dest_value_t, src_value_t >( *BA->block( i, j ) );
                            
                            BC->set_block( i, j, BC_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        return BC;
    }// if
    else if ( is_lowrank( A ) )
    {
        HLR_ASSERT( ! is_compressible( A ) );
        
        auto  RA = cptrcast( &A, Hpro::TRkMatrix< src_value_t > );
        auto  U  = blas::convert< dest_value_t >( RA->blas_mat_A() );
        auto  V  = blas::convert< dest_value_t >( RA->blas_mat_B() );
        auto  RC = std::make_unique< Hpro::TRkMatrix< dest_value_t > >( RA->row_is(), RA->col_is(), std::move( U ), std::move( V ) );

        copy_struct( *RA, *RC );
        
        return RC;
    }// if
    else if ( is_dense( A ) )
    {
        HLR_ASSERT( ! is_compressible( A ) );
        
        auto  DA = cptrcast( &A, Hpro::TDenseMatrix< src_value_t > );
        auto  D  = blas::convert< dest_value_t >( DA->blas_mat() );
        auto  DC = std::make_unique< Hpro::TDenseMatrix< dest_value_t > >( DA->row_is(), DA->col_is(), std::move( D ) );

        copy_struct( *DA, *DC );
        
        return DC;
    }// if
    else
        HLR_ERROR( "unsupported matrix type " + A.typestr() );
}

}}}// namespace hlr::tbb::matrix

#endif // __HLR_TBB_CONVERT_HH
