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
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/compression.hh>

#include <hlr/seq/convert.hh>

namespace hlr { namespace tbb { namespace matrix {

using namespace hlr::matrix;

//
// convert given matrix into dense format
//
template < typename value_t >
std::unique_ptr< matrix::dense_matrix< value_t > >
convert_to_dense ( const Hpro::TMatrix< value_t > &  M )
{
    return hlr::matrix::convert_to_dense< value_t >( M );
}

//
// convert given matrix into lowrank format
//
template < typename value_t,
           typename approx_t >
std::unique_ptr< matrix::lrmatrix< value_t > >
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

        return std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else 
    {
        return hlr::matrix::convert_to_lowrank( M, acc, approx );
    }// if
}

//
// convert given matrix into lowrank format without truncation
// (only implemented for lowrank compatible formats)
//
template < typename value_t >
std::unique_ptr< matrix::lrmatrix< value_t > >
convert_to_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    return hlr::matrix::convert_to_lowrank( M );
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
    else
    {
        return hlr::seq::matrix::convert< dest_value_t, src_value_t >( A );
    }// else
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
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [BM,&B] ( auto  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = convert_to_h( * BM->block( i, j ) );
                    
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        auto  U  = blas::prod( RM->row_basis(), RM->coupling() );
        auto  V  = RM->col_basis();
        auto  R  = std::make_unique< matrix::lrmatrix< value_t > >( RM->row_is(), RM->col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  RM = cptrcast( &M, matrix::h2_lrmatrix< value_t > );
        auto  U  = RM->row_cb().transform_backward( RM->coupling() );
        auto  I  = blas::identity< value_t >( RM->col_rank() );
        auto  V  = RM->col_cb().transform_backward( I );
        auto  R  = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

        R->set_id( M.id() );

        return R;
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  RM = cptrcast( &M, matrix::lrsvmatrix< value_t > );
        auto  U  = blas::prod_diag( RM->U(), RM->S() );
        auto  V  = RM->V();
        auto  R  = std::make_unique< matrix::lrmatrix< value_t > >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );

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
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [BM,&B] ( auto  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        if ( BM->block( i, j ) != nullptr )
                        {
                            auto  B_ij = convert_to_hpro( * BM->block( i, j ) );
                    
                            B_ij->set_parent( B );
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );

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

}}}// namespace hlr::tbb::matrix

#endif // __HLR_TBB_CONVERT_HH
