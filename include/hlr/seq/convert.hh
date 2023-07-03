#ifndef __HLR_SEQ_CONVERT_HH
#define __HLR_SEQ_CONVERT_HH
//
// Project     : HLR
// Module      : matrix/convert
// Description : matrix conversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/config.h>

#include <hlr/matrix/convert.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/compression.hh>

namespace hlr { namespace seq { namespace matrix {

using hlr::matrix::convert_to_lowrank;
using hlr::matrix::convert_to_dense;
using hlr::matrix::convert_to_h;
using hlr::matrix::convert_to_compressible;

// //
// // convert matrix between different floating point precisions
// // - return storage used with destination precision
// //
// template < typename T_value_dest,
//            typename T_value_src >
// size_t
// convert_prec ( Hpro::TMatrix< T_value_src > &  M )
// {
//     if constexpr( std::is_same_v< T_value_dest, T_value_src > )
//         return M.byte_size();
    
//     if ( is_blocked( M ) )
//     {
//         auto    B = ptrcast( &M, Hpro::TBlockMatrix< T_value_src > );
//         size_t  s = sizeof(Hpro::TBlockMatrix< T_value_src >);

//         s += B->nblock_rows() * B->nblock_cols() * sizeof(Hpro::TMatrix< T_value_src > *);
        
//         for ( uint  i = 0; i < B->nblock_rows(); ++i )
//         {
//             for ( uint  j = 0; j < B->nblock_cols(); ++j )
//             {
//                 if ( ! is_null( B->block( i, j ) ) )
//                     s += convert_prec< T_value_dest, T_value_src >( * B->block( i, j ) );
//             }// for
//         }// for

//         return s;
//     }// if
//     else if ( matrix::is_lowrank( M ) )
//     {
//         auto  R = ptrcast( &M, matrix::lrmatrix< T_value_src > );
//         auto  U = blas::copy< T_value_dest >( R->U() );
//         auto  V = blas::copy< T_value_dest >( R->V() );

//         blas::copy< T_value_dest, T_value_src >( U, blas::mat_U< T_value_src >( R ) );
//         blas::copy< T_value_dest, T_value_src >( V, blas::mat_V< T_value_src >( R ) );

//         return R->byte_size() - sizeof(T_value_src) * R->rank() * ( R->nrows() + R->ncols() ) + sizeof(T_value_dest) * R->rank() * ( R->nrows() + R->ncols() ); 
//     }// if
//     else if ( is_uniform_lowrank( M ) )
//     {
//         auto  U = ptrcast( &M, matrix::uniform_lrmatrix< T_value_src > );
//         auto  S = blas::copy< T_value_dest >( U->coeff() );

//         blas::copy< T_value_dest, T_value_src >( S, U->coeff() );

//         return U->byte_size() - sizeof(T_value_src) * S.nrows() * S.ncols() + sizeof(T_value_dest) * S.nrows() * S.ncols(); 
//     }// if
//     else if ( is_dense( M ) )
//     {
//         auto  D  = ptrcast( &M, Hpro::TDenseMatrix< T_value_src > );
//         auto  DD = blas::copy< T_value_dest >( blas::mat< T_value_src >( D ) );

//         blas::copy< T_value_dest, T_value_src >( DD, blas::mat< T_value_src >( D ) );

//         return D->byte_size() - sizeof(T_value_src) * D->nrows() * D->ncols() + sizeof(T_value_dest) * D->nrows() * D->ncols();
//     }// if
//     else
//         HLR_ERROR( "unsupported matrix type : " + M.typestr() );

//     return 0;
// }

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

        for ( uint  i = 0; i < BA->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->block_cols(); ++j )
            {
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    auto  BC_ij = convert< dest_value_t, src_value_t >( *BA->block( i, j ) );
                    
                    BC->set_block( i, j, BC_ij.release() );
                }// if
            }// for
        }// for

        return BC;
    }// if
    else if ( matrix::is_lowrank( A ) )
    {
        auto  RA = cptrcast( &A, matrix::lrmatrix< src_value_t > );
        auto  U  = blas::convert< dest_value_t >( RA->U() );
        auto  V  = blas::convert< dest_value_t >( RA->V() );
        auto  RC = std::make_unique< matrix::lrmatrix< dest_value_t > >( RA->row_is(), RA->col_is(), std::move( U ), std::move( V ) );

        copy_struct( *RA, *RC );
        
        return RC;
    }// if
    else if ( matrix::is_dense( A ) )
    {
        auto  DA = cptrcast( &A, matrix::dense_matrix< src_value_t > );
        auto  D  = blas::convert< dest_value_t >( DA->mat() );
        auto  DC = std::make_unique< matrix::dense_matrix< dest_value_t > >( DA->row_is(), DA->col_is(), std::move( D ) );

        copy_struct( *DA, *DC );
        
        return DC;
    }// if
    else
        HLR_ERROR( "unsupported matrix type " + A.typestr() );
}

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_CONVERT_HH
