#ifndef __HLR_SEQ_CONVERT_HH
#define __HLR_SEQ_CONVERT_HH
//
// Project     : HLib
// Module      : matrix/convert
// Description : matrix conversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
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
        auto    B = ptrcast( &M, Hpro::TBlockMatrix< T_value_src > );
        size_t  s = sizeof(Hpro::TBlockMatrix< T_value_src >);

        s += B->nblock_rows() * B->nblock_cols() * sizeof(Hpro::TMatrix< T_value_src > *);
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    s += convert_prec< T_value_dest, T_value_src >( * B->block( i, j ) );
            }// for
        }// for

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

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_CONVERT_HH
