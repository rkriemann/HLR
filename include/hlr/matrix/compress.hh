#ifndef __HLR_MATRIX_COMPRESS_HH
#define __HLR_MATRIX_COMPRESS_HH
//
// Project     : HLib
// Module      : matrix/compress
// Description : matrix compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2022. All Rights Reserved.
//

#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

namespace hlr { namespace matrix {

//
// apply compression to compressible objects
// (dense_matrix, lrmatrix)
//
template < typename value_t >
void
compress ( Hpro::TMatrix< value_t > &  M,
           const hpro::TTruncAcc &     acc )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block(i,j) ) )
                    compress( *B->block(i,j), acc );
            }// for
        }// for
    }// if
    else if ( is_compressible( M ) )
    {
        dynamic_cast< compressible * >( &M )->compress( acc );
    }// if
}

//
// decompress internal data in compressible objects
// (dense_matrix, lrmatrix)
//
template < typename value_t >
void
decompress ( Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block(i,j) ) )
                    decompress( *B->block(i,j) );
            }// for
        }// for
    }// if
    else if ( is_compressible( M ) )
    {
        dynamic_cast< compressible * >( &M )->decompress();
    }// if
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_COMPRESS_HH
