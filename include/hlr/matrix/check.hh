#ifndef __HLR_MATRIX_CHECK_HH
#define __HLR_MATRIX_CHECK_HH
//
// Project     : HLR
// Module      : matrix/check
// Description : test functions for matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/config.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>

namespace hlr { namespace matrix {

//
// test matrix data for consistency
//
template < typename value_t >
void
check ( const Hpro::TMatrix< value_t > &  M )
{
    //
    // test local data
    //

    // ensure ID is set
    if ( M.id() == -1 )
    {
        HLR_ERROR( "unset ID" );
    }// if
    else
    {
        // and smaller than parent ID
        if ( ! is_null( M.parent() ) )
        {
            if ( M.id() >= M.parent()->id() )
                HLR_ERROR( "parent ID <= ID" );
        }// if
    }// else

    //
    // recurse
    //

    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
                if ( ! is_null( B->block( i, j ) ) )
                    check( * B->block( i, j ) );
    }// if
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_CHECK_HH
