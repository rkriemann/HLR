#ifndef __HLR_ARITH_ADD_HH
#define __HLR_ARITH_ADD_HH
//
// Project     : HLib
// File        : add.hh
// Description : matrix summation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include "hlr/arith/detail/add.hh"
#include "hlr/matrix/lrsmatrix.hh"
#include "hlr/utils/log.hh"

namespace hlr
{

//
// compute C := C + α A with different types of A/C
//
template < typename value_t,
           typename approx_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        C,
      const Hpro::TTruncAcc &           acc,
      const approx_t &                  approx )
{
    using namespace hlr::matrix;
    
    using  matrix::is_lowrankS;
    using  matrix::lrsmatrix;

    if ( is_compressible( C ) )
        dynamic_cast< compressible * >( &C )->decompress();
        
    if ( is_blocked( A ) )
    {
        if      ( is_blocked(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TBlockMatrix< value_t > ), *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TBlockMatrix< value_t > ), *ptrcast( &C, Hpro::TRkMatrix< value_t > ),    acc, approx );
        else if ( is_dense(    C ) ) add< value_t >(           alpha, *cptrcast( &A, Hpro::TBlockMatrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else if ( is_compressible( A ) )
    {
        if ( is_dense( A ) )
        {
            if      ( is_blocked(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, dense_matrix< value_t > ), *ptrcast( &C, Hpro::TBlockMatrix< value_t > ),   acc, approx );
            else if ( is_lowrank(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, dense_matrix< value_t > ), *ptrcast( &C, Hpro::TRkMatrix< value_t > ),      acc, approx );
            else if ( is_dense(    C ) ) add< value_t >(           alpha, *cptrcast( &A, dense_matrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_lowrank( A ) )
        {
            if      ( is_blocked(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, lrmatrix< value_t > ), *ptrcast( &C, Hpro::TBlockMatrix< value_t > ),   acc, approx );
            else if ( is_lowrank(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, lrmatrix< value_t > ), *ptrcast( &C, Hpro::TRkMatrix< value_t > ),      acc, approx );
            else if ( is_dense(    C ) ) add< value_t >(           alpha, *cptrcast( &A, lrmatrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// if
    else
    {
        if ( is_dense( A ) )
        {
            if      ( is_blocked(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TDenseMatrix< value_t > ), *ptrcast( &C, Hpro::TBlockMatrix< value_t > ),   acc, approx );
            else if ( is_lowrank(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TDenseMatrix< value_t > ), *ptrcast( &C, Hpro::TRkMatrix< value_t > ),      acc, approx );
            else if ( is_lowrankS( C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TDenseMatrix< value_t > ), *ptrcast( &C, lrsmatrix< value_t > ), acc, approx );
            else if ( is_dense(    C ) ) add< value_t >(           alpha, *cptrcast( &A, Hpro::TDenseMatrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else if ( is_lowrank( A ) )
        {
            if      ( is_blocked(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TRkMatrix< value_t > ), *ptrcast( &C, Hpro::TBlockMatrix< value_t > ),   acc, approx );
            else if ( is_lowrank(  C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TRkMatrix< value_t > ), *ptrcast( &C, Hpro::TRkMatrix< value_t > ),      acc, approx );
            else if ( is_lowrankS( C ) ) add< value_t, approx_t >( alpha, *cptrcast( &A, Hpro::TRkMatrix< value_t > ), *ptrcast( &C, lrsmatrix< value_t > ), acc, approx );
            else if ( is_dense(    C ) ) add< value_t >(           alpha, *cptrcast( &A, Hpro::TRkMatrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
            else
                HLR_ERROR( "unsupported matrix type : " + C.typestr() );
        }// if
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// else

    if ( is_compressible( C ) )
        dynamic_cast< compressible * >( &C )->compress( acc );
}

//
// general version without approximation
//
template < typename value_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        C )
{
    using namespace hlr::matrix;
    
    HLR_ASSERT( is_dense( C ) );

    if ( is_compressible( C ) )
        HLR_ERROR( "TODO" );

    if ( is_compressible( A ) )
    {
        if      ( is_dense(   A ) ) add< value_t >( alpha, *cptrcast( &A, dense_matrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        else if ( is_lowrank( A ) ) add< value_t >( alpha, *cptrcast( &A, lrmatrix< value_t > ),     *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// if
    else
    {
        if      ( is_blocked( A ) ) add< value_t >( alpha, *cptrcast( &A, Hpro::TBlockMatrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        else if ( is_dense(   A ) ) add< value_t >( alpha, *cptrcast( &A, Hpro::TDenseMatrix< value_t > ), *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        else if ( is_lowrank( A ) ) add< value_t >( alpha, *cptrcast( &A, Hpro::TRkMatrix< value_t > ),    *ptrcast( &C, Hpro::TDenseMatrix< value_t > ) );
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_ADD_HH
