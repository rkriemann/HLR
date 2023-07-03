#ifndef __HLR_ARITH_ADD_HH
#define __HLR_ARITH_ADD_HH
//
// Project     : HLR
// Module      : add.hh
// Description : matrix summation functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
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
      const Hpro::TMatrix< value_t > &  aA,
      Hpro::TMatrix< value_t > &        C,
      const Hpro::TTruncAcc &           acc,
      const approx_t &                  approx )
{
    using namespace hlr::matrix;
    
    using  matrix::is_lowrankS;
    using  matrix::lrsmatrix;

    // if ( compress::is_compressible( C ) )
    // {
    //     auto  lock = std::scoped_lock( C.mutex() );
        
    //     dynamic_cast< compressible * >( &C )->decompress();
    // }// if
        
    if ( is_blocked( aA ) )
    {
        auto  A = cptrcast( &aA, Hpro::TBlockMatrix< value_t > );
        
        if      ( is_blocked( C ) ) add< value_t, approx_t >( alpha, *A, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( C ) ) add< value_t, approx_t >( alpha, *A, *ptrcast( &C, lrmatrix< value_t > ), acc, approx );
        else if ( is_dense(   C ) ) add< value_t >( alpha, *A, *ptrcast( &C, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else if ( is_dense( aA ) )
    {
        auto  A = cptrcast( &aA, dense_matrix< value_t > );
        
        if      ( is_blocked( C ) ) add< value_t, approx_t >( alpha, *A, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( C ) ) add< value_t, approx_t >( alpha, *A, *ptrcast( &C, lrmatrix< value_t > ), acc, approx );
        else if ( is_dense(   C ) ) add< value_t >( alpha, *A, *ptrcast( &C, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else if ( is_lowrank( aA ) )
    {
        auto  A = cptrcast( &aA, lrmatrix< value_t > );
        
        if      ( is_blocked( C ) ) add< value_t, approx_t >( alpha, *A, *ptrcast( &C, Hpro::TBlockMatrix< value_t > ), acc, approx );
        else if ( is_lowrank( C ) ) add< value_t, approx_t >( alpha, *A, *ptrcast( &C, lrmatrix< value_t > ), acc, approx );
        else if ( is_dense(   C ) ) add< value_t >( alpha, *A, *ptrcast( &C, dense_matrix< value_t > ), acc );
        else
            HLR_ERROR( "unsupported matrix type : " + C.typestr() );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + aA.typestr() );

    // if ( compress::is_compressible( C ) )
    // {
    //     auto  lock = std::scoped_lock( C.mutex() );
        
    //     dynamic_cast< compressible * >( &C )->compress( acc );
    // }// if
}

//
// general version without approximation
//
template < typename value_t >
void
add ( const value_t                     alpha,
      const Hpro::TMatrix< value_t > &  A,
      Hpro::TMatrix< value_t > &        aC )
{
    using namespace hlr::matrix;
    
    HLR_ASSERT( is_dense( aC ) );

    if ( compress::is_compressible( aC ) && compress::is_compressed( aC ) )
        HLR_ERROR( "TODO" );

    auto  C = ptrcast( &aC, Hpro::TDenseMatrix< value_t > );
    
    if ( compress::is_compressible( A ) )
    {
        if      ( is_dense(   A ) ) add< value_t >( alpha, *cptrcast( &A, dense_matrix< value_t > ), *C );
        else if ( is_lowrank( A ) ) add< value_t >( alpha, *cptrcast( &A, lrmatrix< value_t > ),     *C );
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// if
    else
    {
        if      ( is_blocked( A ) ) add< value_t >( alpha, *cptrcast( &A, Hpro::TBlockMatrix< value_t > ), *C );
        else if ( is_dense(   A ) ) add< value_t >( alpha, *cptrcast( &A, Hpro::TDenseMatrix< value_t > ), *C );
        else if ( is_lowrank( A ) ) add< value_t >( alpha, *cptrcast( &A, Hpro::TRkMatrix< value_t > ),    *C );
        else
            HLR_ERROR( "unsupported matrix type : " + A.typestr() );
    }// else
}

}// namespace hlr

#endif // __HLR_ARITH_ADD_HH
