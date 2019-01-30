//
// Project     : HLib
// File        : tlr-seq.cc
// Description : Implements sequential TLR arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlib.hh>

using namespace HLIB;

namespace B = HLIB::BLAS;

#include "tlr.inc"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace SEQ
{

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );
    
    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  nbr = BA->nblock_rows();
        auto  nbc = BA->nblock_cols();

        for ( uint  i = 0; i < nbr; ++i )
        {
            auto  A_ii = ptrcast( BA->block( i, i ), TDenseMatrix );
            
            TLR::SEQ::lu< value_t >( A_ii, acc );

            for ( uint  j = i+1; j < nbc; ++j )
            {
                // L is unit diagonal !!!
                // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
                trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
            }// for

            for ( uint  j = i+1; j < nbr; ++j )
            {
                for ( uint  l = i+1; l < nbc; ++l )
                {
                    update< value_t >( BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
                }// for
            }// for
        }// for
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( DA->blas_rmat() );
    }// else
}

template
void
lu< HLIB::real > ( TMatrix *          A,
                   const TTruncAcc &  acc );

}// namespace SEQ

}// namespace TLR
