//
// Project     : HLib
// File        : tlr-tbb.cc
// Description : TLR arithmetic with TBB
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

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

namespace TBB
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
            
            TLR::TBB::lu< value_t >( A_ii, acc );

            tbb::parallel_for( i+1, nbc,
                               [A_ii,BA,i] ( uint  j )
                               {
                                   // L is unit diagonal !!!
                                   // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_rmat_A() );
                                   trsmuh< value_t >( A_ii, BA->block( j, i ) ); // A10->blas_rmat_B() );
                               } );

            tbb::parallel_for( tbb::blocked_range2d< uint >( i+1, nbr,
                                                             i+1, nbc ),
                               [BA,i,&acc] ( const tbb::blocked_range2d< uint > & r )
                               {
                                   for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
                                   {
                                       for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
                                       {
                                           update< value_t >( BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
                                       }// for
                                   }// for
                               } );
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

}// namespace TBB

}// namespace TLR
