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
                auto  A_ji = BA->block( j, i );
                
                for ( uint  l = i+1; l < nbc; ++l )
                {
                    auto  A_il = BA->block( i, l );
                    auto  A_jl = BA->block( j, l );

                    update< value_t >( A_ji, A_il, A_jl, acc );

                    // if ( is_lowrank( A_jl ) )
                    //     if ( is_lowrank( A_il ) )
                    //         if ( is_lowrank( A_ji ) ) update< value_t >( cptrcast( A_ji, TRkMatrix    ), cptrcast( A_il, TRkMatrix    ), ptrcast(  A_jl, TRkMatrix    ), acc );
                    //         else                      update< value_t >( cptrcast( A_ji, TDenseMatrix ), cptrcast( A_il, TRkMatrix    ), ptrcast(  A_jl, TRkMatrix    ), acc );
                    //     else
                    //         if ( is_lowrank( A_ji ) ) update< value_t >( cptrcast( A_ji, TRkMatrix    ), cptrcast( A_il, TDenseMatrix ), ptrcast(  A_jl, TRkMatrix    ), acc );
                    //         else                      update< value_t >( cptrcast( A_ji, TDenseMatrix ), cptrcast( A_il, TDenseMatrix ), ptrcast(  A_jl, TRkMatrix    ), acc );
                    // else
                    //     if ( is_lowrank( A_il ) )
                    //         if ( is_lowrank( A_ji ) ) update< value_t >( cptrcast( A_ji, TRkMatrix    ), cptrcast( A_il, TRkMatrix    ), ptrcast(  A_jl, TDenseMatrix ) );
                    //         else                      update< value_t >( cptrcast( A_ji, TDenseMatrix ), cptrcast( A_il, TRkMatrix    ), ptrcast(  A_jl, TDenseMatrix ) );
                    //     else
                    //         if ( is_lowrank( A_ji ) ) update< value_t >( cptrcast( A_ji, TRkMatrix    ), cptrcast( A_il, TDenseMatrix ), ptrcast(  A_jl, TDenseMatrix ) );
                    //         else                      update< value_t >( cptrcast( A_ji, TDenseMatrix ), cptrcast( A_il, TDenseMatrix ), ptrcast(  A_jl, TDenseMatrix ) );
                    
                    // if ( j == l )
                    //     update( cptrcast( BA->block( j, i ), TRkMatrix ),
                    //             cptrcast( BA->block( i, l ), TRkMatrix ),
                    //             ptrcast(  BA->block( j, l ), TDenseMatrix ) );
                    // else
                    //     update( cptrcast( BA->block( j, i ), TRkMatrix ),
                    //             cptrcast( BA->block( i, l ), TRkMatrix ),
                    //             ptrcast(  BA->block( j, l ), TRkMatrix ),
                    //             acc );
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
