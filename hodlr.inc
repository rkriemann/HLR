//
// Project     : HLib
// File        : hodlr-seq.cc
// Description : sequential HODLR arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

namespace HODLR
{

template < typename value_t >
void
trsml ( const TMatrix *         L,
        B::Matrix< value_t > &  X )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "trsml( %d )", L->id() );
    
    if ( is_blocked( L ) )
    {
        auto  BL  = cptrcast( L, TBlockMatrix );
        auto  L00 = BL->block( 0, 0 );
        auto  L10 = cptrcast( BL->block( 1, 0 ), TRkMatrix );
        auto  L11 = BL->block( 1, 1 );

        B::Matrix< value_t >  X0( X, L00->row_is() - L->row_ofs(), B::Range::all );
        B::Matrix< value_t >  X1( X, L11->row_is() - L->row_ofs(), B::Range::all );
            
        trsml( L00, X0 );

        auto  T = B::prod( value_t(1), B::adjoint( blas_mat_B< value_t >( L10 ) ), X0 );
        
        B::prod( value_t(-1), blas_mat_A< value_t >( L10 ), T, value_t(1), X1 );

        trsml( L11, X1 );
    }// if
    else
    {
        //
        // UNIT DIAGONAL !!!
        //
        
        // auto  DL = cptrcast( L, TDenseMatrix );
        
        // B::Matrix< value_t >  Y( X, copy_value );

        // B::prod( value_t(1), blas_mat< value_t >( DL ), Y, value_t(0), X );
    }// else
}

template < typename value_t >
void
trsmuh ( const TMatrix *         U,
         B::Matrix< value_t > &  X )
{
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "trsmuh( %d )", U->id() );
    
    if ( is_blocked( U ) )
    {
        auto  BU  = cptrcast( U, TBlockMatrix );
        auto  U00 = BU->block( 0, 0 );
        auto  U01 = cptrcast( BU->block( 0, 1 ), TRkMatrix );
        auto  U11 = BU->block( 1, 1 );

        B::Matrix< value_t >  X0( X, U00->col_is() - U->col_ofs(), B::Range::all );
        B::Matrix< value_t >  X1( X, U11->col_is() - U->col_ofs(), B::Range::all );
            
        trsmuh( U00, X0 );

        auto  T = B::prod( value_t(1), B::adjoint( blas_mat_A< value_t >( U01 ) ), X0 );
        
        B::prod( value_t(-1), blas_mat_B< value_t >( U01 ), T, value_t(1), X1 );

        trsmuh( U11, X1 );
    }// if
    else
    {
        auto  DU = cptrcast( U, TDenseMatrix );
        
        B::Matrix< value_t >  Y( X, copy_value );

        B::prod( value_t(1), B::adjoint( blas_mat< value_t >( DU ) ), Y, value_t(0), X );
    }// else
}

}// namespace HODLR

// Local Variables:
// mode: c++
// End:
