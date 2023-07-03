#ifndef __HLR_ARITH_DETAIL_SOLVE_VEC_HH
#define __HLR_ARITH_DETAIL_SOLVE_VEC_HH
//
// Project     : HLR
// Module      : solve.hh
// Description : vector solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/arith/mulvec.hh>
#include <hlr/matrix/dense_matrix.hh>

namespace hlr
{

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with lower triangular matrix L
//
// solve op(L) x = v 
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
void
solve_lower_tri ( const Hpro::matop_t               op_L,
                  const Hpro::TMatrix< value_t > &  L,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_lower_tri( %d )", L.id() ) );
        
    if ( is_blocked( L ) )
    {
        auto        BL  = cptrcast( & L, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BL->nblock_rows();
        const auto  nbc = BL->nblock_cols();
            
        if ( op_L == Hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->col_is() );
                
                    solve_lower_tri( op_L, *L_ii, v_i, diag_mode );
                }// if
            
                for ( uint  j = i+1; j < nbr; ++j )
                {
                    auto  L_ji = BL->block( j, i );

                    if ( ! is_null( L_ji ) )
                    {
                        auto  v_j = v.sub_vector( L_ji->row_is() );
                        auto  v_i = v.sub_vector( L_ji->col_is() );
                    
                        mul_vec< value_t >( value_t(-1), op_L, *L_ji, v_i, v_j );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block L_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min( nbr, nbc )-1; i >= 0; --i )
            {
                auto  L_ii = BL->block( i, i );
            
                if ( ! is_null( L_ii ) )
                {
                    auto  v_i = v.sub_vector( L_ii->row_is() );
                    
                    solve_lower_tri( op_L, *L_ii, v_i, diag_mode );
                }// if

                for ( int  j = i-1; j >= 0; --j )
                {
                    auto  L_ij = BL->block( i, j );
                    
                    if ( ! is_null( L_ij ) )
                    {
                        auto  v_i = v.sub_vector( L_ij->col_is() );
                        auto  v_j = v.sub_vector( L_ij->row_is() );
                    
                        mul_vec( value_t(-1), op_L, *L_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( matrix::is_dense( L ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming L contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_L, L, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for L : " + L.typestr() );
}

////////////////////////////////////////////////////////////////////////////////
//
// vector solving with upper triangular matrix U
//
// solve op(U) x = v
// - on exit, v contains x
//
////////////////////////////////////////////////////////////////////////////////
    
template < typename value_t >
void
solve_upper_tri ( const Hpro::matop_t               op_U,
                  const Hpro::TMatrix< value_t > &  U,
                  Hpro::TScalarVector< value_t > &  v,
                  const Hpro::diag_type_t           diag_mode )
{
    HLR_LOG( 4, Hpro::to_string( "solve_upper_tri( %d )", U.id() ) );
        
    if ( is_blocked( U ) )
    {
        auto        BU  = cptrcast( & U, Hpro::TBlockMatrix< value_t > );
        const auto  nbr = BU->nblock_rows();
        const auto  nbc = BU->nblock_cols();
            
        if ( op_U == Hpro::apply_normal )
        {
            //
            // solve from top to bottom
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( int  i = std::min<int>( nbr, nbc )-1; i >= 0; --i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->col_is() );
                
                    solve_upper_tri( op_U, *U_ii, v_i, diag_mode );
                }// if
            
                for ( int j = i-1; j >= 0; --j )
                {
                    auto  U_ji = BU->block( j, i );

                    if ( ! is_null( U_ji ) )
                    {
                        auto  v_j = v.sub_vector( U_ji->row_is() );
                        auto  v_i = v.sub_vector( U_ji->col_is() );
                    
                        mul_vec( value_t(-1), op_U, *U_ji, v_i, v_j );
                    }// if
                }// for
            }// for
        }// if
        else
        {
            //
            // solve bottom to top
            // - first diagonal block U_ii
            // - then update RHS with currently solved vector block v_i
            //
        
            for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
            {
                auto  U_ii = BU->block( i, i );
            
                if ( ! is_null( U_ii ) )
                {
                    auto  v_i = v.sub_vector( U_ii->row_is() );
                
                    solve_upper_tri( op_U, *U_ii, v_i, diag_mode );
                }// if

                for ( uint  j = i+1; j < nbc; ++j )
                {
                    auto  U_ij = BU->block( i, j );
                    
                    if ( ! is_null( U_ij ) )
                    {
                        auto  v_i = v.sub_vector( U_ij->col_is() );
                        auto  v_j = v.sub_vector( U_ij->row_is() );
                    
                        mul_vec( value_t(-1), op_U, *U_ij, v_j, v_i );
                    }// if
                }// for
            }// for
        }// else
    }// if
    else if ( matrix::is_dense( U ) )
    {
        if ( diag_mode == Hpro::general_diag )
        {
            //
            // assuming U contains inverse (store_inverse!)
            //

            auto  vc = Hpro::TScalarVector< value_t >( v );

            v.scale( 0 );
            mul_vec( value_t(1), op_U, U, vc, v );
        }// if
    }// if
    else
        HLR_ERROR( "unsupported matrix type for U : " + U.typestr() );
}

}// namespace hlr

#endif // __HLR_ARITH_DETAIL_SOLVE_VEC_HH
