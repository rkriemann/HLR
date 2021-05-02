//
// Project     : HLib
// File        : solve.cc
// Description : matrix solve functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/vector/TScalarVector.hh>

#include "hlr/utils/checks.hh"
#include "hlr/seq/arith.hh"

namespace hlr { namespace seq {

using namespace HLIB;

// namespace
// {

// // return sub vector of v corresponding to is
// TScalarVector
// sub_vector ( TScalarVector &    v,
//              const TIndexSet &  is )
// {
//     return v.sub_vector( is );
// }

// }// namespace anonymous

// //
// // solve op(L) x = y with lower triangular L
// //
// void
// trsvl ( const HLIB::matop_t      op_L,
//         const HLIB::TMatrix &    L,
//         HLIB::TScalarVector &    v,
//         const HLIB::diag_type_t  diag_mode )
// {
//     HLR_LOG( 4, HLIB::to_string( "trsvl( %d )", L.id() ) );
        
//     if ( is_blocked( L ) )
//     {
//         auto        BL  = cptrcast( & L, TBlockMatrix );
//         const auto  nbr = BL->nblock_rows();
//         const auto  nbc = BL->nblock_cols();
            
//         if ( op_L == apply_normal )
//         {
//             //
//             // solve from top to bottom
//             //
        
//             for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
//             {
//                 //
//                 // solve diagonal block
//                 //
            
//                 auto  L_ii = BL->block( i, i );
            
//                 if ( ! is_null( L_ii ) )
//                 {
//                     TScalarVector  v_i( sub_vector( v, L_ii->col_is() ) );
                
//                     trsvl( op_L, *L_ii, v_i, diag_mode );
//                 }// if
            
//                 //
//                 // update RHS with currently solved vector block
//                 //

//                 for ( uint  j = i+1; j < nbr; ++j )
//                 {
//                     auto  L_ji = BL->block( j, i );

//                     if ( ! is_null( L_ji ) )
//                     {
//                         TScalarVector  v_j( sub_vector( v, L_ji->row_is() ) );
//                         TScalarVector  v_i( sub_vector( v, L_ji->col_is() ) );
                    
//                         HLR_LOG( 4, HLIB::to_string( "update( %d, ", L_ji->id() ) + v_i.is().to_string() + ", " + v_j.is().to_string() );
//                         mul_vec( real(-1), L_ji, & v_i, real(1), & v_j, op_L );
//                     }// if
//                 }// for
//             }// for
//         }// if
//         else
//         {
//             //
//             // solve bottom to top
//             //
        
//             for ( int  i = std::min( nbr, nbc )-1; i >= 0; --i )
//             {
//                 //
//                 // solve diagonal block
//                 //
            
//                 auto  L_ii = BL->block( i, i );
            
//                 if ( ! is_null( L_ii ) )
//                 {
//                     TScalarVector  v_i( sub_vector( v, L_ii->row_is() ) );
                
//                     trsvl( op_L, *L_ii, v_i, diag_mode );
//                 }// if

//                 //
//                 // update RHS
//                 //

//                 for ( int  j = i-1; j >= 0; --j )
//                 {
//                     auto  L_ij = BL->block( i, j );
                    
//                     if ( ! is_null( L_ij ) )
//                     {
//                         TScalarVector  v_i( sub_vector( v, L_ij->col_is() ) );
//                         TScalarVector  v_j( sub_vector( v, L_ij->row_is() ) );
                    
//                         HLR_LOG( 4, HLIB::to_string( "update( %d, ", L_ij->id() ) + v_j.is().to_string() + ", " + v_i.is().to_string() );
//                         mul_vec( real(-1), L_ij, & v_j, real(1), & v_i, op_L );
//                     }// if
//                 }// for
//             }// for
//         }// else
//     }// if
//     else if ( is_dense( & L ) )
//     {
//         if ( diag_mode == general_diag )
//         {
//             //
//             // assuming L contains inverse (store_inverse!)
//             //

//             TScalarVector  vc( v );

//             mul_vec( real(1), & L, & vc, real(0), & v, op_L );
//         }// if
//     }// if
//     else
//         assert( false );
// }

// //
// // solve op(U) x = y with upper triangular U
// //
// void
// trsvu ( const HLIB::matop_t      op_U,
//         const HLIB::TMatrix &    U,
//         HLIB::TScalarVector &    v,
//         const HLIB::diag_type_t  diag_mode )
// {
//     HLR_LOG( 4, HLIB::to_string( "trsvu( %d )", U.id() ) );
        
//     if ( is_blocked( U ) )
//     {
//         auto        BU  = cptrcast( & U, TBlockMatrix );
//         const auto  nbr = BU->nblock_rows();
//         const auto  nbc = BU->nblock_cols();
            
//         if ( op_U == apply_normal )
//         {
//             //
//             // solve from top to bottom
//             //
        
//             for ( int  i = std::min<int>( nbr, nbc )-1; i >= 0; --i )
//             {
//                 //
//                 // solve diagonal block
//                 //
            
//                 auto  U_ii = BU->block( i, i );
            
//                 if ( ! is_null( U_ii ) )
//                 {
//                     TScalarVector  v_i( sub_vector( v, U_ii->col_is() ) );
                
//                     trsvu( op_U, *U_ii, v_i, diag_mode );
//                 }// if
            
//                 //
//                 // update RHS with currently solved vector block
//                 //

//                 for ( int j = i-1; j >= 0; --j )
//                 {
//                     auto  U_ji = BU->block( j, i );

//                     if ( ! is_null( U_ji ) )
//                     {
//                         TScalarVector  v_j( sub_vector( v, U_ji->row_is() ) );
//                         TScalarVector  v_i( sub_vector( v, U_ji->col_is() ) );
                    
//                         HLR_LOG( 4, HLIB::to_string( "update( %d, ", U_ji->id() ) + v_i.is().to_string() + ", " + v_j.is().to_string() );
//                         mul_vec( real(-1), U_ji, & v_i, real(1), & v_j, op_U );
//                     }// if
//                 }// for
//             }// for
//         }// if
//         else
//         {
//             //
//             // solve bottom to top
//             //
        
//             for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
//             {
//                 //
//                 // solve diagonal block
//                 //
            
//                 auto  U_ii = BU->block( i, i );
            
//                 if ( ! is_null( U_ii ) )
//                 {
//                     TScalarVector  v_i( sub_vector( v, U_ii->row_is() ) );
                
//                     trsvu( op_U, *U_ii, v_i, diag_mode );
//                 }// if

//                 //
//                 // update RHS
//                 //

//                 for ( uint  j = i+1; j < nbc; ++j )
//                 {
//                     auto  U_ij = BU->block( i, j );
                    
//                     if ( ! is_null( U_ij ) )
//                     {
//                         TScalarVector  v_i( sub_vector( v, U_ij->col_is() ) );
//                         TScalarVector  v_j( sub_vector( v, U_ij->row_is() ) );
                    
//                         HLR_LOG( 4, HLIB::to_string( "update( %d, ", U_ij->id() ) + v_j.is().to_string() + ", " + v_i.is().to_string() );
//                         mul_vec( real(-1), U_ij, & v_j, real(1), & v_i, op_U );
//                     }// if
//                 }// for
//             }// for
//         }// else
//     }// if
//     else if ( is_dense( & U ) )
//     {
//         if ( diag_mode == general_diag )
//         {
//             //
//             // assuming U contains inverse (store_inverse!)
//             //

//             TScalarVector  vc( v );

//             mul_vec( real(1), & U, & vc, real(0), & v, op_U );
//         }// if
//     }// if
//     else
//         assert( false );
// }

}}// namespace hlr::seq
