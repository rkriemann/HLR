#ifndef __HLR_ARITH_UNIFORM_HH
#define __HLR_ARITH_UNIFORM_HH
//
// Project     : HLib
// Module      : arith/uniform.hh
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <list>

#include <hlr/arith/detail/uniform_bases.hh>
#include <hlr/arith/detail/uniform.hh>
#include <hlr/arith/detail/uniform_tlr.hh>
#include <hlr/arith/detail/uniform_accu.hh>
#include <hlr/arith/detail/uniform_accu3.hh>

namespace hlr { namespace uniform {

//
// construct mappings of A_{t × s} to set of uniform leaves per t/s
//
std::pair< detail::uniform_map_t,
           detail::uniform_map_t >
construct_indexset_to_block_maps ( hpro::TMatrix &  A )
{
    auto  rowmap = detail::uniform_map_t();
    auto  colmap = detail::uniform_map_t();

    auto  blocks = std::list< hpro::TMatrix * >{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix * >();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = ptrcast( M, hpro::TBlockMatrix );

                for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        if ( ! is_null( BM->block( i, j ) ) )
                            subblocks.push_back( BM->block( i, j ) );
            }// if
            else if ( matrix::is_uniform_lowrank( M ) )
            {
                rowmap[ M->row_is() ].push_back( M );
                colmap[ M->col_is() ].push_back( M );
            }// if
        }// for

        blocks = std::move( subblocks );
    }// while

    return { std::move( rowmap ), std::move( colmap ) };
}

////////////////////////////////////////////////////////////////////////////////
//
// functions for general uniform H-matrices
//
////////////////////////////////////////////////////////////////////////////////

//
// mat-vec : y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                             alpha,
          const hpro::matop_t                       op_M,
          const hpro::TMatrix &                     M,
          const vector::scalar_vector< value_t > &  x,
          vector::scalar_vector< value_t > &        y,
          matrix::cluster_basis< value_t > &        rowcb,
          matrix::cluster_basis< value_t > &        colcb )
{
    if ( alpha == value_t(0) )
        return;

    HLR_ASSERT( hpro::is_complex_type< value_t >::value == M.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == x.is_complex() );
    HLR_ASSERT( hpro::is_complex_type< value_t >::value == y.is_complex() );
    
    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x );
    auto  uy = detail::make_uniform(      op_M == hpro::apply_normal ? rowcb : colcb );

    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y );
}

//
// matrix multiplication (eager version)
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    A,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    //
    // construct mapping of A_{t × s} to set of uniform leaves per t/s
    //

    auto  rowmap = detail::uniform_map_t();
    auto  colmap = detail::uniform_map_t();

    auto  blocks = std::list< hpro::TMatrix * >{ &C };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix * >();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = ptrcast( M, hpro::TBlockMatrix );

                for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        if ( ! is_null( BM->block( i, j ) ) )
                            subblocks.push_back( BM->block( i, j ) );
            }// if
            else if ( matrix::is_uniform_lowrank( M ) )
            {
                rowmap[ M->row_is() ].push_back( M );
                colmap[ M->col_is() ].push_back( M );
            }// if
        }// for

        blocks = std::move( subblocks );
    }// while

    //
    // perform actual LU factorization
    //

    detail::multiply< value_t >( alpha, op_A, A, op_B, B, C, acc, approx, rowmap, colmap );
}

//
// LU factorization (eager version)
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx,
     hpro::TMatrix &          /* REF */ )
{
    //
    // construct mapping of A_{t × s} to set of uniform leaves per t/s
    //

    auto  rowmap = detail::uniform_map_t();
    auto  colmap = detail::uniform_map_t();

    auto  blocks = std::list< hpro::TMatrix *>{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix *>();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  B = ptrcast( M, hpro::TBlockMatrix );

                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        if ( ! is_null( B->block( i, j ) ) )
                            subblocks.push_back( B->block( i, j ) );
            }// if
            else if ( matrix::is_uniform_lowrank( M ) )
            {
                rowmap[ M->row_is() ].push_back( M );
                colmap[ M->col_is() ].push_back( M );
            }// if
        }// for

        blocks = std::move( subblocks );
    }// while

    //
    // perform actual LU factorization
    //

    detail::lu< value_t >( A, acc, approx, rowmap, colmap );
}

//////////////////////////////////////////////////////////////////////
//
// accumulator version
//
//////////////////////////////////////////////////////////////////////

namespace accu
{

template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const matop_t            op_A,
           const hpro::TMatrix &    A,
           const matop_t            op_B,
           const hpro::TMatrix &    B,
           hpro::TMatrix &          C,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    auto  [ rowmap, colmap ] = construct_indexset_to_block_maps( C );
    auto  inner_prod         = detail::inner_map_t();
    auto  accu               = detail::accumulator( nullptr );

    accu.add_update( op_A, A, op_B, B );
    
    detail::multiply( alpha, C, accu, acc, approx, rowmap, colmap ); //, REF );
}

template < typename value_t,
           typename approx_t >
void
multiply_cached ( const value_t            alpha,
                  const matop_t            op_A,
                  const hpro::TMatrix &    A,
                  const matop_t            op_B,
                  const hpro::TMatrix &    B,
                  hpro::TMatrix &          C,
                  const hpro::TTruncAcc &  acc,
                  const approx_t &         approx )
{
    auto  [ rowmap, colmap ] = construct_indexset_to_block_maps( C );
    auto  prod_inner         = detail::inner_map_t();
    auto  accu               = detail::accumulator( & prod_inner );

    accu.add_update( op_A, A, op_B, B );
    
    detail::multiply( alpha, C, accu, acc, approx, rowmap, colmap ); //, REF );

    // size_t  mem  = 0;
    // size_t  nmat = 0;
    
    // for( const auto & [ is, mat ] : prod_inner )
    // {
    //     mem += mat.byte_size();
    //     nmat++;
    //     // std::cout << std::max( mat.nrows(), mat.ncols() ) << std::endl;
    // }// for

    // std::cout << "inner  : " << mem << ", " << nmat << std::endl;
    
    // mem  = 0;
    // nmat = 0;

    // for( const auto & [ is, mat ] : prod_A )
    // {
    //     mem += mat.byte_size();
    //     nmat++;
    //     // std::cout << std::max( mat.nrows(), mat.ncols() ) << std::endl;
    // }// for

    // std::cout << "prod_A : " << mem << ", " << nmat << std::endl;
    
    // mem  = 0;
    // nmat = 0;

    // for( const auto & [ is, mat ] : prod_B )
    // {
    //     mem += mat.byte_size();
    //     nmat++;
    //     // std::cout << std::max( mat.nrows(), mat.ncols() ) << std::endl;
    // }// for

    // std::cout << "prod_B : " << mem << ", " << nmat << std::endl;
}

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx,
     hpro::TMatrix &          /* REF */ )
{
    auto  [ rowmap, colmap ] = construct_indexset_to_block_maps( A );
    auto  inner_prod         = detail::inner_map_t();
    auto  accu               = detail::accumulator( & inner_prod );

    detail::lu< value_t >( A, accu, acc, approx, rowmap, colmap ); //, REF );
}

}// namespace accu


namespace accu3
{

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     hpro::TMatrix &          L,
     hpro::TMatrix &          U,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx,
     hpro::TMatrix &          /* REF */ )
{
    auto  [ rowmap_L, colmap_L ] = construct_indexset_to_block_maps( L );
    auto  [ rowmap_U, colmap_U ] = construct_indexset_to_block_maps( U );
    auto  accu                   = detail::accumulator();
    
    detail::lu< value_t >( A, L, U, accu, acc, approx,
                           rowmap_L, colmap_L,
                           rowmap_U, colmap_U ); //, REF );
}

}// namespace accu3

//////////////////////////////////////////////////////////////////////
//
// TLR versions
//
//////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// add global low-rank matrix W·X' to H²-matrix M
//
// template < typename value_t >
// void
// addlr ( hpro::TMatrix &                  M,
//         const blas::matrix< value_t > &  W,
//         const blas::matrix< value_t > &  X,
//         const hpro::TTruncAcc &          acc )
// {
//     HLR_ASSERT( is_blocked( M ) );

//     auto  B = ptrcast( &M, hpro::TBlockMatrix );
    
//     //
//     // use inefficient method adding only local updates
//     //

//     for ( uint  i = 0; i < B->nblock_rows(); ++i )
//     {
//         for ( uint  j = 0; j < B->nblock_cols(); ++j )
//         {
//             auto  B_ij = B->block( i, j );
//             auto  W_i  = blas::matrix( W, B_ij->row_is() - B->row_ofs(), blas::range::all );
//             auto  X_j  = blas::matrix( X, B_ij->col_is() - B->col_ofs(), blas::range::all );
//             auto  I    = blas::identity< value_t >( X_j.ncols() );
                        
//             if ( matrix::is_uniform_lowrank( B_ij ) )
//             {
//                 auto  R_ij = ptrcast( B_ij, matrix::uniform_lrmatrix< value_t > );

//                 detail::addlr_global( *B, *R_ij, i, j, W_i, X_j, acc );
//             }// if
//             else if ( is_dense( B_ij ) )
//             {
//                 auto  D_ij = ptrcast( B_ij, hpro::TDenseMatrix );

//                 blas::prod( value_t(1), W_i, blas::adjoint( X_j ), value_t(1), blas::mat< value_t >( D_ij ) );
//             }// if
//             else
//                 HLR_ERROR( "unsupported matrix type : " + B_ij->typestr() );
//         }// for
//     }// for
// }

//
// matrix multiplication
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t            alpha,
           const hpro::matop_t      op_A,
           const hpro::TMatrix &    aA,
           const hpro::matop_t      op_B,
           const hpro::TMatrix &    aB,
           hpro::TMatrix &          aC,
           const hpro::TTruncAcc &  acc,
           const approx_t &         approx )
{
    HLR_ASSERT( is_blocked_all( aA, aB, aC ) );

    auto  A = cptrcast( &aA, hpro::TBlockMatrix );
    auto  B = cptrcast( &aB, hpro::TBlockMatrix );
    auto  C = ptrcast(  &aC, hpro::TBlockMatrix );

    HLR_ASSERT( C->nblock_rows()       == A->nblock_rows( op_A ) );
    HLR_ASSERT( C->nblock_cols()       == B->nblock_cols( op_B ) );
    HLR_ASSERT( A->nblock_cols( op_A ) == B->nblock_rows( op_B ) );

    for ( uint  i = 0; i < C->nblock_rows(); ++i )
    {
        for ( uint  j = 0; j < C->nblock_cols(); ++j )
        {
            HLR_ASSERT( ! is_null( C->block( i, j ) ) );

            for ( uint  k = 0; k < A->nblock_cols( op_A ); ++k )
            {
                detail::multiply( alpha, op_A, *A, op_B, *B, *C, i, k, j, acc, approx );
            }// for
        }// for
    }// for
}

//
// LU factorization A = L·U, with unit lower triangular L and upper triangular U
//
template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix &          A,
     const hpro::TTruncAcc &  acc,
     const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  D_ii = blas::mat< value_t >( A_ii );
            
        blas::invert( D_ii );

        //
        // L is unit diagonal so just solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, blas::mat< value_t >( A_ii ), value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                //
                // X_ji U_ii = Ũ_j Ŝ_ji Ṽ_i' U_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' U_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( U_ii^-T V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( blas::adjoint( blas::mat< value_t >( A_ii ) ), V_i );

                detail::extend_col_basis< value_t >( *BA, *R_ji, j, i, MV_i, acc, approx );
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, j, i, l, acc, approx );
            }// for
        }// for
    }// for
}

//
// LU factorization A = L·U, with unit lower triangular L and upper triangular U
// - version with separate matrices for L/U and also separate cluster bases
// - A is assumed to be stored in L (lower part) and U (upper part including diagonal)
//   upon entry
//
template < typename value_t,
           typename approx_t >
void
lu_sep ( hpro::TMatrix &          L,
         hpro::TMatrix &          U,
         const hpro::TTruncAcc &  acc,
         const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", U.id() ) );
    
    HLR_ASSERT( is_blocked_all( L, U ) );

    auto  BL  = ptrcast( &L, hpro::TBlockMatrix );
    auto  BU  = ptrcast( &U, hpro::TBlockMatrix );
    auto  nbr = BU->nblock_rows();
    auto  nbc = BU->nblock_cols();

    HLR_ASSERT(( BL->nblock_rows() == BU->nblock_rows() ) &&
               ( BL->nblock_cols() == BU->nblock_cols() ));
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  U_ii  = ptrcast( BU->block( i, i ), hpro::TDenseMatrix );
        auto  L_ii  = ptrcast( BL->block( i, i ), hpro::TDenseMatrix );
        auto  DU_ii = blas::mat< value_t >( U_ii );
        auto  DL_ii = blas::mat< value_t >( L_ii );
            
        blas::invert( DU_ii );
        DL_ii = blas::identity< value_t >( DL_ii.nrows() );

        //
        // X_ji U_ii = U_j S_ji ( V_i' U_ii ) = U_j S_ji X_i'
        // is solved as X_i = U_ii'^-1 V_i
        //
        // As V_i is shared among all X_ji, so is X_i.
        //

        // update shared column basis of L
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  L_ji = BL->block( j, i );

            if ( matrix::is_uniform_lowrank( L_ji ) )
            {
                auto  R_ji = ptrcast( L_ji, matrix::uniform_lrmatrix< value_t > );
                auto  V_i  = R_ji->col_basis();
                auto  X_i  = blas::prod( blas::adjoint( DU_ii ), V_i );

                // const_cast< matrix::cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( blas::copy( QX ) ) );
                R_ji->col_cb().set_basis( std::move( X_i ) );

                break;
            }// if
        }// for

        // solve dense blocks
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  L_ji = BL->block( j, i );

            if ( is_dense( L_ji ) )
            {
                auto  D_ji = ptrcast( L_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, DU_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
        }// for

        //
        // update trailing submatrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < j; ++l )
            {
                detail::multiply( value_t(-1), apply_normal, *BL, apply_normal, *BU, *BL, j, i, l, acc, approx );
            }// for
        }// for

        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = j; l < nbc; ++l )
            {
                detail::multiply( value_t(-1), apply_normal, *BL, apply_normal, *BU, *BU, j, i, l, acc, approx );
            }// for
        }// for
    }// for
}

template < typename value_t,
           typename approx_t >
void
lu_lazy ( hpro::TMatrix &          A,
          const hpro::TTruncAcc &  acc,
          const approx_t &         approx,
          hpro::TMatrix &          /* REF */ )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( & A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    // DEBUG {
    // auto  approx     = approx::SVD< value_t >();
    // auto  comp_error = [BA,&REF] ( const int  i,
    //                                const int  j )
    //                    {
    //                        auto  BREF   = cptrcast( & REF, hpro::TBlockMatrix );
    //                        auto  REF_ij = matrix::convert_to_dense< value_t >( * BREF->block( i, j ) );
    //                        auto  LOC_ij = matrix::convert_to_dense< value_t >( * BA->block( i, j ) );
    //                        auto  M1     = blas::copy( blas::mat< value_t >( REF_ij ) );
    //                        auto  M2     = blas::copy( blas::mat< value_t >( LOC_ij ) );
                           
    //                        blas::add( value_t(-1), M1, M2 );

    //                        const auto  err = blas::norm_2( M2 ) / blas::norm_2( M1 );

    //                        std::cout << "  error " << BA->block( i, j )->id() << " : " << boost::format( "%.4e" ) % err << std::endl;
    //                    };

    // auto  BREF = ptrcast( & REF, hpro::TBlockMatrix );
    // DEBUG }
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        for ( int  k = 0; k < int(i); ++k )
            detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, i, k, i, acc, approx );

        // // DEBUG {
        // for ( int  k = 0; k < int(i); k++ )
        //     hlr::seq::multiply< value_t >( value_t(-1),
        //                                    apply_normal, *BREF->block( i, k ),
        //                                    apply_normal, *BREF->block( k, i ),
        //                                    *BREF->block( i, i ), acc, approx );
        
        // auto  REFA_ii = ptrcast( BREF->block( i, i ), hpro::TDenseMatrix );
        // auto  REFD_ii = blas::mat< value_t >( REFA_ii );

        // // comp_error( i, i );
        
        // blas::invert( REFD_ii );
        // // DEBUG }
        
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  D_ii = blas::mat< value_t >( A_ii );
            
        blas::invert( D_ii );

        // comp_error( i, i );

        //
        // solve with L, e.g. L_ii X_ij = M_ij
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            // comp_error( i, j );
            
            auto  A_ij = BA->block( i, j );

            // // DEBUG {
            // // only update block as L = I
            // for ( int  k = 0; k < int(i); k++ )
            //     hlr::seq::multiply< value_t >( value_t(-1),
            //                                    hpro::apply_normal, *BREF->block( i, k ),
            //                                    hpro::apply_normal, *BREF->block( k, j ),
            //                                    *BREF->block( i, j ), acc, approx );
            // // DEBUG }
                
            if ( is_dense( A_ij ) )
            {
                for ( int  k = 0; k < int(i); ++k )
                    detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, i, k, j, acc, approx );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ij ) )
            {
                auto  [ Uu, Su, Vu ] = detail::apply_updates< value_t >( *BA, i, j, acc );

                // even without solving, still need to update bases
                detail::replace_row_col_basis< value_t >( *BA, i, j, Uu, Su, Vu, acc );
                
                // comp_error( i, j );
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + A_ij->typestr() );
        }// for
        
        //
        // solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            // // DEBUG {
            // for ( int  k = 0; k < int(i); k++ )
            //     hlr::seq::multiply< value_t >( value_t(-1),
            //                                    hpro::apply_normal, *BREF->block( j, k ),
            //                                    hpro::apply_normal, *BREF->block( k, i ),
            //                                    *BREF->block( j, i ), acc, approx );
            // // DEBUG }
            
            if ( is_dense( A_ji ) )
            {
                for ( int  k = 0; k < int(i); ++k )
                    detail::multiply( value_t(-1), apply_normal, *BA, apply_normal, *BA, *BA, j, k, i, acc, approx );
                
                // // DEBUG {
                // {
                //     auto  D_ji = ptrcast( BREF->block( j, i ), hpro::TDenseMatrix );
                //     auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                //     blas::prod( value_t(1), T_ji, REFD_ii, value_t(0), blas::mat< value_t >( D_ji ) );
                // }
                // // DEBUG }
                
                // X_ji = M_ji U_ii^-1
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                auto  [ Uu, Su, Vu ] = detail::apply_updates< value_t >( *BA, j, i, acc );

                // {
                //     auto  REF_ij = matrix::convert_to_dense< value_t >( * BREF->block( j, i ) );
                //     auto  M1     = blas::copy( blas::mat< value_t >( REF_ij ) );
                //     auto  T1     = blas::prod( Uu, Su );
                //     auto  M2     = blas::prod( T1, blas::adjoint( Vu ) );
                           
                //     blas::add( value_t(-1), M2, M1 );
                //     std::cout << "  error " << BA->block( j, i )->id() << " : " << boost::format( "%.4e" ) % blas::norm_2( M1 ) << std::endl;
                // }

                // // DEBUG {
                // {
                //     auto  REFR_ji = ptrcast( BREF->block( j, i ), hpro::TRkMatrix );
                //     auto  V       = blas::copy( blas::mat_V< value_t >( REFR_ji ) );

                //     blas::prod( value_t(1), blas::adjoint( REFD_ii ), V, value_t(0), blas::mat_V< value_t >( REFR_ji ) );
                // }
                // // DEBUG }
                
                //
                // X_ji U_ii = Ũ_j Ŝ_ji Ṽ_i' U_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' U_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( U_ii^-T V_i )
                //
                
                auto  MV_i = blas::prod( blas::adjoint( D_ii ), Vu );
                auto  RV   = blas::matrix< value_t >();

                // ensure orthogonality in new basis
                blas::qr( MV_i, RV );
                
                auto  T = blas::prod( Su, blas::adjoint( RV ) );
                    
                detail::replace_row_col_basis< value_t >( *BA, j, i, Uu, T, MV_i, acc );

                // comp_error( j, i );
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for
    }// for
}

//
// LDU factorization A = L·D·U, with unit lower/upper triangular L/U and diagonal D
//
template < typename value_t,
           typename approx_t >
void
ldu ( hpro::TMatrix &          A,
      const hpro::TTruncAcc &  acc,
      const approx_t &         approx )
{
    HLR_LOG( 4, hpro::to_string( "ldu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        HLR_ASSERT( is_dense( BA->block( i, i ) ) );

        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix );
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = blas::mat< value_t >( ptrcast( A_ii, hpro::TDenseMatrix ) );
        
        blas::invert( D_ii );

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix );
                auto  T_ji = blas::copy( blas::mat< value_t >( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat< value_t >( D_ji ) );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                //
                // X_ji D_ii = Ũ_j Ŝ_ji Ṽ_i' D_ii = U_j S_ji V_i'
                // is solved as U_j S_ji V_i' D_ii^-1
                // and hence Ũ_j = U_j, Ŝ_ji = S_ji and Ṽ_i = ( D_ii^-T V_i )
                //
                
                auto  R_ji = ptrcast( A_ji, matrix::uniform_lrmatrix< value_t > );
                auto  V_i  = R_ji->col_cb().basis();
                auto  MV_i = blas::prod( blas::adjoint( D_ii ), V_i );

                detail::extend_col_basis< value_t >( *BA, *R_ji, j, i, MV_i, acc, approx );
            }// if
            else
                HLR_ERROR( "matrix type not supported : " + A_ji->typestr() );
        }// for

        //
        // L_ii D_ii U_ij = A_ij, since L_ii = I, we have U_ij = D_ii^-1·A_ij
        //

        for ( uint  j = i+1; j < nbr; ++j )
        {
            auto  U_ij = BA->block( i, j );

            if ( is_dense( U_ij ) )
            {
                auto  D_ij = ptrcast( U_ij, hpro::TDenseMatrix );
                auto  T_ij = blas::copy( blas::mat< value_t >( D_ij ) );

                blas::prod( value_t(1), D_ii, T_ij, value_t(0), blas::mat< value_t >( D_ij ) );
            }// else
            else if ( matrix::is_uniform_lowrank( U_ij ) )
            {
                // U_ij = W·T·X' = D_ii^-1·U·S·V' = D_ii^-1·A_ij
                // ⟶ W = D_ii^-1·U, T=S, X = V
                auto  R_ij = ptrcast( U_ij, matrix::uniform_lrmatrix< value_t > );
                auto  U_i  = R_ij->row_cb().basis();
                auto  MU_i = blas::prod( D_ii, U_i );

                detail::extend_row_basis< value_t >( *BA, *R_ij, i, j, MU_i, acc );
            }// if
        }// for

        //
        // update trailing sub matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            for ( uint  l = i+1; l < nbc; ++l )
            {
                detail::multiply( value_t(-1),
                                  apply_normal, *BA,
                                  apply_normal, *cptrcast( T_ii.get(), hpro::TDenseMatrix ),
                                  apply_normal, *BA,
                                  *BA, j, i, l, acc );
            }// for
        }// for
    }// for
}

}// namespace tlr

}}// namespace hlr::uniform

#endif // __HLR_ARITH_UNIFORM_HH
