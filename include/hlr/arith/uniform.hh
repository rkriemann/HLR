#ifndef __HLR_ARITH_UNIFORM_HH
#define __HLR_ARITH_UNIFORM_HH
//
// Project     : HLR
// Module      : arith/uniform.hh
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <list>

#include <hlr/arith/detail/uniform_basis.hh>
#include <hlr/arith/detail/uniform.hh>
#include <hlr/arith/detail/uniform_tlr.hh>
#include <hlr/arith/detail/uniform_accu.hh>
#include <hlr/arith/detail/uniform_accu2.hh>
#include <hlr/arith/detail/uniform_accu3.hh>
#include <hlr/arith/detail/uniform_accu4.hh>

#include <hlr/utils/timer.hh> // DEBUG

namespace hlr { namespace uniform {

//
// construct mappings of A_{t × s} to set of uniform leaves per t/s
//
template < typename value_t >
is_matrix_cmap_t< value_t >
construct_indexset_to_block_map_rows ( const hpro::TMatrix< value_t > &  A,
                                       const bool                        all_leaves )
{
    auto  rowmap = is_matrix_cmap_t< value_t >();
    auto  blocks = std::list< const hpro::TMatrix< value_t > * >{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< const hpro::TMatrix< value_t > * >();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = cptrcast( M, hpro::TBlockMatrix< value_t > );

                for ( uint  i = 0; i < BM->nblock_rows(); ++i )
                    for ( uint  j = 0; j < BM->nblock_cols(); ++j )
                        if ( ! is_null( BM->block( i, j ) ) )
                            subblocks.push_back( BM->block( i, j ) );
            }// if
            else
            {
                if ( all_leaves || matrix::is_uniform_lowrank( M ) )
                    rowmap[ M->row_is() ].push_back( M );
            }// if
        }// for

        blocks = std::move( subblocks );
    }// while

    return rowmap;
}

//
// construct mappings of A_{t × s} to set of uniform leaves per t/s
//
template < typename value_t >
std::pair< is_matrix_map_t< value_t >,
           is_matrix_map_t< value_t > >
construct_indexset_to_block_maps ( hpro::TMatrix< value_t > &  A )
{
    auto  rowmap = is_matrix_map_t< value_t >();
    auto  colmap = is_matrix_map_t< value_t >();

    auto  blocks = std::list< hpro::TMatrix< value_t > * >{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix< value_t > * >();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = ptrcast( M, hpro::TBlockMatrix< value_t > );

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
mul_vec ( const value_t                              alpha,
          const hpro::matop_t                        op_M,
          const hpro::TMatrix< value_t > &           M,
          const vector::scalar_vector< value_t > &   x,
          vector::scalar_vector< value_t > &         y,
          matrix::shared_cluster_basis< value_t > &  rowcb,
          matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    //
    // construct uniform representation of x and y
    //

    // auto  tic = timer::now();
    auto  ux = detail::scalar_to_uniform( ( op_M == hpro::apply_normal ? colcb : rowcb ), x );
    auto  uy = hlr::vector::make_uniform< value_t, matrix::shared_cluster_basis< value_t > >( ( op_M == hpro::apply_normal ? rowcb : colcb ) );
    // auto  toc = timer::since( tic );
    // auto  t1  = toc.seconds();

    // tic = timer::now();
    detail::mul_vec( alpha, op_M, M, *ux, *uy, x, y );
    detail::add_uniform_to_scalar( *uy, y );
    // toc = timer::since( tic );
    // auto  t2  = toc.seconds();

    // std::cout << t1 << " / " << t2 << std::endl;
}

template < typename value_t >
void
mul_vec2 ( const value_t                              alpha,
           const hpro::matop_t                        op_M,
           const hpro::TMatrix< value_t > &           M,
           const vector::scalar_vector< value_t > &   x,
           vector::scalar_vector< value_t > &         y,
           matrix::shared_cluster_basis< value_t > &  rowcb,
           matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    //
    // construct uniform representation of x and y
    //

    // auto  tic = timer::now();
    auto  cmap   = detail::is_veccoeff_map_t< value_t >();
    auto  rowmap = construct_indexset_to_block_map_rows( M, true );
    
    detail::scalar_to_uniform( op_M == hpro::apply_normal ? colcb : rowcb, x, cmap );
    // auto  toc = timer::since( tic );
    // auto  t1  = toc.seconds();
    
    // tic = timer::now();
    if ( op_M == apply_normal )
        detail::mul_vec2( alpha, op_M, M, cmap, x, y, rowmap );
    else
        HLR_ERROR( "TO DO" );
    // toc = timer::since( tic );
    // auto  t2  = toc.seconds();

    // std::cout << t1 << " / " << t2 << std::endl;
}

template < typename value_t >
void
mul_vec_hier ( const value_t                                        alpha,
               const hpro::matop_t                                  op_M,
               const matrix::level_hierarchy< value_t > &           M,
               const vector::scalar_vector< value_t > &             x,
               vector::scalar_vector< value_t > &                   y,
               matrix::shared_cluster_basis_hierarchy< value_t > &  rowcb,
               matrix::shared_cluster_basis_hierarchy< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    //
    // construct uniform representation of x and y
    //

    auto  ux = detail::scalar_to_uniform( ( op_M == hpro::apply_normal ? colcb : rowcb ), x );

    detail::mul_vec_hier( alpha, op_M, M, *ux, x, y, ( op_M == hpro::apply_normal ? rowcb : colcb ) );
}

//
// block row wise computation using IDs of involved objects
//
template < typename value_t >
void
mul_vec_row ( const value_t                                                         alpha,
              const hpro::matop_t                                                   op_M,
              const vector::scalar_vector< value_t > &                              x,
              vector::scalar_vector< value_t > &                                    y,
              const matrix::shared_cluster_basis< value_t > &                       rowcb,
              const matrix::shared_cluster_basis< value_t > &                       colcb,
              const std::vector< matrix::shared_cluster_basis< value_t > * > &      colcb_map,
              const std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap )
{
    if ( alpha == value_t(0) )
        return;

    auto  xcoeff = std::vector< blas::vector< value_t > >( colcb.id() + 1 );

    detail::scalar_to_uniform( colcb, x, xcoeff );
    detail::mul_vec_row< value_t >( alpha, op_M, rowcb, colcb_map, blockmap, xcoeff, x, y );
}

template < typename value_t >
std::vector< matrix::shared_cluster_basis< value_t > * >
build_id2cb ( matrix::shared_cluster_basis< value_t > &  cb )
{
    HLR_ASSERT( cb.id() != -1 );
    
    auto  idmap = std::vector< matrix::shared_cluster_basis< value_t > * >( cb.id() + 1 );

    detail::build_id2cb( cb, idmap );

    return idmap;
}

template < typename value_t >
std::vector< std::list< const Hpro::TMatrix< value_t > * > >
build_id2blocks ( const matrix::shared_cluster_basis< value_t > &  cb,
                  const Hpro::TMatrix< value_t > &                 M,
                  const bool                                       transposed )
{
    HLR_ASSERT( cb.id() != -1 );

    auto  blockmap = std::vector< std::list< const Hpro::TMatrix< value_t > * > >( cb.id() + 1 );

    detail::build_id2blocks( cb, M, blockmap, transposed );

    return blockmap;
}

//
// return FLOPs needed for computing y = y + α op( M ) x
// (implicit vectors)
//
template < typename value_t,
           typename cluster_basis_t >
flops_t
mul_vec_flops ( const Hpro::matop_t               op_M,
                const Hpro::TMatrix< value_t > &  M,
                const cluster_basis_t &           rowcb,
                const cluster_basis_t &           colcb )
{
    flops_t  flops = 0;

    flops += detail::scalar_to_uniform_flops( ( op_M == apply_normal ? colcb : rowcb ) );
    flops += detail::mul_vec_flops( op_M, M );
    flops += detail::add_uniform_to_scalar_flops( ( op_M == apply_normal ? rowcb : colcb ) );

    return flops;
}

//
// return size of data involved in computing y = y + α op( M ) x
//
template < typename value_t,
           typename cluster_basis_t >
flops_t
mul_vec_datasize ( const Hpro::matop_t               op_M,
                   const Hpro::TMatrix< value_t > &  M,
                   const cluster_basis_t &           rowcb,
                   const cluster_basis_t &           colcb )
{
    size_t  dsize = 0;

    // scalar to uniform: cluster basis and vector
    dsize += ( op_M == apply_normal ? colcb : rowcb ).data_byte_size() + sizeof( value_t ) * M.ncols( op_M );

    // actual matvec: matrix size
    dsize += M.data_byte_size();

    // uniform to scalar: cluster basis and vector
    dsize += ( op_M == apply_normal ? rowcb : colcb ).data_byte_size() + sizeof( value_t ) * M.nrows( op_M );

    return dsize;
}

//
// matrix multiplication (eager version)
//
template < typename value_t,
           typename approx_t >
void
multiply ( const value_t                     alpha,
           const hpro::matop_t               op_A,
           const hpro::TMatrix< value_t > &  A,
           const hpro::matop_t               op_B,
           const hpro::TMatrix< value_t > &  B,
           hpro::TMatrix< value_t > &        C,
           const hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
{
    //
    // construct mapping of A_{t × s} to set of uniform leaves per t/s
    //

    auto  rowmap = is_matrix_map_t< value_t >();
    auto  colmap = is_matrix_map_t< value_t >();

    auto  blocks = std::list< hpro::TMatrix< value_t > * >{ &C };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix< value_t > * >();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  BM = ptrcast( M, hpro::TBlockMatrix< value_t > );

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
lu ( hpro::TMatrix< value_t > &  A,
     const hpro::TTruncAcc &     acc,
     const approx_t &            approx,
     hpro::TMatrix< value_t > &          /* REF */ )
{
    //
    // construct mapping of A_{t × s} to set of uniform leaves per t/s
    //

    auto  rowmap = is_matrix_map_t< value_t >();
    auto  colmap = is_matrix_map_t< value_t >();

    auto  blocks = std::list< hpro::TMatrix< value_t > *>{ &A };

    while ( ! blocks.empty() )
    {
        auto  subblocks = std::list< hpro::TMatrix< value_t > *>();

        for ( auto  M : blocks )
        {
            if ( is_blocked( M ) )
            {
                auto  B = ptrcast( M, hpro::TBlockMatrix< value_t > );

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
multiply ( const value_t                     alpha,
           const matop_t                     op_A,
           const hpro::TMatrix< value_t > &  A,
           const matop_t                     op_B,
           const hpro::TMatrix< value_t > &  B,
           hpro::TMatrix< value_t > &        C,
           const hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
{
    auto  [ rowmap, colmap ] = construct_indexset_to_block_maps( C );
    auto  prod_inner         = detail::inner_map_t< value_t >();
    auto  accu               = detail::accumulator< value_t >( & prod_inner );

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
}

template < typename value_t,
           typename approx_t >
void
multiply_cached ( const value_t                     alpha,
                  const matop_t                     op_A,
                  const hpro::TMatrix< value_t > &  A,
                  const matop_t                     op_B,
                  const hpro::TMatrix< value_t > &  B,
                  hpro::TMatrix< value_t > &        C,
                  const hpro::TTruncAcc &           acc,
                  const approx_t &                  approx )
{
    multiply< value_t, approx_t >( alpha, op_A, A, op_B, B, C, acc, approx );
}

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix< value_t > &  A,
     const hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    auto  [ rowmap, colmap ] = construct_indexset_to_block_maps( A );
    // auto  inner_prod         = detail::inner_map_t< value_t >();
    auto  accu               = detail::accumulator< value_t >();

    detail::lu< value_t >( A, accu, acc, approx, rowmap, colmap );
}

}// namespace accu


namespace accu2
{

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix< value_t > &                 A,
     hpro::TMatrix< value_t > &                 L,
     hpro::TMatrix< value_t > &                 U,
     const hpro::TTruncAcc &                    acc,
     const approx_t &                           approx,
     matrix::shared_cluster_basis< value_t > &  rowcb_L,
     matrix::shared_cluster_basis< value_t > &  colcb_L,
     matrix::shared_cluster_basis< value_t > &  rowcb_U,
     matrix::shared_cluster_basis< value_t > &  colcb_U )
{
    auto  rowmap_L = is_matrix_map_t< value_t >();
    auto  colmap_L = is_matrix_map_t< value_t >();
    auto  rowmap_U = is_matrix_map_t< value_t >();
    auto  colmap_U = is_matrix_map_t< value_t >();
    auto  accu     = detail::accumulator< value_t >();
    
    detail::lu< value_t >( A, L, U, accu, acc, approx,
                           rowcb_L, colcb_L,
                           rowcb_U, colcb_U,
                           rowmap_L, colmap_L,
                           rowmap_U, colmap_U ); //, REF );
}

}// namespace accu2

namespace accu3
{

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix< value_t > &  A,
     hpro::TMatrix< value_t > &  L,
     hpro::TMatrix< value_t > &  U,
     const hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    auto  [ rowmap_L, colmap_L ] = construct_indexset_to_block_maps( L );
    auto  [ rowmap_U, colmap_U ] = construct_indexset_to_block_maps( U );
    auto  accu                   = detail::accumulator< value_t >();
    
    detail::lu< value_t >( A, L, U, accu, acc, approx,
                           rowmap_L, colmap_L,
                           rowmap_U, colmap_U ); //, REF );
}

}// namespace accu3

namespace accu4
{

template < typename value_t,
           typename approx_t >
void
lu ( hpro::TMatrix< value_t > &                 A,
     const hpro::TTruncAcc &                    acc,
     const approx_t &                           approx,
     matrix::shared_cluster_basis< value_t > &  rowcb,
     matrix::shared_cluster_basis< value_t > &  colcb )
{
    auto  rowmap = is_matrix_map_t< value_t >();
    auto  colmap = is_matrix_map_t< value_t >();
    auto  accu   = detail::accumulator< value_t >();
    
    detail::lu< value_t >( A, accu, acc, approx, rowcb, colcb, rowmap, colmap );
}

}// namespace accu4

//////////////////////////////////////////////////////////////////////
//
// TLR versions
//
//////////////////////////////////////////////////////////////////////

namespace tlr
{

//
// mat-vec : y = y + α op( M ) x
//
template < typename value_t >
void
mul_vec ( const value_t                              alpha,
          const hpro::matop_t                        op_M,
          const hpro::TMatrix< value_t > &           M,
          const vector::scalar_vector< value_t > &   x,
          vector::scalar_vector< value_t > &         y,
          matrix::shared_cluster_basis< value_t > &  rowcb,
          matrix::shared_cluster_basis< value_t > &  colcb )
{
    if ( alpha == value_t(0) )
        return;

    if ( op_M == apply_normal )
        detail::mul_vec( alpha, op_M, M, x, y, rowcb, colcb );
    else
        detail::mul_vec( alpha, op_M, M, x, y, colcb, rowcb );
}

//
// add global low-rank matrix W·X' to H²-matrix M
//
// template < typename value_t >
// void
// addlr ( hpro::TMatrix< value_t > &                  M,
//         const blas::matrix< value_t > &  W,
//         const blas::matrix< value_t > &  X,
//         const hpro::TTruncAcc &          acc )
// {
//     HLR_ASSERT( is_blocked( M ) );

//     auto  B = ptrcast( &M, hpro::TBlockMatrix< value_t > );
    
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
//                 auto  D_ij = ptrcast( B_ij, hpro::TDenseMatrix< value_t > );

//                 blas::prod( value_t(1), W_i, blas::adjoint( X_j ), value_t(1), blas::mat( D_ij ) );
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
multiply ( const value_t                     alpha,
           const hpro::matop_t               op_A,
           const hpro::TMatrix< value_t > &  aA,
           const hpro::matop_t               op_B,
           const hpro::TMatrix< value_t > &  aB,
           hpro::TMatrix< value_t > &        aC,
           const hpro::TTruncAcc &           acc,
           const approx_t &                  approx )
{
    HLR_ASSERT( is_blocked_all( aA, aB, aC ) );

    auto  A = cptrcast( &aA, hpro::TBlockMatrix< value_t > );
    auto  B = cptrcast( &aB, hpro::TBlockMatrix< value_t > );
    auto  C = ptrcast(  &aC, hpro::TBlockMatrix< value_t > );

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
lu ( hpro::TMatrix< value_t > &  A,
     const hpro::TTruncAcc &     acc,
     const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix< value_t > );
        auto  D_ii = blas::mat( A_ii );
            
        blas::invert( D_ii );

        //
        // L is unit diagonal so just solve with U, e.g. X_ji U_ii = M_ji
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix< value_t > );
                auto  T_ji = blas::copy( blas::mat( D_ji ) );

                blas::prod( value_t(1), T_ji, blas::mat( A_ii ), value_t(0), blas::mat( D_ji ) );
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
                auto  MV_i = blas::prod( blas::adjoint( blas::mat( A_ii ) ), V_i );

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
lu_sep ( hpro::TMatrix< value_t > &  L,
         hpro::TMatrix< value_t > &  U,
         const hpro::TTruncAcc &     acc,
         const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", U.id() ) );
    
    HLR_ASSERT( is_blocked_all( L, U ) );

    auto  BL  = ptrcast( &L, hpro::TBlockMatrix< value_t > );
    auto  BU  = ptrcast( &U, hpro::TBlockMatrix< value_t > );
    auto  nbr = BU->nblock_rows();
    auto  nbc = BU->nblock_cols();

    HLR_ASSERT(( BL->nblock_rows() == BU->nblock_rows() ) &&
               ( BL->nblock_cols() == BU->nblock_cols() ));
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        auto  U_ii  = ptrcast( BU->block( i, i ), hpro::TDenseMatrix< value_t > );
        auto  L_ii  = ptrcast( BL->block( i, i ), hpro::TDenseMatrix< value_t > );
        auto  DU_ii = blas::mat( U_ii );
        auto  DL_ii = blas::mat( L_ii );
            
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

                // const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( blas::copy( QX ) ) );
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
                auto  D_ji = ptrcast( L_ji, hpro::TDenseMatrix< value_t > );
                auto  T_ji = blas::copy( blas::mat( D_ji ) );

                blas::prod( value_t(1), T_ji, DU_ii, value_t(0), blas::mat( D_ji ) );
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
lu_lazy ( hpro::TMatrix< value_t > &  A,
          const hpro::TTruncAcc &     acc,
          const approx_t &            approx,
          hpro::TMatrix< value_t > &  /* REF */ )
{
    HLR_LOG( 4, hpro::to_string( "lu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( & A, hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();

    // DEBUG {
    // auto  approx     = approx::SVD< value_t >();
    // auto  comp_error = [BA,&REF] ( const int  i,
    //                                const int  j )
    //                    {
    //                        auto  BREF   = cptrcast( & REF, hpro::TBlockMatrix< value_t > );
    //                        auto  REF_ij = matrix::convert_to_dense< value_t >( * BREF->block( i, j ) );
    //                        auto  LOC_ij = matrix::convert_to_dense< value_t >( * BA->block( i, j ) );
    //                        auto  M1     = blas::copy( blas::mat( REF_ij ) );
    //                        auto  M2     = blas::copy( blas::mat( LOC_ij ) );
                           
    //                        blas::add( value_t(-1), M1, M2 );

    //                        const auto  err = blas::norm_2( M2 ) / blas::norm_2( M1 );

    //                        std::cout << "  error " << BA->block( i, j )->id() << " : " << boost::format( "%.4e" ) % err << std::endl;
    //                    };

    // auto  BREF = ptrcast( & REF, hpro::TBlockMatrix< value_t > );
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
        
        // auto  REFA_ii = ptrcast( BREF->block( i, i ), hpro::TDenseMatrix< value_t > );
        // auto  REFD_ii = blas::mat( REFA_ii );

        // // comp_error( i, i );
        
        // blas::invert( REFD_ii );
        // // DEBUG }
        
        
        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix< value_t > );
        auto  D_ii = blas::mat( A_ii );
            
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
                //     auto  D_ji = ptrcast( BREF->block( j, i ), hpro::TDenseMatrix< value_t > );
                //     auto  T_ji = blas::copy( blas::mat( D_ji ) );

                //     blas::prod( value_t(1), T_ji, REFD_ii, value_t(0), blas::mat( D_ji ) );
                // }
                // // DEBUG }
                
                // X_ji = M_ji U_ii^-1
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix< value_t > );
                auto  T_ji = blas::copy( blas::mat( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat( D_ji ) );
            }// if
            else if ( matrix::is_uniform_lowrank( A_ji ) )
            {
                auto  [ Uu, Su, Vu ] = detail::apply_updates< value_t >( *BA, j, i, acc );

                // {
                //     auto  REF_ij = matrix::convert_to_dense< value_t >( * BREF->block( j, i ) );
                //     auto  M1     = blas::copy( blas::mat( REF_ij ) );
                //     auto  T1     = blas::prod( Uu, Su );
                //     auto  M2     = blas::prod( T1, blas::adjoint( Vu ) );
                           
                //     blas::add( value_t(-1), M2, M1 );
                //     std::cout << "  error " << BA->block( j, i )->id() << " : " << boost::format( "%.4e" ) % blas::norm_2( M1 ) << std::endl;
                // }

                // // DEBUG {
                // {
                //     auto  REFR_ji = ptrcast( BREF->block( j, i ), hpro::TRkMatrix< value_t > );
                //     auto  V       = blas::copy( blas::mat_V( REFR_ji ) );

                //     blas::prod( value_t(1), blas::adjoint( REFD_ii ), V, value_t(0), blas::mat_V( REFR_ji ) );
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
ldu ( hpro::TMatrix< value_t > &  A,
      const hpro::TTruncAcc &     acc,
      const approx_t &            approx )
{
    HLR_LOG( 4, hpro::to_string( "ldu( %d )", A.id() ) );
    
    assert( is_blocked( A ) );

    auto  BA  = ptrcast( &A, hpro::TBlockMatrix< value_t > );
    auto  nbr = BA->nblock_rows();
    auto  nbc = BA->nblock_cols();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        //
        // invert diagonal block
        //
        
        HLR_ASSERT( is_dense( BA->block( i, i ) ) );

        auto  A_ii = ptrcast( BA->block( i, i ), hpro::TDenseMatrix< value_t > );
        auto  T_ii = A_ii->copy(); // need original for matrix updates below
        auto  D_ii = blas::mat( ptrcast( A_ii, hpro::TDenseMatrix< value_t > ) );
        
        blas::invert( D_ii );

        //
        // L_ji D_ii U_ii = A_ji, since U_ii = I, we have L_ji = A_ji D_ii^-1
        //
        
        for ( uint  j = i+1; j < nbc; ++j )
        {
            auto  A_ji = BA->block( j, i );

            if ( is_dense( A_ji ) )
            {
                auto  D_ji = ptrcast( A_ji, hpro::TDenseMatrix< value_t > );
                auto  T_ji = blas::copy( blas::mat( D_ji ) );

                blas::prod( value_t(1), T_ji, D_ii, value_t(0), blas::mat( D_ji ) );
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
                auto  D_ij = ptrcast( U_ij, hpro::TDenseMatrix< value_t > );
                auto  T_ij = blas::copy( blas::mat( D_ij ) );

                blas::prod( value_t(1), D_ii, T_ij, value_t(0), blas::mat( D_ij ) );
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
                                  apply_normal, *cptrcast( T_ii.get(), hpro::TDenseMatrix< value_t > ),
                                  apply_normal, *BA,
                                  *BA, j, i, l, acc );
            }// for
        }// for
    }// for
}

}// namespace tlr

}}// namespace hlr::uniform

#endif // __HLR_ARITH_UNIFORM_HH
