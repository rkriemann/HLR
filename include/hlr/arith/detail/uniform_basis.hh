#ifndef __HLR_ARITH_DETAIL_UNIFORM_BASIS_HH
#define __HLR_ARITH_DETAIL_UNIFORM_BASIS_HH
//
// Project     : HLR
// Module      : arith/uniform
// Description : functions for handling changes in basis for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

// #include <boost/format.hpp> // DEBUG

#include <hlr/arith/blas.hh>
#include <hlr/arith/norm.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
// #include <hlr/utils/io.hh> // DEBUG

namespace hlr { namespace uniform {

//
// maps index set to set of all matrices in block row (or column)
//
template < typename value_t >
using  is_matrix_map_t  = std::unordered_map< indexset, std::list< Hpro::TMatrix< value_t > * >, indexset_hash >;

template < typename value_t >
using  is_matrix_cmap_t = std::unordered_map< indexset, std::list< const Hpro::TMatrix< value_t > * >, indexset_hash >;

//////////////////////////////////////////////////////////////////////
//
// general versions
//
//////////////////////////////////////////////////////////////////////

namespace detail
{

using matrix::shared_cluster_basis;
using matrix::uniform_lrmatrix;
using matrix::is_uniform_lowrank;

//
// extend row basis <cb> by block W·T·X' (X is not needed for computation)
//
// - <matmap> contains mapping of cluster basis indexset to set of matrices sharing <cb>
// - if M != nullptr, it is excluded from computation
// - W is assumed to be orthogonal
//
template < typename value_t,
           typename approx_t >
blas::matrix< value_t >
compute_extended_row_basis ( const shared_cluster_basis< value_t > &         cb,
                             const blas::matrix< value_t > &                 W,
                             const blas::matrix< value_t > &                 T,
                             const accuracy &                                acc,
                             const approx_t &                                approx,
                             const is_matrix_map_t< value_t > &              matmap,
                             const uniform_lrmatrix< value_t > *             M  = nullptr,
                             blas::vector< Hpro::real_type_t< value_t > > *  sv = nullptr )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // compute new row basis of
    //
    //   (U·S₁·V₁'  U·S₂·V₂'  …  U·S_j·V_j'  W·T·X')
    //
    //    = (U W) ⎛S₁·V₁'  S₂·V₂' …  S_j·V_j'  0  ⎞
    //            ⎝  0       0          0     T·X'⎠
    //
    //    = (U W) ⎛V₁·S₁'    0  ⎞'
    //            ⎜V₂·S₂'    0  ⎟
    //            ⎜      …      ⎟
    //            ⎜V_j·S_j'  0  ⎟
    //            ⎝   0     X·T'⎠
    //
    //    = (U W) ⎛⎛V₁         ⎞ ⎛S₁'   0 ⎞⎞'
    //            ⎜⎜  V₂       ⎟ ⎜S₂'   0 ⎟⎟
    //            ⎜⎜    …      ⎟·⎜   ...  ⎟⎟
    //            ⎜⎜      V_j  ⎟ ⎜S_j'  0 ⎟⎟
    //            ⎝⎝          X⎠ ⎝ 0    T'⎠⎠
    //
    // Since V_i and X are orthogonal, one can skip those for basis computation.
    // Compute QR factorization
    //
    //   Q·R = ⎛S₁'  0 ⎞ = S
    //         ⎜S₂'  0 ⎟
    //         ⎜  ...  ⎟
    //         ⎜S_j' 0 ⎟
    //         ⎝ 0   T'⎠
    //
    // of which also Q is omitted, which leaves to compute the column basis of
    //
    //   (U W) R' = U_e R'
    //
    // The S_i and T are scaled by the (spectral) norm of the corresponding block
    // U_i·S_i·V' and W·T·X' to achieve the relative precision for all blocks.
    //

    // determine number of rows of matrix S below (sum of column ranks)
    size_t  nrows_S = T.ncols();  // known apriori

    if ( matmap.find( cb.is() ) != matmap.end() )
    {
        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( matrix::is_uniform_lowrank( M_i ) && ( M_i != M ))
                nrows_S += cptrcast( M_i, matrix::uniform_lrmatrix< value_t > )->col_rank();
        }// for
    }// if

    if ( nrows_S == T.ncols() )
    {
        //
        // since there is no other low-rank block, new row basis is W
        //

        if ( ! is_null( sv ) )
        {
            // determine singular values from T
            auto  S = blas::sv( T );

            HLR_ASSERT( W.ncols() >= S.length() );
            
            if ( sv->length() != W.ncols() )
                *sv = blas::vector< real_t >( W.ncols() );

            for ( uint  i = 0; i < W.ncols(); ++i )
                (*sv)(i) = S(i);
        }// if
        
        return std::move( blas::copy( W ) );
    }// if
    else
    {
        // extended row basis
        auto  U  = cb.basis();
        auto  Ue = blas::join_row< value_t >( { U, W } );

        // compute QR of column basis for each block in row and assemble
        // all results into common matrix Q
        auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
        size_t  pos = 0;

        for ( auto  M_i : matmap.at( cb.is() ) )     // being here also implies non-empty matrix set
        {
            if ( matrix::is_uniform_lowrank( M_i ) && ( M_i != M ))
            {
                // R_i = U_i·S_i·V' with U/V being orthogonal, hence |R_i| = |S_i|
                const auto  R_i  = cptrcast( M_i, matrix::uniform_lrmatrix< value_t > );
                const auto  rank = R_i->col_rank();
                auto        S_i  = blas::copy( R_i->coeff() );
                auto        norm = norm::spectral( S_i );

                if ( norm != real_t(0) )
                    blas::scale( value_t(1) / norm, S_i );
            
                auto  S_sub = blas::matrix< value_t >( S,
                                                       blas::range( pos, pos + rank-1 ),
                                                       blas::range( 0, U.ncols() - 1 ) );

                blas::copy( blas::adjoint( S_i ), S_sub );
                pos += rank;
            }// else
        }// for

        //
        // and add part from W·T·X'
        //
        
        {
            // W/X are orthogonal, hence |W·T·X'| = |T|
            const auto  rank = T.ncols();
            auto        S_i  = blas::copy( T );
            auto        norm = norm::spectral( T );
            
            if ( norm != real_t(0) )
                blas::scale( value_t(1) / norm, S_i );
            
            auto  S_sub = blas::matrix< value_t >( S,
                                                   blas::range( pos, pos + rank-1 ),
                                                   blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
            blas::copy( blas::adjoint( S_i ), S_sub );
        }
        
        // compute QR of assembled matrix, and compute SVD of
        // product with extended column basis
        auto  R = blas::matrix< value_t >();
        
        blas::qr( S, R, false );

        auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
        auto  Un  = approx.column_basis( UeR, acc, sv );

        return  Un;
    }// else
}

//
// compute new row basis for block row of M with M being replaced by W·T·X'
//
template < typename value_t,
           typename approx_t >
blas::matrix< value_t >
compute_updated_row_basis ( const uniform_lrmatrix< value_t > &             M,
                            const blas::matrix< value_t > &                 W,
                            const blas::matrix< value_t > &                 T,
                            const accuracy &                                acc,
                            const approx_t &                                approx,
                            const is_matrix_map_t< value_t > &              matmap,
                            blas::vector< Hpro::real_type_t< value_t > > *  sv = nullptr )
{
    return compute_extended_row_basis( M.row_cb(), W, T, acc, approx, matmap, &M, sv );
}

//
// extend column basis <cb> by block W·T·X' (W is not needed for computation)
//
// - <matmap> contains mapping of cluster basis indexset to set of matrices sharing <cb>
// - if M != nullptr, it is excluded from computation
// - X is assumed to be orthogonal
//
template < typename value_t,
           typename approx_t >
blas::matrix< value_t >
compute_extended_col_basis ( const shared_cluster_basis< value_t > &         cb,
                             const blas::matrix< value_t > &                 T,
                             const blas::matrix< value_t > &                 X,
                             const accuracy &                                acc,
                             const approx_t &                                approx,
                             const is_matrix_map_t< value_t > &              matmap,
                             const uniform_lrmatrix< value_t > *             M  = nullptr,
                             blas::vector< Hpro::real_type_t< value_t > > *  sv = nullptr )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // compute new column basis
    //
    //   ⎛ U₁·S₁·V' ⎞' 
    //   ⎜ U₂·S₂·V' ⎟
    //   ⎜  ...     ⎟ = (V X) ⎛S₁'·U₁' S₂'·U₂' ... S_j'·U_j'   0  ⎞
    //   ⎜U_j S_j V'⎟         ⎝   0       0            0     T'·W'⎠
    //   ⎝  W T X'  ⎠
    //
    //                = (V X) ⎛ U₁·S₁     ⎞'   (V X) ⎛⎛U₁         ⎞⎛S₁   ⎞⎞'
    //                        ⎜ U₂·S₂     ⎟          ⎜⎜  U₂       ⎟⎜S₂   ⎟⎟
    //                        ⎜    ...    ⎟  =       ⎜⎜    ...    ⎟⎜ ... ⎟⎟
    //                        ⎜U_j·S_j    ⎟          ⎜⎜       U_j ⎟⎜S_j  ⎟⎟
    //                        ⎝        W·T⎠          ⎝⎝          W⎠⎝    T⎠⎠
    //
    // Since U_* and W are orthogonal, one can skip those for basis computation.
    // Compute QR factorization
    //
    //   Q·R = ⎛S₁  0⎞ = S
    //         ⎜S₂  0⎟
    //         ⎜ ... ⎟
    //         ⎜S_j 0⎟
    //         ⎝ 0  T⎠
    //
    // and finally column basis of
    //
    //   (V X) R' = V_e R'
    //
    // Please note, that the S_i and T are scaled by the (spectral) norm of the
    // corresponding block U_i·S_i·V' and W·T·X'
    //
                                  
    // determine number of rows of matrix S below (sum of row ranks)
    size_t  nrows_S = T.nrows(); // known apriori
    
    if ( matmap.find( cb.is() ) != matmap.end() )
    {
        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( matrix::is_uniform_lowrank( M_i ) && ( M_i != M ))
                nrows_S += cptrcast( M_i, matrix::uniform_lrmatrix< value_t > )->row_rank();
        }// for
    }// if

    if ( nrows_S == T.nrows() )
    {
        //
        // since there is no other low-rank block, new basis is X
        //

        if ( ! is_null( sv ) )
        {
            // determine singular values from T
            auto  S = blas::sv( T );

            HLR_ASSERT( X.ncols() >= S.length() );
            
            if ( sv->length() != X.ncols() )
                *sv = blas::vector< real_t >( X.ncols() );

            for ( uint  i = 0; i < X.ncols(); ++i )
                (*sv)(i) = S(i);
        }// if
        
        return std::move( blas::copy( X ) );
    }// if
    else
    {
        //
        // otherwise compute new basis
        //
            
        auto  V  = cb.basis();
        auto  Ve = blas::join_row< value_t >( { V, X } );
    
        // assemble normalized coefficient matrices into common matrix S
        auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
        size_t  pos = 0;

        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( matrix::is_uniform_lowrank( M_i ) && ( M_i != M ))
            {
                // R_i = U S_i V_i' and U/V_i are orthogonal, hence |R_i| = |S_i|
                const auto  R_i  = cptrcast( M_i, matrix::uniform_lrmatrix< value_t > );
                const auto  rank = R_i->row_rank();
                auto        S_i  = blas::copy( R_i->coeff() );
                auto        norm = norm::spectral( S_i );

                if ( norm != real_t(0) )
                    blas::scale( value_t(1) / norm, S_i );

                auto  S_sub = blas::matrix< value_t >( S,
                                                       blas::range( pos, pos + rank-1 ),
                                                       blas::range( 0, V.ncols() - 1 ) );

                blas::copy( S_i, S_sub );
                pos += rank;
            }// else
        }// for

        //
        // add part from W·T·X'
        //
        
        {
            // W/X are orthogonal, hence |W·T·X'| = |T|
            const auto  rank = T.nrows();
            auto        S_i  = blas::copy( T );
            auto        norm = norm::spectral( T );
            
            if ( norm != real_t(0) )
                blas::scale( value_t(1) / norm, S_i );
                
            auto  S_sub = blas::matrix< value_t >( S,
                                                   blas::range( pos, pos + rank-1 ),
                                                   blas::range( V.ncols(), Ve.ncols() - 1 ) );

            blas::copy( S_i, S_sub );
            pos += rank;
        }// if

        // compute QR of assembled matrix, and compute SVD of
        // product with extended column basis
        auto  R = blas::matrix< value_t >();

        blas::qr( S, R, false );

        auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
        auto  Vn  = approx.column_basis( VeR, acc, sv );
        
        return  Vn;
    }// else
}

//
// compute new column basis for block row of M with M being replaced by W·T·X'
//
template < typename value_t,
           typename approx_t >
blas::matrix< value_t >
compute_updated_col_basis ( const uniform_lrmatrix< value_t > &             M,
                            const blas::matrix< value_t > &                 T,
                            const blas::matrix< value_t > &                 X,
                            const accuracy &                                acc,
                            const approx_t &                                approx,
                            const is_matrix_map_t< value_t > &              matmap,
                            blas::vector< Hpro::real_type_t< value_t > > *  sv = nullptr )
{
    return compute_extended_col_basis( M.col_cb(), T, X, acc, approx, matmap, &M, sv );
}

//
// update coupling matrices for all blocks sharing column basis <cb> to new basis <Vn>
//
// - if <M> is non-null, it is excluded from update
// - ATTENTION: no test if column basis of blocks is identical to <Vn>
//
template < typename value_t >
void
update_col_coupling ( const shared_cluster_basis< value_t > &  cb,
                      const blas::matrix< value_t > &          Vn,
                      const is_matrix_map_t< value_t > &       matmap,
                      const uniform_lrmatrix< value_t > *      M = nullptr )
{
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   S_i V' Vn = S_i·TV' with TV = Vn'·V
    //

    if ( matmap.find( cb.is() ) != matmap.end() )
    {
        auto  V  = cb.basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( matrix::is_uniform_lowrank( M_i ) && ( M_i != M ))
            {
                auto  R_i  = ptrcast( M_i, matrix::uniform_lrmatrix< value_t > );
                auto  Sn_i = blas::prod( R_i->coeff(), blas::adjoint( TV ) );

                // {// DEBUG {
                //     auto  US1   = blas::prod( R_i->row_basis(), R_i->coeff() );
                //     auto  M1    = blas::prod( US1, blas::adjoint( R_i->col_basis() ) );
                //     auto  US2   = blas::prod( R_i->row_basis(), Sn_i );
                //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
                //     blas::add( value_t(-1), M1, M2 );
                //     std::cout << "    col " << R_i->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
                // }// DEBUG }
                
                R_i->set_coeff_unsafe( std::move( Sn_i ) );
            }// if
        }// for
    }// if
}

//
// update coupling matrices for all blocks sharing row basis <cb> to new basis <Un>
//
// - if <M> is non-null, it is excluded from update
// - ATTENTION: no test if row basis of blocks is identical to <Un>
//
template < typename value_t >
void
update_row_coupling ( const shared_cluster_basis< value_t > &  cb,
                      const blas::matrix< value_t > &          Un,
                      const is_matrix_map_t< value_t > &       matmap,
                      const uniform_lrmatrix< value_t > *      M = nullptr )
{
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   Un'·U·S_i = TU·S_i  with TU = Un'·U
    //

    if ( matmap.find( cb.is() ) != matmap.end() )
    {
        auto  U  = cb.basis();
        auto  TU = blas::prod( blas::adjoint( Un ), U );

        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( matrix::is_uniform_lowrank( M_i ) && ( M_i != M ))
            {
                auto  R_i  = ptrcast( M_i, matrix::uniform_lrmatrix< value_t > );
                auto  Sn_i = blas::prod( TU, R_i->coeff() );

                // {// DEBUG {
                //     auto  US1   = blas::prod( R_i->row_basis(), R_i->coeff() );
                //     auto  M1    = blas::prod( US1, blas::adjoint( R_i->col_basis() ) );
                //     auto  US2   = blas::prod( Un, Sn_i );
                //     auto  M2    = blas::prod( US2, blas::adjoint( R_i->col_basis() ) );
                        
                //     blas::add( value_t(-1), M1, M2 );
                //     std::cout << "    row " << R_i->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
                // }// DEBUG }

                R_i->set_coeff_unsafe( std::move( Sn_i ) );
            }// if
        }// for
    }// if
}

//
// replace U·S·V' of M by W·T·X' and update row/column basis
// - ASSUMPTION: W and X are orthogonal
//
template < typename value_t,
           typename approx_t >
void
update_row_col_basis ( uniform_lrmatrix< value_t > &       M,
                       const blas::matrix< value_t > &     W,
                       const blas::matrix< value_t > &     T,
                       const blas::matrix< value_t > &     X,
                       const accuracy &                    acc,
                       const approx_t &                    approx,
                       const is_matrix_map_t< value_t > &  rowmap,
                       const is_matrix_map_t< value_t > &  colmap )
{
    // io::matlab::write( W, "W" );
    // io::matlab::write( T, "T" );
    // io::matlab::write( X, "X" );

    auto  Vn = compute_updated_col_basis( M, T, X, acc, approx, colmap );
    auto  Un = compute_updated_row_basis( M, W, T, acc, approx, rowmap );

    update_col_coupling( M.col_cb(), Vn, colmap, &M );
    update_row_coupling( M.row_cb(), Un, rowmap, &M );
    
    // {
    //     //
    //     // transform coupling matrix for blocks in current block column as
    //     //
    //     //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
    //     //

    //     auto  V  = M.col_cb().basis();
    //     auto  TV = blas::prod( blas::adjoint( Vn ), V );

    //     for ( auto  M_kj : colmap.at( M.col_is() ) )
    //     {
    //         if ( ! matrix::is_uniform_lowrank( M_kj ) )
    //             continue;
                    
    //         if ( M_kj != &M )
    //         {
    //             auto  R_kj  = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
    //             auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

    //             // // DEBUG {
    //             // {
    //             //     auto  US1   = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
    //             //     auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
    //             //     auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
    //             //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //             //     blas::add( value_t(-1), M1, M2 );
    //             //     std::cout << "    ext col/row : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
    //             // }
    //             // // DEBUG }

    //             R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
    //         }// if
    //     }// for
    // }

    // {
    //     //
    //     // transform coupling matrix for blocks in current block column as
    //     //
    //     //   Un'·U·S_i = TU·S_i  with TU = Un'·U
    //     //

    //     auto  U  = M.row_cb().basis();
    //     auto  TU = blas::prod( blas::adjoint( Un ), U );

    //     for ( auto  M_ik : rowmap.at( M.row_is() ) )
    //     {
    //         if ( ! matrix::is_uniform_lowrank( M_ik ) )
    //             continue;
                    
    //         if ( M_ik != &M )
    //         {
    //             auto  R_ik  = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
    //             auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

    //             // // DEBUG {
    //             // {
    //             //     auto  US1   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
    //             //     auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
    //             //     auto  US2   = blas::prod( Un, Sn_ik );
    //             //     auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
                        
    //             //     blas::add( value_t(-1), M1, M2 );
    //             //     std::cout << "    ext row/col : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    //             // }
    //             // // DEBUG }

    //             R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
    //         }// if
    //     }// for
    // }

    //
    // compute coupling of M_ij as Un' W T X' Vn
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  TX = blas::prod( blas::adjoint( Vn ), X );
    auto  S1 = blas::prod( TW, T );
    auto  Sn = blas::prod( S1, blas::adjoint( TX ) );

    // // DEBUG {
    // {
    //     // io::matlab::write( Un, "Un" );
    //     // io::matlab::write( Sn, "Sn" );
    //     // io::matlab::write( Vn, "Vn" );
        
    //     auto  US1   = blas::prod( W, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( X ) );
    //     auto  US2   = blas::prod( Un, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext    /    : " << M.id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    M.set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M.col_cb() )->set_basis( std::move( Vn ) );
    const_cast< matrix::shared_cluster_basis< value_t > * >( & M.row_cb() )->set_basis( std::move( Un ) );
}

//
// replace M=U·S·V' by W·T·V' and update row basis of
// all other blocks in block row
// - ASSUMPTION: W is orthogonal
//
template < typename value_t,
           typename approx_t >
void
update_row_basis ( uniform_lrmatrix< value_t > &       M,
                   const blas::matrix< value_t > &     W,
                   const blas::matrix< value_t > &     T,
                   const accuracy &                    acc,
                   const approx_t &                    approx,
                   const is_matrix_map_t< value_t > &  rowmap )
{
    auto  Un = compute_updated_row_basis( M, W, T, acc, approx, rowmap );

    // io::matlab::write( Un, "Un" );
    
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   TU ⎛S_kj⎞  or  TU ⎛  0 ⎞
    //      ⎝ 0  ⎠         ⎝S_kj⎠
    //

    auto  U  = M.row_cb().basis();
    auto  TU = blas::prod( blas::adjoint( Un ), U );

    for ( auto  M_ik : rowmap.at( M.row_is() ) )
    {
        if ( ! matrix::is_uniform_lowrank( M_ik ) )
            continue;
                    
        if ( M_ik != &M )
        {
            auto  R_ik  = ptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
            auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

            // // DEBUG {
            // {
            //     auto  US1   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
            //     auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            //     auto  US2   = blas::prod( Un, Sn_ik );
            //     auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
                        
            //     blas::add( value_t(-1), M1, M2 );
            //     std::cout << "    ext row     : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            // }
            // // DEBUG }

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
    }// for

    //
    // compute coupling of M_ij as Un' W T
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  Sn = blas::prod( TW, T );

    // // DEBUG {
    // {
    //     // io::matlab::write( Un, "Un" );
    //     // io::matlab::write( Sn, "Sn" );
    
    //     auto  V     = M.col_cb().basis();
    //     auto  US1   = blas::prod( W, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( V ) );
    //     auto  US2   = blas::prod( Un, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( V ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext row     : " << M.id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    M.set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M.row_cb() )->set_basis( std::move( Un ) );
}

//
// replace M=U·S·V' by U·T·X' and update row basis of
// all other blocks in block column
// - ASSUMPTION: X is orthogonal
//
template < typename value_t,
           typename approx_t >
void
update_col_basis ( uniform_lrmatrix< value_t > &       M,
                   const blas::matrix< value_t > &     T,
                   const blas::matrix< value_t > &     X,
                   const accuracy &                    acc,
                   const approx_t &                    approx,
                   const is_matrix_map_t< value_t > &  colmap )
{
    auto  Vn = compute_updated_col_basis( M, T, X, acc, approx, colmap );

    {
        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        auto  V  = M.col_cb().basis();
        auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( auto  M_kj : colmap.at( M.col_is() ) )
        {
            if ( ! matrix::is_uniform_lowrank( M_kj ) )
                continue;
                    
            if ( M_kj != &M )
            {
                auto  R_kj  = ptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

                // // DEBUG {
                // {
                //     auto  US1   = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
                //     auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
                //     auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
                //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
                //     blas::add( value_t(-1), M1, M2 );
                //     std::cout << "    ext col     : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
                // }
                // // DEBUG }

                R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
            }// if
        }// for
    }

    //
    // compute coupling of M as T X' Vn
    //

    auto  TX = blas::prod( T,  blas::adjoint( X ) );
    auto  Sn = blas::prod( TX, Vn );

    // // DEBUG {
    // {
    //     // io::matlab::write( Sn, "Sn" );
    //     // io::matlab::write( Vn, "Vn" );
    
    //     auto  U     = M.row_cb().basis();
    //     auto  US1   = blas::prod( U, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( X ) );
    //     auto  US2   = blas::prod( U, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext col     : " << M.id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    M.set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M.col_cb() )->set_basis( std::move( Vn ) );
}

}// namespace detail

//////////////////////////////////////////////////////////////////////
//
// TLR versions
//
//////////////////////////////////////////////////////////////////////

namespace tlr
{

namespace detail
{

//
// replace column basis of block M_ij by X and update basis
// of block row to [ V, X ]
//
template < typename value_t,
           typename approx_t >
void
extend_col_basis ( Hpro::TBlockMatrix< value_t > &        M,
                   matrix::uniform_lrmatrix< value_t > &  M_ij,
                   const uint                             i,
                   const uint                             j,
                   const blas::matrix< value_t > &        X,
                   const accuracy &                       acc,
                   const approx_t &                       approx )
{
    //
    // compute QR of X for norm computation later
    //

    auto  QX = blas::copy( X );
    auto  RX = blas::matrix< value_t >();

    blas::qr( QX, RX );
    
    // determine number of rows of matrix R below (sum of row ranks)
    size_t  nrows_Q    = 0;
    bool    have_other = false;
    
    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( matrix::is_uniform_lowrank( M_kj ) )
        {
            if ( k != i )
                have_other = true;
            
            nrows_Q += cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > )->row_rank();
        }// if
    }// for

    if ( ! have_other )
    {
        //
        // since there is no other low-rank block, just replace basis by (orthogonalized) X
        //
        
        auto  Sn = blas::prod( M_ij.coeff(), blas::adjoint( RX ) );

        M_ij.set_coeff_unsafe( std::move( Sn ) );
        const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( blas::copy( QX ) ) );
        return;
    }// if
    
    // extended column basis
    auto  V  = M_ij.col_cb().basis();
    auto  Ve = blas::join_row< value_t >( { V, X } );

    // io::matlab::write( V, "V" );
    // io::matlab::write( Ve, "Ve" );
    // io::matlab::write( X, "X" );
    
    // compute QR of row basis for each block in column and assemble
    // all results into common matrix Q
    auto    Qe  = blas::matrix< value_t >( nrows_Q, Ve.ncols() );
    size_t  pos = 0;

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  M_kj = M.block( k, j );
        
        if ( ! matrix::is_uniform_lowrank( M_kj ) )
            continue;
        
        const auto  R_kj   = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
        const auto  rank_k = R_kj->row_rank();
        auto        S_kj   = blas::copy( R_kj->coeff() );

        if ( k == i )
        {
            // R_kj = U_k S_kj X' and U_k is orthogonal,
            // therefore |R_kj| = |S_kj X'| = |S_kj RX' QX'| = |S_kj RX'|
            auto  SR_kj = blas::prod( S_kj, blas::adjoint( RX ) );

            // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( SR_kj ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( SR_kj ), S_kj );
                
            auto  Qe_k = blas::matrix< value_t >( Qe,
                                                  blas::range( pos, pos + rank_k-1 ),
                                                  blas::range( V.ncols(), Ve.ncols() - 1 ) );

            blas::copy( S_kj, Qe_k );
        }// if
        else
        {
            // R_kj = U_k S_kj V_j' and U_k/V_j are orthogonal,
            // therefore |R_kj| = |S_kj|
            // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( S_kj ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( S_kj ), S_kj );

            auto  Qe_k = blas::matrix< value_t >( Qe,
                                                  blas::range( pos, pos + rank_k-1 ),
                                                  blas::range( 0, V.ncols() - 1 ) );

            blas::copy( S_kj, Qe_k );
        }// else

        pos += rank_k;
    }// for

    // io::matlab::write( Qe, "Qe" );
    
    // compute QR of assembled matrix, and compute SVD of
    // product with extended column basis
    auto  R = blas::matrix< value_t >( Qe.ncols(), Qe.ncols() );
        
    blas::qr( Qe, R, false );

    // io::matlab::write( R, "R" );
    
    auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
    auto  Vn  = approx.column_basis( VeR, acc );

    // auto  Ss  = blas::vector< real_t >();

    // blas::svd( VeR, Ss );

    // // io::matlab::write( VeR, "VeR" );
    // // io::matlab::write( Ss, "Ss" );
    
    // const auto  rank   = acc.trunc_rank( Ss );
    // const auto  V_rank = blas::matrix< value_t >( VeR, blas::range::all, blas::range( 0, rank-1 ) );
    // auto        Vn     = blas::copy( V_rank );
    
    // io::matlab::write( Vn, "Vn" );

    //
    // transform coupling matrix for blocks in current block column as
    //
    //   (S_kj 0) TV  or  ( 0 S_ij ) TV
    //

    const auto  TV = blas::prod( blas::adjoint( Vn ), V );

    // io::matlab::write( TV, "TV" );

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if ( ! matrix::is_uniform_lowrank( B_kj ) )
            continue;
                    
        auto  R_kj = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        auto  S_kj = R_kj->coeff();

        if ( k == i )
        {
            auto  TX    = blas::prod( blas::adjoint( Vn ), X );
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TX ) );

            // auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            // auto  M1    = blas::prod( US1, blas::adjoint( X ) );
            // auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            // auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );

            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// if
        else
        {
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TV ) );

            // auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            // auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            // auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
            
            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;

            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// else
    }// for

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
}

//
// replace column basis of block M_ij by X and update basis
// of block row to [ V, X ]
// - use SVD of full block column to compute new basis
//
template < typename value_t >
void
extend_col_basis_ref ( Hpro::TBlockMatrix< value_t > &        M,
                       matrix::uniform_lrmatrix< value_t > &  M_ij,
                       const uint                             i,
                       const uint                             j,
                       const blas::matrix< value_t > &        X,
                       const accuracy &                       acc )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // construct full block column Xt, perform SVD and
    // use singular vectors for new basis (exact approach)
    //

    auto  Vn = blas::matrix< value_t >();

    {
        // determine dimensions of block column
        size_t  nrows = 0;
        size_t  ncols = 0;

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) )
            {
                nrows += M_kj->nrows();
                ncols  = M_kj->ncols();
            }// if
        }// for

        auto    Xt = blas::matrix< value_t >( ncols, nrows );  // adjoint (!)
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) )
            {
                auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                auto  U    = R_kj->row_cb().basis();
                auto  S    = R_kj->coeff();
                auto  V    = R_kj->col_cb().basis();

                auto  D_kj = blas::matrix< value_t >();

                if ( i == k )
                {
                    // replace V by X
                    auto  XS = blas::prod( X, blas::adjoint( S ) );

                    D_kj = std::move( blas::prod( XS, blas::adjoint( U ) ) );
                }// if
                else
                {
                    auto  VS = blas::prod( V, blas::adjoint( S ) );

                    D_kj = std::move( blas::prod( VS, blas::adjoint( U ) ) );
                }// else
                
                auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

                // std::cout << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_kj ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( D_kj ), D_kj );
                blas::copy( D_kj, Xt_k );

                pos += D_kj.ncols();
            }// if
        }// for

        // io::matlab::write( Xt, "Xt" );
        
        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Vn = std::move( blas::copy( V_rank ) );

        // io::matlab::write( Vn, "Vn" );
    }

    //
    // transform coupling matrix for blocks in current block column as
    //
    //   Vn'·V·S_kj' or Vn'·X·S_ij'
    //

    const auto  V  = M_ij.col_cb().basis();
    const auto  TV = blas::prod( blas::adjoint( Vn ), V );

    // io::matlab::write( TV, "TV" );

    for ( uint  k = 0; k < M.nblock_rows(); ++k )
    {
        auto  B_kj = M.block( k, j );
                    
        if ( ! matrix::is_uniform_lowrank( B_kj ) )
            continue;
                    
        auto  R_kj = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
        auto  S_kj = R_kj->coeff();

        if ( k == i )
        {
            // was: U_i S_ij X' -> U_i ( S_ij ( X' Vn ) ) Vn'
            auto  TX    = blas::prod( blas::adjoint( Vn ), X );
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TX ) );

            auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            auto  M1    = blas::prod( US1, blas::adjoint( X ) );
            auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );

            blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// if
        else
        {
            // was: U_k S_kj V_j' -> U_i ( S_ij ( V_j' Vn ) ) Vn' = U_i ( S_ij TV' ) Vn'
            auto  Sn_kj = blas::prod( S_kj, blas::adjoint( TV ) );

            auto  US1   = blas::prod( R_kj->row_cb().basis(), S_kj );
            auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );

            blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend col : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;

            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// else
    }// for

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.col_cb() )->set_basis( std::move( Vn ) );
}

//
// replace row basis for M_ij by W and update cluster basis of
// block row to [ U, W ]
//
template < typename value_t >
void
extend_row_basis ( Hpro::TBlockMatrix< value_t > &        M,
                   matrix::uniform_lrmatrix< value_t > &  M_ij,
                   const uint                             i,
                   const uint                             j,
                   const blas::matrix< value_t > &        W,
                   const accuracy &                       acc )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // compute QR of W for norm computation later
    //

    auto  QW = blas::copy( W );
    auto  RW = blas::matrix< value_t >();

    blas::qr( QW, RW );
    
    // determine number of rows of matrix R below (sum of column ranks)
    size_t  nrows_Q    = 0;
    bool    have_other = false;
    
    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( matrix::is_uniform_lowrank( M_ik ) )
        {
            if ( k != j )
                have_other = true;
            
            nrows_Q += cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > )->col_rank();
        }// if
    }// for

    if ( ! have_other )
    {
        //
        // since there is no other low-rank block, just replace basis by (orthogonalized) W and return
        //
        
        auto  Sn = blas::prod( RW, M_ij.coeff() );
        
        M_ij.set_coeff_unsafe( std::move( Sn ) );
        const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( blas::copy( QW ) ) );
        return;
    }// if
    
    // extended row basis
    auto  U  = M_ij.row_cb().basis();
    auto  Ue = blas::join_row< value_t >( { U, W } );

    // io::matlab::write( U, "U" );
    // io::matlab::write( Ue, "Ue" );
    // io::matlab::write( W, "W" );
    
    // compute QR of column basis for each block in row and assemble
    // all results into common matrix Q
    auto    Qe  = blas::matrix< value_t >( nrows_Q, Ue.ncols() );
    size_t  pos = 0;

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  M_ik = M.block( i, k );
        
        if ( ! matrix::is_uniform_lowrank( M_ik ) )
            continue;
        
        const auto  R_ik   = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
        const auto  rank_k = R_ik->col_rank();
        auto        S_ik   = blas::copy( R_ik->coeff() );

        // io::matlab::write( R_ik->row_cb().basis(), "Ui" );
        // io::matlab::write( R_ik->coeff(), "Sik" );
        // io::matlab::write( R_ik->col_cb().basis(), "Vk" );
        
        if ( k == j )
        {
            // R_ik = W S_ik V_k' and V_k is orthogonal,
            // therefore |R_ik| = |W S_ik| = |QW RW S_ik| = |RW S_ik|
            auto  RS_ik = blas::prod( RW, S_ik );

            // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( RS_ik ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( RS_ik ), S_ik );

            auto  Qe_k = blas::matrix< value_t >( Qe,
                                                  blas::range( pos, pos + rank_k-1 ),
                                                  blas::range( U.ncols(), Ue.ncols() - 1 ) );

            blas::copy( blas::adjoint( S_ik ), Qe_k );
        }// if
        else
        {
            // R_ik = U_i S_ik V_k' and U_i/V_k are orthogonal,
            // therefore |R_ik| = |S_ik|

            // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( S_ik ) ) << std::endl;
            blas::scale( value_t(1) / blas::norm_2( S_ik ), S_ik );
            
            auto  Qe_k = blas::matrix< value_t >( Qe,
                                                  blas::range( pos, pos + rank_k-1 ),
                                                  blas::range( 0, U.ncols() - 1 ) );

            blas::copy( blas::adjoint( S_ik ), Qe_k );
        }// else

        // io::matlab::write( Qe, "Qe" );
        
        pos += rank_k;
    }// for

    // io::matlab::write( Qe, "Qe" );
    
    // compute QR of assembled matrix, and compute SVD of
    // product with extended column basis
    auto  R = blas::matrix< value_t >( Qe.ncols(), Qe.ncols() );
        
    blas::qr( Qe, R, false );

    // io::matlab::write( R, "R" );
    
    auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
    auto  Ss  = blas::vector< real_t >();

    blas::svd( UeR, Ss );

    // io::matlab::write( UeR, "UeR" );
    // io::matlab::write( Ss, "Ss" );
    
    const auto  rank   = acc.trunc_rank( Ss );
    const auto  U_rank = blas::matrix< value_t >( UeR, blas::range::all, blas::range( 0, rank-1 ) );
    auto        Un     = blas::copy( U_rank );
    const auto  TU     = blas::prod( blas::adjoint( Un ), U );

    // io::matlab::write( Un, "Un" );
    
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   TU ⎛S_kj⎞  or  TU ⎛  0 ⎞
    //      ⎝ 0  ⎠         ⎝S_kj⎠
    //

    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  B_ik = M.block( i, k );
                    
        if ( ! matrix::is_uniform_lowrank( B_ik ) )
            continue;
                    
        auto  R_ik = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
        auto  S_ik = R_ik->coeff();

        if ( k == j )
        {
            auto  TW    = blas::prod( blas::adjoint( Un ), W );
            auto  Sn_ik = blas::prod( TW, S_ik );

            // auto  US1   = blas::prod( W, S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
        else
        {
            auto  Sn_ik = blas::prod( TU, S_ik );

            // auto  US1   = blas::prod( R_ik->row_cb().basis(), S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
            
            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// else
    }// for

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
}

//
// replace row basis for M_ij by W and update cluster basis of
// block row to [ U, W ]
// - use SVD of full block row matrix to compute new basis
//   (reference implementation)
//
template < typename value_t >
void
extend_row_basis_ref ( Hpro::TBlockMatrix< value_t > &        M,
                       matrix::uniform_lrmatrix< value_t > &  M_ij,
                       const uint                             i,
                       const uint                             j,
                       const blas::matrix< value_t > &        W,
                       const accuracy &                       acc )
{
    using  real_t = Hpro::real_type_t< value_t >;

    //
    // compute full block row, perform SVD and use
    // singular vectors as new row basis
    //

    const auto  U  = M_ij.row_cb().basis();
    auto        Un = blas::matrix< value_t >();

    {
        // determine dimensions of block row
        size_t  nrows = 0;
        size_t  ncols = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) )
            {
                nrows  = M_ik->nrows();
                ncols += M_ik->ncols();
            }// if
        }// for

        auto    Xt = blas::matrix< value_t >( nrows, ncols );
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) )
            {
                auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                auto  S    = R_ik->coeff();
                auto  V    = R_ik->col_cb().basis();
                auto  D_ik = blas::matrix< value_t >();

                if ( k == j )
                {
                    auto  WS = blas::prod( W, S );

                    D_ik = std::move( blas::prod( WS, blas::adjoint( V ) ) );
                }// if
                else
                {
                    auto  US = blas::prod( U, S );

                    D_ik = std::move( blas::prod( US, blas::adjoint( V ) ) );
                }// else

                // io::matlab::write( D_ik, "Dik" );
                
                // std::cout << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_2( D_ik ) ) << std::endl;
                blas::scale( value_t(1) / blas::norm_2( D_ik ), D_ik );
                
                auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

                blas::copy( D_ik, Xt_k );

                pos += D_ik.ncols();
            }// if
        }// for

        // io::matlab::write( Xt, "Xt" );
        
        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Un = std::move( blas::copy( U_rank ) );

        // io::matlab::write( Un, "Un" );
    }
    
    //
    // transform coupling matrix for blocks in current block column as
    //
    //   Un' U S_kj  or  Un' W S_ij
    //

    const auto  TU = blas::prod( blas::adjoint( Un ), U );
        
    for ( uint  k = 0; k < M.nblock_cols(); ++k )
    {
        auto  B_ik = M.block( i, k );
                    
        if ( ! matrix::is_uniform_lowrank( B_ik ) )
            continue;
                    
        auto  R_ik = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
        auto  S_ik = R_ik->coeff();

        if ( k == j )
        {
            auto  TW    = blas::prod( blas::adjoint( Un ), W );
            auto  Sn_ik = blas::prod( TW, S_ik );

            // auto  US1   = blas::prod( W, S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );

            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// if
        else
        {
            auto  Sn_ik = blas::prod( TU, S_ik );

            // auto  US1   = blas::prod( R_ik->row_cb().basis(), S_ik );
            // auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            // auto  US2   = blas::prod( Un, Sn_ik );
            // auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
            
            // blas::add( value_t(-1), M1, M2 );
            // std::cout << "    extend row : " << R_ik->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;

            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// else
    }// for

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & M_ij.row_cb() )->set_basis( std::move( Un ) );
}

//
// recompute i'th row basis and j'th column basis while replacing
// block M_ij by W·T·X', e.g., extend row basis to cover [ U, W ]
// and column basis for [ V, X ] with U,V being the current row/
// column basis
//
// ASSUMPTIONs
//  - W and X are orthogonal
//  - M_ij is a uniform lowrank matrix
//
template < typename value_t >
void
replace_row_col_basis ( Hpro::TBlockMatrix< value_t > &  M,
                        const uint                       i,
                        const uint                       j,
                        blas::matrix< value_t > &        W,
                        blas::matrix< value_t > &        T,
                        blas::matrix< value_t > &        X,
                        const accuracy &                 acc )
{
    using  real_t = Hpro::real_type_t< value_t >;

    // io::matlab::write( W, "W" );
    // io::matlab::write( T, "T" );
    // io::matlab::write( X, "X" );

    auto  M_ij = M.block( i, j );
    auto  R_ij = ptrcast( M_ij, matrix::uniform_lrmatrix< value_t > );

    //
    // compute new column basis
    //
    //   ⎛U_1 S_1 V'⎞' 
    //   ⎜U_2 S_2 V'⎟
    //   ⎜  ...     ⎟  = (V X) ⎛S_1'·U_1' S_2'·U_2' ... S_j'·U_j'   0  ⎞
    //   ⎜U_j S_j V'⎟          ⎝    0         0             0     T'·W'⎠
    //   ⎝W   T   X'⎠
    //                 = (V X) ⎛U_1·S_1  0 ⎞'
    //                         ⎜U_2·S_2  0 ⎟
    //                         ⎜     ...   ⎟
    //                         ⎜U_j·S_j  0 ⎟
    //                         ⎝   0    W·T⎠
    //
    // Since U_i and W are orthogonal, one can skip those for basis computation.
    // Compute QR factorization
    //
    //   Q·R = ⎛S_1  0⎞ = S
    //         ⎜S_2  0⎟
    //         ⎜ ...  ⎟
    //         ⎜S_j  0⎟
    //         ⎝ 0   T⎠
    //
    // and finally column basis of
    //
    //   (V X) R' = V_e R'
    //
    // Please note, that the S_i and T are scaled by the (spectral) norm of the
    // corresponding block U_i·S_i·V' and W·T·X'
    //
                                  
    auto  Vn = blas::matrix< value_t >();

    {
        // determine number of rows of matrix S below (sum of row ranks)
        size_t  nrows_S = T.nrows(); // known apriori
    
        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) && ( k != i ))
                nrows_S += cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > )->row_rank();
        }// for

        if ( nrows_S == T.nrows() )
        {
            //
            // since there is no other low-rank block, new basis is X
            //

            Vn = std::move( blas::copy( X ) );
        }// if
        else
        {
            //
            // otherwise compute new basis
            //
            
            auto  V  = R_ij->col_cb().basis();
            auto  Ve = blas::join_row< value_t >( { V, X } );
    
            // assemble normalized coefficient matrices into common matrix S
            auto    S   = blas::matrix< value_t >( nrows_S, Ve.ncols() );
            size_t  pos = 0;

            for ( uint  k = 0; k < M.nblock_rows(); ++k )
            {
                auto  M_kj = M.block( k, j );
        
                if ( ! matrix::is_uniform_lowrank( M_kj ) )
                    continue;

                if ( k == i )
                {
                    // R_kj = W T X' with W/X being orthogonal, hence |R_kj| = |T|
                    const auto  rank = T.nrows();
                    auto        S_kj = blas::copy( T );
                    
                    blas::scale( value_t(1) / blas::norm_2( T ), S_kj );
                
                    auto  S_k = blas::matrix< value_t >( S,
                                                         blas::range( pos, pos + rank-1 ),
                                                         blas::range( V.ncols(), Ve.ncols() - 1 ) );

                    blas::copy( S_kj, S_k );
                    pos += rank;
                }// if
                else
                {
                    // R_kj = U_k S_kj V_j' and U_k/V_j are orthogonal, hence |R_kj| = |S_kj|
                    const auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                    const auto  rank = R_kj->row_rank();
                    auto        S_kj = blas::copy( R_kj->coeff() );
                    
                    blas::scale( value_t(1) / blas::norm_2( S_kj ), S_kj );

                    auto  S_k = blas::matrix< value_t >( S,
                                                         blas::range( pos, pos + rank-1 ),
                                                         blas::range( 0, V.ncols() - 1 ) );

                    blas::copy( S_kj, S_k );
                    pos += rank;
                }// else
            }// for

            // compute QR of assembled matrix, and compute SVD of
            // product with extended column basis
            auto  R = blas::matrix< value_t >();

            blas::qr( S, R, false );

            auto  VeR = blas::prod( Ve, blas::adjoint( R ) );
            auto  Ss  = blas::vector< real_t >();

            blas::svd( VeR, Ss );

            const auto  rank   = acc.trunc_rank( Ss );
            const auto  V_rank = blas::matrix< value_t >( VeR, blas::range::all, blas::range( 0, rank-1 ) );

            Vn = std::move( blas::copy( V_rank ) );
    
            //
            // transform coupling matrix for blocks in current block column as
            //
            //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
            //

            const auto  TV = blas::prod( blas::adjoint( Vn ), V );

            for ( uint  k = 0; k < M.nblock_rows(); ++k )
            {
                auto  B_kj = M.block( k, j );
                    
                if ( ! matrix::is_uniform_lowrank( B_kj ) )
                    continue;
                    
                if ( k != i )
                {
                    auto  R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
                    auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

                    // // DEBUG {
                    // {
                    //     auto  US1   = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
                    //     auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
                    //     auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
                    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
                    //     blas::add( value_t(-1), M1, M2 );
                    //     std::cout << "    ext col/row : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
                    // }
                    // // DEBUG }

                    R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
                }// if
            }// for
        }// else
    }

    //
    // compute new row basis of
    //
    //   (U·S_1· V_1'  U·S_2·V_2'  ...  U·S_j·V_j'  W·T·X')
    //
    //    = (U W) ⎛S_1·V_1'  S_2·V_2' ... S_j·V_j'  0  ⎞
    //            ⎝   0         0            0     T·X'⎠
    //
    //    = (U W) ⎛V_1·S_1'  0  ⎞'
    //            ⎜V_2·S_2'  0  ⎟
    //            ⎜      ...    ⎟
    //            ⎜V_j·S_j'  0  ⎟
    //            ⎝   0     X·T'⎠
    //
    //    = (U W) ⎛⎛V_1              ⎞ ⎛S_1'  0 ⎞⎞'
    //            ⎜⎜    V_2          ⎟ ⎜S_2'  0 ⎟⎟
    //            ⎜⎜        ...      ⎟·⎜   ...  ⎟⎟
    //            ⎜⎜            V_j  ⎟ ⎜S_j'  0 ⎟⎟
    //            ⎝⎝                X⎠ ⎝ 0    T'⎠⎠
    //
    // Since V_i and X are orthogonal, one can skip those for basis computation.
    // Compute QR factorization
    //
    //   Q·R = ⎛S_1' 0 ⎞ = S
    //         ⎜S_2' 0 ⎟
    //         ⎜  ...  ⎟
    //         ⎜S_j' 0 ⎟
    //         ⎝ 0   T'⎠
    //
    // of which also Q is omitted, which leaves to compute the column basis of
    //
    //   (U W) R' = U_e R'
    //
    // The S_i and T are scaled by the (spectral) norm of the corresponding block
    // U_i·S_i·V' and W·T·X' to achieve the relative precision for all blocks.
    //

    auto  Un = blas::matrix< value_t >();

    {
        // determine number of rows of matrix S below (sum of column ranks)
        size_t  nrows_S = T.ncols();  // known apriori
    
        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) && ( k != j ))
                nrows_S += cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > )->col_rank();
        }// for

        if ( nrows_S == T.ncols() )
        {
            //
            // since there is no other low-rank block, new row basis is W
            //

            Un = std::move( blas::copy( W ) );
        }// if
        else
        {
            // extended row basis
            auto  U  = R_ij->row_cb().basis();
            auto  Ue = blas::join_row< value_t >( { U, W } );

            // compute QR of column basis for each block in row and assemble
            // all results into common matrix Q
            auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
            size_t  pos = 0;

            for ( uint  k = 0; k < M.nblock_cols(); ++k )
            {
                auto  M_ik = M.block( i, k );
        
                if ( ! matrix::is_uniform_lowrank( M_ik ) )
                    continue;
        
                if ( k == j )
                {
                    // R_ik = W T X' with W/X being orthogonal, hence |R_ik| = |T|
                    const auto  rank = T.ncols();
                    auto        S_ik = blas::copy( T );

                    blas::scale( value_t(1) / blas::norm_2( T ), S_ik );

                    auto  S_k = blas::matrix< value_t >( S,
                                                         blas::range( pos, pos + rank-1 ),
                                                         blas::range( U.ncols(), Ue.ncols() - 1 ) );

                    blas::copy( blas::adjoint( S_ik ), S_k );
                    pos += rank;
                }// if
                else
                {
                    // R_ik = U_i S_ik V_k' with U_i/V_k being orthogonal, hence |R_ik| = |S_ik|
                    const auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                    const auto  rank = R_ik->col_rank();
                    auto        S_ik = blas::copy( R_ik->coeff() );
                    
                    blas::scale( value_t(1) / blas::norm_2( S_ik ), S_ik );
            
                    auto  S_k = blas::matrix< value_t >( S,
                                                         blas::range( pos, pos + rank-1 ),
                                                         blas::range( 0, U.ncols() - 1 ) );

                    blas::copy( blas::adjoint( S_ik ), S_k );
                    pos += rank;
                }// else
            }// for

            // compute QR of assembled matrix, and compute SVD of
            // product with extended column basis
            auto  R = blas::matrix< value_t >();
        
            blas::qr( S, R, false );

            auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
            auto  Ss  = blas::vector< real_t >();

            blas::svd( UeR, Ss );

            const auto  rank   = acc.trunc_rank( Ss );
            const auto  U_rank = blas::matrix< value_t >( UeR, blas::range::all, blas::range( 0, rank-1 ) );

            Un = std::move( blas::copy( U_rank ) );

            //
            // transform coupling matrix for blocks in current block column as
            //
            //   Un'·U·S_i = TU·S_i  with TU = Un'·U
            //

            const auto  TU = blas::prod( blas::adjoint( Un ), U );

            for ( uint  k = 0; k < M.nblock_cols(); ++k )
            {
                auto  B_ik = M.block( i, k );
                    
                if ( ! matrix::is_uniform_lowrank( B_ik ) )
                    continue;
                    
                if ( k != j )
                {
                    auto  R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
                    auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

                    // // DEBUG {
                    // {
                    //     auto  US1   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
                    //     auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
                    //     auto  US2   = blas::prod( Un, Sn_ik );
                    //     auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
                        
                    //     blas::add( value_t(-1), M1, M2 );
                    //     std::cout << "    ext row/col : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
                    // }
                    // // DEBUG }

                    R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
                }// if
            }// for
        }// else
    }

    //
    // compute coupling of M_ij as Un' W T X' Vn
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  TX = blas::prod( blas::adjoint( Vn ), X );
    auto  S1 = blas::prod( TW, T );
    auto  Sn = blas::prod( S1, blas::adjoint( TX ) );

    // // DEBUG {
    // io::matlab::write( Un, "Un" );
    // io::matlab::write( Sn, "Sn" );
    // io::matlab::write( Vn, "Vn" );
    
    // {
    //     auto  US1   = blas::prod( W, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( X ) );
    //     auto  US2   = blas::prod( Un, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext    /    : " << R_ij->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    R_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & R_ij->col_cb() )->set_basis( std::move( Vn ) );
    const_cast< matrix::shared_cluster_basis< value_t > * >( & R_ij->row_cb() )->set_basis( std::move( Un ) );
}

template < typename value_t >
void
replace_row_col_basis_ref ( Hpro::TBlockMatrix< value_t > &  M,
                            const uint                       i,
                            const uint                       j,
                            blas::matrix< value_t > &        W,
                            blas::matrix< value_t > &        T,
                            blas::matrix< value_t > &        X,
                            const accuracy &                 acc )
{
    using  real_t = Hpro::real_type_t< value_t >;

    auto  M_ij = M.block( i, j );
    auto  R_ij = ptrcast( M_ij, matrix::uniform_lrmatrix< value_t > );

    //
    // compute new column basis of block column
    //
    //   ⎛U_1 S_1 V'⎞
    //   ⎜U_2 S_2 V'⎟
    //   ⎜  ...     ⎟
    //   ⎜U_j S_j V'⎟
    //   ⎝W   T   X'⎠
    //
                                  
    auto  Vn = blas::matrix< value_t >();

    {
        // determine dimensions of block column
        size_t  nrows = W.nrows();
        size_t  ncols = X.nrows();

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) && ( k != i ))
            {
                nrows += M_kj->nrows();
                ncols  = M_kj->ncols();
            }// if
        }// for

        auto    Xt = blas::matrix< value_t >( ncols, nrows );  // adjoint (!)
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  M_kj = M.block( k, j );
        
            if ( matrix::is_uniform_lowrank( M_kj ) )
            {
                auto  D_kj = blas::matrix< value_t >();

                if ( i == k )
                {
                    auto  XT = blas::prod( X, blas::adjoint( T ) );

                    D_kj = std::move( blas::prod( XT, blas::adjoint( W ) ) );
                }// if
                else
                {
                    auto  R_kj = cptrcast( M_kj, matrix::uniform_lrmatrix< value_t > );
                    auto  VS   = blas::prod( R_kj->col_cb().basis(), blas::adjoint( R_kj->coeff() ) );

                    D_kj = std::move( blas::prod( VS, blas::adjoint( R_kj->row_cb().basis() ) ) );
                }// else
                
                auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_kj.ncols() - 1 ) );

                blas::scale( value_t(1) / blas::norm_2( D_kj ), D_kj );
                blas::copy( D_kj, Xt_k );

                pos += D_kj.ncols();
            }// if
        }// for

        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  V_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Vn = std::move( blas::copy( V_rank ) );

        //
        // transform coupling matrix for blocks in current block column as
        //
        //   S_kj V' Vn = S_kj·TV' with TV = Vn'·V
        //

        const auto  V  = R_ij->col_cb().basis();
        const auto  TV = blas::prod( blas::adjoint( Vn ), V );

        for ( uint  k = 0; k < M.nblock_rows(); ++k )
        {
            auto  B_kj = M.block( k, j );
            
            if ( ! matrix::is_uniform_lowrank( B_kj ) || ( k == i ))
                continue;
            
            auto  R_kj  = ptrcast( B_kj, matrix::uniform_lrmatrix< value_t > );
            auto  Sn_kj = blas::prod( R_kj->coeff(), blas::adjoint( TV ) );

            // // DEBUG {
            // {
            //     auto  US1   = blas::prod( R_kj->row_cb().basis(), R_kj->coeff() );
            //     auto  M1    = blas::prod( US1, blas::adjoint( R_kj->col_cb().basis() ) );
            //     auto  US2   = blas::prod( R_kj->row_cb().basis(), Sn_kj );
            //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
            
            //     blas::add( value_t(-1), M1, M2 );
            //     std::cout << "    ext col/row : " << R_kj->id() << " : " << boost::format( "%.4e" ) % ( blas::norm_F( M2 ) / blas::norm_F( M1 ) ) << std::endl;
            // }
            // // DEBUG }
            
            R_kj->set_coeff_unsafe( std::move( Sn_kj ) );
        }// for
    }

    //
    // compute new row basis of block row
    //
    //   (U·S_1· V_1'  U·S_2·V_2'  ...  U·S_j·V_j'  W·T·X')
    //

    auto  Un = blas::matrix< value_t >();

    {
        // determine dimensions of block row
        size_t  nrows = 0;
        size_t  ncols = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) )
            {
                nrows  = M_ik->nrows();
                ncols += M_ik->ncols();
            }// if
        }// for

        auto    Xt = blas::matrix< value_t >( nrows, ncols );
        size_t  pos = 0;

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  M_ik = M.block( i, k );
        
            if ( matrix::is_uniform_lowrank( M_ik ) )
            {
                auto  D_ik = blas::matrix< value_t >();

                if ( k == j )
                {
                    auto  WT = blas::prod( W, T );

                    D_ik = std::move( blas::prod( WT, blas::adjoint( X ) ) );
                }// if
                else
                {
                    auto  R_ik = cptrcast( M_ik, matrix::uniform_lrmatrix< value_t > );
                    auto  US   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );

                    D_ik = std::move( blas::prod( US, blas::adjoint( R_ik->col_cb().basis() ) ) );
                }// else

                blas::scale( value_t(1) / blas::norm_2( D_ik ), D_ik );
                
                auto  Xt_k = blas::matrix< value_t >( Xt, blas::range::all, blas::range( pos, pos + D_ik.ncols() - 1 ) );

                blas::copy( D_ik, Xt_k );

                pos += D_ik.ncols();
            }// if
        }// for

        auto  Ss = blas::vector< real_t >();

        blas::svd( Xt, Ss );

        const auto  rank   = acc.trunc_rank( Ss );
        const auto  U_rank = blas::matrix< value_t >( Xt, blas::range::all, blas::range( 0, rank-1 ) );

        Un = std::move( blas::copy( U_rank ) );

        //
        // transform coupling matrix for blocks in current block column as
        //
        //   Un'·U·S_i = TU·S_i  with TU = Un'·U
        //
        
        const auto  U  = R_ij->row_cb().basis();
        const auto  TU = blas::prod( blas::adjoint( Un ), U );

        for ( uint  k = 0; k < M.nblock_cols(); ++k )
        {
            auto  B_ik = M.block( i, k );
                    
            if ( ! matrix::is_uniform_lowrank( B_ik ) || ( k == j ))
                continue;
                    
            auto  R_ik  = ptrcast( B_ik, matrix::uniform_lrmatrix< value_t > );
            auto  Sn_ik = blas::prod( TU, R_ik->coeff() );

            // // DEBUG {
            // {
            //     auto  US1   = blas::prod( R_ik->row_cb().basis(), R_ik->coeff() );
            //     auto  M1    = blas::prod( US1, blas::adjoint( R_ik->col_cb().basis() ) );
            //     auto  US2   = blas::prod( Un, Sn_ik );
            //     auto  M2    = blas::prod( US2, blas::adjoint( R_ik->col_cb().basis() ) );
            
            //     blas::add( value_t(-1), M1, M2 );
            //     std::cout << "    ext row/col : " << R_ik->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
            // }
            // // DEBUG }
            
            R_ik->set_coeff_unsafe( std::move( Sn_ik ) );
        }// for
    }

    //
    // compute coupling of M_ij as Un' W T X' Vn
    //

    auto  TW = blas::prod( blas::adjoint( Un ), W );
    auto  TX = blas::prod( blas::adjoint( Vn ), X );
    auto  S1 = blas::prod( TW, T );
    auto  Sn = blas::prod( S1, blas::adjoint( TX ) );

    // // DEBUG {
    // {
    //     auto  US1   = blas::prod( W, T );
    //     auto  M1    = blas::prod( US1, blas::adjoint( X ) );
    //     auto  US2   = blas::prod( Un, Sn );
    //     auto  M2    = blas::prod( US2, blas::adjoint( Vn ) );
                        
    //     blas::add( value_t(-1), M1, M2 );
    //     std::cout << "    ext    /    : " << R_ij->id() << " : " << blas::norm_F( M2 ) / blas::norm_F( M1 ) << std::endl;
    // }
    // // DEBUG }
    
    R_ij->set_coeff_unsafe( std::move( Sn ) );

    //
    // finally adjust cluster basis
    //

    const_cast< matrix::shared_cluster_basis< value_t > * >( & R_ij->col_cb() )->set_basis( std::move( Vn ) );
    const_cast< matrix::shared_cluster_basis< value_t > * >( & R_ij->row_cb() )->set_basis( std::move( Un ) );
}

}}}}// namespace hlr::uniform::tlr::detail

#endif // __HLR_ARITH_DETAIL_UNIFORM_BASIS_HH
