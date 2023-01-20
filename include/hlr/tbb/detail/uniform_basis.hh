#ifndef __HLR_TBB_DETAIL_UNIFORM_BASIS_HH
#define __HLR_TBB_DETAIL_UNIFORM_BASIS_HH
//
// Project     : HLib
// Module      : arith/uniform_basis
// Description : functions for cluster basis manipulation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <algorithm>

#include <tbb/parallel_for.h>

#include <hlr/arith/blas.hh>
#include <hlr/arith/detail/uniform_basis.hh>
#include <hlr/matrix/cluster_basis.hh>

namespace hlr { namespace tbb { namespace uniform { namespace detail {

using hlr::matrix::cluster_basis;
using hlr::matrix::uniform_lrmatrix;
using hlr::uniform::is_matrix_map_t;

//
// extend row basis <cb> by block W·T·X' (X is not needed for computation)
//
// - identical to implementation in "arith/detail/uniform.hh" but thread-safe,
//   hence, for details look into original code
//
template < typename value_t,
           typename basisapx_t >
blas::matrix< value_t >
compute_extended_basis ( const cluster_basis< value_t > &  cb,
                         const blas::matrix< value_t > &   W,
                         const blas::matrix< value_t > &   T,
                         const hpro::TTruncAcc &           acc,
                         const basisapx_t &                basisapx,
                         is_matrix_map_t< value_t > &      matmap,
                         std::mutex &                      matmapmtx,
                         const matop_t                     op,
                         uniform_lrmatrix< value_t > *     M = nullptr )
{
    using  real_t = hpro::real_type_t< value_t >;

    // zero basis implies empty matrix list
    if ( cb.basis().ncols() == 0 )
        return std::move( blas::copy( W ) );
            
    //
    // copy uniform matrices for basis to local list for minimal blocking
    //
        
    auto  uni_mats = typename is_matrix_map_t< value_t >::mapped_type();

    {
        auto  lock = std::scoped_lock( matmapmtx );

        HLR_ASSERT( matmap.find( cb.is() ) != matmap.end() );
            
        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( M_i != M )
                uni_mats.push_back( M_i );
        }// for
    }
    
    //
    // collect scaled coupling matrices and filter out zero couplings
    //

    auto    couplings = std::list< blas::matrix< value_t > >();
    size_t  nrows_S   = T.ncols();
    auto    cmtx      = std::mutex();

    ::tbb::parallel_for_each(
        uni_mats,
        [&] ( auto  M_i )
        {
            const auto  R_i = cptrcast( M_i, uniform_lrmatrix< value_t > );
            auto        S_i = blas::matrix< value_t >();
                        
            {
                auto  lock = std::scoped_lock( M_i->mutex() );

                S_i = std::move( blas::copy( blas::mat_view( op, R_i->coeff() ) ) );
            }
                        
            HLR_ASSERT( S_i.ncols() == cb.basis().ncols() );
            
            const auto  norm = norm::spectral( S_i );
                        
            if ( norm != real_t(0) )
            {
                blas::scale( value_t(1) / norm, S_i );

                {
                    auto  lock = std::scoped_lock( cmtx );
                    
                    nrows_S += S_i.nrows();
                    couplings.push_back( std::move( S_i ) );
                }
            }// if
        } );

    //
    // assemble all scaled coupling matrices into joined matrix
    //

    auto    U   = cb.basis();
    auto    Ue  = blas::join_row< value_t >( { U, W } );
    auto    S   = blas::matrix< value_t >( nrows_S, Ue.ncols() );
    size_t  pos = 0;
            
    for ( auto  S_i : couplings )
    {
        HLR_ASSERT( pos + S_i.nrows() <= S.nrows() );
        HLR_ASSERT( S_i.ncols() == U.ncols() );
            
        auto  S_sub = blas::matrix< value_t >( S,
                                               blas::range( pos, pos + S_i.nrows()-1 ),
                                               blas::range( 0, U.ncols() - 1 ) );
                        
        blas::copy( S_i, S_sub );
        pos += S_i.nrows();
    }// for

    //
    // add part from W·T·X'
    //
        
    auto  S_i  = blas::copy( blas::mat_view( op, T ) );
    auto  norm = norm::spectral( T );
            
    if ( norm != real_t(0) )
        blas::scale( value_t(1) / norm, S_i );
            
    HLR_ASSERT( pos + S_i.nrows() <= S.nrows() );
    HLR_ASSERT( S_i.ncols() == Ue.ncols() - U.ncols() );
        
    auto  S_sub = blas::matrix< value_t >( S,
                                           blas::range( pos, pos + S_i.nrows()-1 ),
                                           blas::range( U.ncols(), Ue.ncols() - 1 ) );
            
    blas::copy( S_i, S_sub );
        
    //
    // form product Ue·S and compute column basis
    //
            
    auto  R = blas::matrix< value_t >();
        
    blas::qr( S, R, false );

    auto  UeR = blas::prod( Ue, blas::adjoint( R ) );
    auto  Un  = basisapx.column_basis( UeR, acc );

    return  Un;
}

//
// update coupling matrices for all blocks sharing basis <cb> to new basis <Un>
//
template < typename value_t >
void
update_coupling ( const cluster_basis< value_t > &  cb,
                  const blas::matrix< value_t > &   Un,
                  is_matrix_map_t< value_t > &      matmap,
                  std::mutex &                      matmapmtx,
                  const bool                        cols,
                  uniform_lrmatrix< value_t > *     M = nullptr )
{
    if ( cb.basis().ncols() == 0 )
        return;
            
    auto  uni_mats = typename is_matrix_map_t< value_t >::mapped_type();

    {
        auto  lock = std::scoped_lock( matmapmtx );
                    
        HLR_ASSERT( matmap.find( cb.is() ) != matmap.end() );
            
        for ( auto  M_i : matmap.at( cb.is() ) )
        {
            if ( M_i != M )
                uni_mats.push_back( M_i );
        }// for
    }
        
    auto  U  = cb.basis();
    auto  TU = blas::prod( blas::adjoint( Un ), U );

    ::tbb::parallel_for_each(
        uni_mats,
        [&] ( auto  M_i )
        {
            auto  lock = std::scoped_lock( M_i->mutex() );
            auto  R_i  = ptrcast( M_i, uniform_lrmatrix< value_t > );
            auto  S_i  = ( cols
                           ? blas::prod( R_i->coeff(), blas::adjoint( TU ) )
                           : blas::prod( TU, R_i->coeff() ) );

            R_i->set_coeff_unsafe( std::move( S_i ) );
        } );
}

}}}}// namespace hlr::tbb::uniform::detail

#endif // __HLR_TBB_DETAIL_UNIFORM_BASIS_HH
