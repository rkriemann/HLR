#ifndef __HLR_SEQ_DETAIL_MATRIX_HH
#define __HLR_SEQ_DETAIL_MATRIX_HH
//
// Project     : HLib
// File        : matrix.hh
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <unordered_map>

#include <hlr/arith/detail/uniform.hh>

namespace hlr { namespace seq { namespace matrix { namespace detail {

namespace hpro = HLIB;

using namespace hlr::matrix;

using  uniform_map_t = std::unordered_map< indexset, std::list< hpro::TMatrix * >, indexset_hash >;

template < typename coeff_t,
           typename lrapx_t,
           typename basisapx_t >
std::unique_ptr< hpro::TMatrix >
build_uniform_rec ( const hpro::TBlockCluster *                   bct,
                    const coeff_t &                               coeff,
                    const lrapx_t &                               lrapx,
                    const basisapx_t &                            basisapx,
                    const hpro::TTruncAcc &                       acc,
                    cluster_basis< typename coeff_t::value_t > &  rowcb,
                    cluster_basis< typename coeff_t::value_t > &  colcb,
                    uniform_map_t &                               rowmap,
                    uniform_map_t &                               colmap )
{
    using value_t = typename coeff_t::value_t;

    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< hpro::TMatrix >  M;
    
    if ( bct->is_leaf() )
    {
        if ( bct->is_adm() )
        {
            M = std::unique_ptr< hpro::TMatrix >( lrapx.build( bct, acc ) );

            if ( is_lowrank( *M ) )
            {
                //
                // form U·V' = W·T·X' with orthogonal W/X
                //

                auto  R  = ptrcast( M.get(), hpro::TRkMatrix );
                auto  W  = blas::mat_U< value_t >( R );
                auto  X  = blas::mat_V< value_t >( R );
                auto  Rw = blas::matrix< value_t >();
                auto  Rx = blas::matrix< value_t >();

                blas::qr( W, Rw );
                blas::qr( X, Rx );

                auto  T  = blas::prod( Rw, blas::adjoint( Rx ) );
                
                //
                // update cluster bases
                //

                auto  Un = hlr::uniform::detail::compute_extended_row_basis( rowcb, W, T, acc, basisapx, rowmap );
                auto  Vn = hlr::uniform::detail::compute_extended_col_basis( colcb, T, X, acc, basisapx, colmap );

                hlr::uniform::detail::update_row_coupling( rowcb, Un, rowmap );
                hlr::uniform::detail::update_col_coupling( colcb, Vn, colmap );

                //
                // compute coupling matrix with new row/col bases Un/Vn
                //

                auto  UW = blas::prod( blas::adjoint( Un ), W );
                auto  VX = blas::prod( blas::adjoint( Vn ), X );
                auto  T1 = blas::prod( UW, T );
                auto  S  = blas::prod( T1, blas::adjoint( VX ) );

                // update bases in cluster bases objects (only now since Un/Vn are used before)
                rowcb.set_basis( std::move( Un ) );
                colcb.set_basis( std::move( Vn ) );
                
                auto  RU = std::make_unique< uniform_lrmatrix< value_t > >( R->row_is(), R->col_is(), rowcb, colcb, std::move( S ) );

                rowmap[ RU->row_is() ].push_back( RU.get() );
                colmap[ RU->col_is() ].push_back( RU.get() );

                // {// DEBUG {
                //     auto  M1 = blas::prod( U, blas::adjoint( V ) );
                //     auto  T2 = blas::prod( W, T );
                //     auto  M2 = blas::prod( T2, blas::adjoint( X ) );
                //     auto  T3 = blas::prod( rowcb.basis(), RU->coeff() );
                //     auto  M3 = blas::prod( T3, blas::adjoint( colcb.basis() ) );

                //     blas::add( value_t(-1), M1, M2 );
                //     blas::add( value_t(-1), M1, M3 );

                //     std::cout << blas::norm_F( M2 ) / blas::norm_F( M1 ) << "    "
                //               << blas::norm_F( M3 ) / blas::norm_F( M1 ) << std::endl;
                // }// DEBUG }
                
                M = std::move( RU );
            }// if
        }// if
        else
        {
            M = coeff.build( bct->is().row_is(), bct->is().col_is() );
        }// else
    }// if
    else
    {
        M = std::make_unique< hpro::TBlockMatrix >( bct );
        
        auto  B = ptrcast( M.get(), hpro::TBlockMatrix );

        // make sure, block structure is correct
        if (( B->nblock_rows() != bct->nrows() ) ||
            ( B->nblock_cols() != bct->ncols() ))
            B->set_block_struct( bct->nrows(), bct->ncols() );

        // recurse
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );

            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( bct->son( i, j ) ) )
                {
                    if ( is_null( rowcb_i ) )
                    {
                        rowcb_i = new cluster_basis< value_t >( bct->son( i, j )->is().row_is() );
                        rowcb_i->set_nsons( bct->son( i, j )->rowcl()->nsons() );
                        rowcb.set_son( i, rowcb_i );
                    }// if
            
                    if ( is_null( colcb_j ) )
                    {
                        colcb_j = new cluster_basis< value_t >( bct->son( i, j )->is().col_is() );
                        colcb_j->set_nsons( bct->son( i, j )->colcl()->nsons() );
                        colcb.set_son( j, colcb_j );
                    }// if
            
                    auto  B_ij = build_uniform_rec( bct->son( i, j ), coeff, lrapx, basisapx, acc, *rowcb_i, *colcb_j, rowmap, colmap );

                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        // make value type consistent in block matrix and sub blocks
        B->adjust_value_type();
    }// else

    // M->set_cluster_force( bct );
    M->set_id( bct->id() );
    M->set_procs( bct->procs() );

    return M;
}

}}}}// namespace hlr::seq::detail::matrix

#endif // __HLR_SEQ_DETAIL_MATRIX_HH
