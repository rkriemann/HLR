#ifndef __HLR_SEQ_COMPRESS_HH
#define __HLR_SEQ_COMPRESS_HH
//
// Project     : HLib
// Module      : seq/compress
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/matrix/compress.hh>

namespace hlr { namespace seq { namespace matrix {

using hlr::matrix::compress_replace;

//
// build H-matrix from given dense matrix without reording rows/columns
// starting lowrank approximation at blocks of size ntile × ntile and
// then trying to agglomorate low-rank blocks up to the root
//
namespace detail
{

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< hpro::TMatrix >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const hpro::TTruncAcc &          acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;
    
    if ( std::min( D.nrows(), D.ncols() ) <= ntile )
    {
        //
        // build leaf
        //
        // Apply low-rank approximation and compare memory consumption
        // with dense representation. If low-rank format uses less memory
        // the leaf is represented as low-rank (considered admissible).
        // Otherwise a dense representation is used.
        //

        if ( ! acc.is_exact() )
        {
            auto  Dc       = blas::copy( D );  // do not modify D (!)
            auto  [ U, V ] = approx( Dc, acc( rowis, colis ) );
            
            if ( U.byte_size() + V.byte_size() < Dc.byte_size() )
            {
                return std::make_unique< lrmatrix >( rowis, colis, std::move( U ), std::move( V ) );
            }// if
        }// if

        return std::make_unique< dense_matrix >( rowis, colis, std::move( blas::copy( D ) ) );
    }// if
    else
    {
        //
        // Recursion
        //
        // If all sub blocks are low-rank, an agglomorated low-rank matrix of all sub-blocks
        // is constructed. If the memory of this low-rank matrix is smaller compared to the
        // combined memory of the sub-block, it is kept. Otherwise a block matrix with the
        // already constructed sub-blocks is created.
        //

        const auto  mid_row = ( rowis.first() + rowis.last() + 1 ) / 2;
        const auto  mid_col = ( colis.first() + colis.last() + 1 ) / 2;

        indexset    sub_rowis[2] = { indexset( rowis.first(), mid_row-1 ), indexset( mid_row, rowis.last() ) };
        indexset    sub_colis[2] = { indexset( colis.first(), mid_col-1 ), indexset( mid_col, colis.last() ) };
        auto        sub_D        = tensor2< std::unique_ptr< hpro::TMatrix > >( 2, 2 );
        bool        all_lowrank  = true;
        bool        all_dense    = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                       sub_colis[j] - colis.first() );
                
                sub_D(i,j) = compress( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zconf );
                
                HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );

                if ( ! is_generic_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;

                if ( ! is_generic_dense( *sub_D(i,j) ) )
                    all_dense = false;
            }// for
        }// for

        if ( all_lowrank )
        {
            //
            // construct larger lowrank matrix out of smaller sub blocks
            //

            // compute initial total rank
            uint  rank_sum = 0;

            for ( uint  i = 0; i < 2; ++i )
                for ( uint  j = 0; j < 2; ++j )
                    rank_sum += ptrcast( sub_D(i,j).get(), lrmatrix )->rank();

            // copy sub block data into global structure
            auto    U    = blas::matrix< value_t >( rowis.size(), rank_sum );
            auto    V    = blas::matrix< value_t >( colis.size(), rank_sum );
            auto    pos  = 0; // pointer to next free space in U/V
            size_t  smem = 0; // holds memory of sub blocks
            
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    auto  Rij   = ptrcast( sub_D(i,j).get(), lrmatrix );
                    auto  Uij   = Rij->U< value_t >();
                    auto  Vij   = Rij->V< value_t >();
                    auto  U_sub = U( sub_rowis[i] - rowis.first(), blas::range( pos, pos + Uij.ncols() ) );
                    auto  V_sub = V( sub_colis[j] - colis.first(), blas::range( pos, pos + Uij.ncols() ) );

                    blas::copy( Uij, U_sub );
                    blas::copy( Vij, V_sub );

                    pos  += Uij.ncols();
                    smem += Uij.byte_size() + Vij.byte_size();
                }// for
            }// for

            //
            // try to approximate again in lowrank format and use
            // approximation if it uses less memory 
            //
            
            auto  [ W, X ] = approx( U, V, acc( rowis, colis ) );

            if ( W.byte_size() + X.byte_size() < smem )
            {
                return std::make_unique< lrmatrix >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            return std::make_unique< dense_matrix >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if

        //
        // either not all low-rank or memory gets larger: construct block matrix
        //

        auto  B = std::make_unique< hpro::TBlockMatrix >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zconf ) )
                {
                    if ( is_generic_lowrank( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), lrmatrix )->compress( *zconf );
                
                    if ( is_generic_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix )->compress( *zconf );
                }// if
                
                B->set_block( i, j, sub_D(i,j).release() );
            }// for
        }// for

        return B;
    }// else
}

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< hpro::TMatrix >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const hpro::TTruncAcc &          acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    //
    // compute lowrank approximation
    //

    {
        auto  Dc       = blas::copy( D );  // do not modify D (!)
        auto  acc_loc  = acc( rowis, colis );
        auto  max_rank = std::min( Dc.nrows(), Dc.ncols() ) / 8;

        acc_loc.set_max_rank( max_rank );
        
        auto  [ U, V ] = approx( Dc, acc_loc );
            
        if ( U.ncols() < max_rank )
        {
            auto  R = std::make_unique< lrmatrix >( rowis, colis, std::move( U ), std::move( V ) );

            if ( ! is_null( zconf ) )
                ptrcast( R.get(), lrmatrix )->compress( *zconf );

            return R;
        }// if
    }
    
    if ( std::min( D.nrows(), D.ncols() ) <= ntile )
    {
        return std::make_unique< dense_matrix >( rowis, colis, std::move( blas::copy( D ) ) );
    }// if
    else
    {
        //
        // Recursion
        //
        // If all sub blocks are low-rank, an agglomorated low-rank matrix of all sub-blocks
        // is constructed. If the memory of this low-rank matrix is smaller compared to the
        // combined memory of the sub-block, it is kept. Otherwise a block matrix with the
        // already constructed sub-blocks is created.
        //

        const auto  mid_row = ( rowis.first() + rowis.last() + 1 ) / 2;
        const auto  mid_col = ( colis.first() + colis.last() + 1 ) / 2;

        indexset    sub_rowis[2] = { indexset( rowis.first(), mid_row-1 ), indexset( mid_row, rowis.last() ) };
        indexset    sub_colis[2] = { indexset( colis.first(), mid_col-1 ), indexset( mid_col, colis.last() ) };
        auto        sub_D        = tensor2< std::unique_ptr< hpro::TMatrix > >( 2, 2 );
        bool        all_dense    = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                       sub_colis[j] - colis.first() );
                
                sub_D(i,j) = compress_topdown( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zconf );
                
                HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );

                if ( ! is_generic_dense( *sub_D(i,j) ) )
                    all_dense = false;
            }// for
        }// for

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            return std::make_unique< dense_matrix >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if

        //
        // either not all low-rank or memory gets larger: construct block matrix
        //

        auto  B = std::make_unique< hpro::TBlockMatrix >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zconf ) )
                {
                    if ( is_generic_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix )->compress( *zconf );
                }// if
                
                B->set_block( i, j, sub_D(i,j).release() );
            }// for
        }// for

        return B;
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< hpro::TMatrix >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const hpro::TTruncAcc &          acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress( rowis, colis, D, acc, approx, ntile, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle SZ/ZFP compression for global lowrank/dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_generic_lowrank( *M ) )
            ptrcast( M.get(), lrmatrix )->compress( *zconf );
                
        if ( is_generic_dense( *M ) )
            ptrcast( M.get(), dense_matrix )->compress( *zconf );
    }// if

    return M;
}

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< hpro::TMatrix >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const hpro::TTruncAcc &          acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress_topdown( rowis, colis, D, acc, approx, ntile, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle SZ/ZFP compression for global dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_generic_dense( *M ) )
            ptrcast( M.get(), dense_matrix )->compress( *zconf );
    }// if

    return M;
}

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_COMPRESS_HH
