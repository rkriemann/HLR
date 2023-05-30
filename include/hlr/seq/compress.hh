#ifndef __HLR_SEQ_COMPRESS_HH
#define __HLR_SEQ_COMPRESS_HH
//
// Project     : HLR
// Module      : seq/compress
// Description : matrix related functionality
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/matrix/compress.hh>
#include <hlr/matrix/shared_cluster_basis.hh>
#include <hlr/matrix/nested_cluster_basis.hh>
#include <hlr/tensor/compress.hh>
#include <hlr/approx/aca.hh>
#include <hlr/utils/io.hh>

namespace hlr { namespace seq {

namespace matrix
{

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
std::unique_ptr< Hpro::TMatrix< value_t > >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const Hpro::TTruncAcc &          acc,
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
                return std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( U ), std::move( V ) );
            }// if
        }// if

        return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
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
        auto        sub_D        = tensor2< std::unique_ptr< Hpro::TMatrix< value_t > > >( 2, 2 );
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

                if ( ! is_compressible_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;

                if ( ! is_compressible_dense( *sub_D(i,j) ) )
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
                    rank_sum += ptrcast( sub_D(i,j).get(), lrmatrix< value_t > )->rank();

            // copy sub block data into global structure
            auto    U    = blas::matrix< value_t >( rowis.size(), rank_sum );
            auto    V    = blas::matrix< value_t >( colis.size(), rank_sum );
            auto    pos  = 0; // pointer to next free space in U/V
            size_t  smem = 0; // holds memory of sub blocks
            
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    auto  Rij   = ptrcast( sub_D(i,j).get(), lrmatrix< value_t > );
                    auto  Uij   = Rij->U();
                    auto  Vij   = Rij->V();
                    auto  U_sub = U( sub_rowis[i] - rowis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );
                    auto  V_sub = V( sub_colis[j] - colis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );

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
                return std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if

        //
        // either not all low-rank or memory gets larger: construct block matrix
        //

        auto  B = std::make_unique< Hpro::TBlockMatrix< value_t > >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zconf ) )
                {
                    if ( is_compressible_lowrank( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), lrmatrix< value_t > )->compress( *zconf );
                
                    if ( is_compressible_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix< value_t > )->compress( *zconf );
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
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const Hpro::TTruncAcc &          acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const size_t                     max_rank,
                   const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    //
    // compute lowrank approximation
    //

    //
    // compute lowrank approximation
    //

    const bool  is_leaf = ( std::min( D.nrows(), D.ncols() ) <= ntile );
    
    {
        auto  Dc      = blas::copy( D );  // do not modify D (!)
        auto  acc_loc = acc( rowis, colis );

        if ( is_leaf )
        {
            auto  [ U, V ] = approx( Dc, acc_loc );
            
            if ( U.ncols() < std::min( D.nrows(), D.ncols() ) / 2 )
            {
                auto  R = std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( U ), std::move( V ) );

                if ( ! is_null( zconf ) )
                    ptrcast( R.get(), lrmatrix< value_t > )->compress( *zconf );

                return R;
            }// if
        }// if
        else
        {
            auto  aca = approx::ACA< value_t >();

            // +1 to test for convergence
            acc_loc.set_max_rank( max_rank+1 );
            
            auto  [ U, V ] = aca( Dc, acc_loc );
            
            if ( U.ncols() <= std::min( max_rank, std::min( D.nrows(), D.ncols() ) / 2 - 1 ) )
            {
                auto  R = std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( U ), std::move( V ) );
                
                if ( ! is_null( zconf ) )
                    ptrcast( R.get(), lrmatrix< value_t > )->compress( *zconf );
                
                return R;
            }// if
        }// else
    }
    
    if ( is_leaf )
    {
        //
        // low-rank approximation did not compress, so stick with dense format
        //
        
        return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
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
        auto        sub_D        = tensor2< std::unique_ptr< Hpro::TMatrix< value_t > > >( 2, 2 );
        bool        all_lowrank  = true;
        bool        all_dense    = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                       sub_colis[j] - colis.first() );
                
                sub_D(i,j) = compress_topdown( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, max_rank, zconf );
                
                HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );

                if ( ! is_compressible_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;

                if ( ! is_compressible_dense( *sub_D(i,j) ) )
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
                    rank_sum += ptrcast( sub_D(i,j).get(), lrmatrix< value_t > )->rank();

            // copy sub block data into global structure
            auto    U    = blas::matrix< value_t >( rowis.size(), rank_sum );
            auto    V    = blas::matrix< value_t >( colis.size(), rank_sum );
            auto    pos  = 0; // pointer to next free space in U/V
            size_t  smem = 0; // holds memory of sub blocks
            
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    auto  Rij   = ptrcast( sub_D(i,j).get(), lrmatrix< value_t > );
                    auto  Uij   = Rij->U();
                    auto  Vij   = Rij->V();
                    auto  U_sub = U( sub_rowis[i] - rowis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );
                    auto  V_sub = V( sub_colis[j] - colis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );

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
                return std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if

        //
        // either not all low-rank or memory gets larger: construct block matrix
        //

        auto  B = std::make_unique< Hpro::TBlockMatrix< value_t > >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zconf ) )
                {
                    if ( is_compressible_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix< value_t > )->compress( *zconf );
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
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown_orig ( const indexset &                 rowis,
                        const indexset &                 colis,
                        const blas::matrix< value_t > &  D,
                        const Hpro::TTruncAcc &          acc,
                        const approx_t &                 approx,
                        const size_t                     ntile,
                        const size_t                     max_rank,
                        const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    //
    // compute lowrank approximation
    //

    const bool  is_leaf = ( std::min( D.nrows(), D.ncols() ) <= ntile );
    
    {
        auto  Dc      = blas::copy( D );  // do not modify D (!)
        auto  acc_loc = acc( rowis, colis );

        if ( is_leaf )
        {
            auto  [ U, V ] = approx( Dc, acc_loc );
            
            if ( U.ncols() < std::min( D.nrows(), D.ncols() ) / 2 )
            {
                auto  R = std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( U ), std::move( V ) );

                if ( ! is_null( zconf ) )
                    ptrcast( R.get(), lrmatrix< value_t > )->compress( *zconf );

                return R;
            }// if
        }// if
        else
        {
            auto  aca = approx::ACA< value_t >();

            // +1 to test for convergence
            acc_loc.set_max_rank( max_rank+1 );
            
            auto  [ U, V ] = aca( Dc, acc_loc );
            
            if ( U.ncols() <= std::min( max_rank, std::min( D.nrows(), D.ncols() ) / 2 - 1 ) )
            {
                auto  R = std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( U ), std::move( V ) );
                
                if ( ! is_null( zconf ) )
                    ptrcast( R.get(), lrmatrix< value_t > )->compress( *zconf );
                
                return R;
            }// if
        }// else
    }
    
    if ( is_leaf )
    {
        //
        // low-rank approximation did not compress, so stick with dense format
        //
        
        return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
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
        auto        sub_D        = tensor2< std::unique_ptr< Hpro::TMatrix< value_t > > >( 2, 2 );
        bool        all_lowrank  = true;
        bool        all_dense    = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                       sub_colis[j] - colis.first() );
                
                sub_D(i,j) = compress_topdown_orig( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, max_rank, zconf );
                
                HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );

                if ( ! is_compressible_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;

                if ( ! is_compressible_dense( *sub_D(i,j) ) )
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
                    rank_sum += ptrcast( sub_D(i,j).get(), lrmatrix< value_t > )->rank();

            // copy sub block data into global structure
            auto    U    = blas::matrix< value_t >( rowis.size(), rank_sum );
            auto    V    = blas::matrix< value_t >( colis.size(), rank_sum );
            auto    pos  = 0; // pointer to next free space in U/V
            size_t  smem = 0; // holds memory of sub blocks
            
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    auto  Rij   = ptrcast( sub_D(i,j).get(), lrmatrix< value_t > );
                    auto  Uij   = Rij->U();
                    auto  Vij   = Rij->V();
                    auto  U_sub = U( sub_rowis[i] - rowis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );
                    auto  V_sub = V( sub_colis[j] - colis.first(), blas::range( pos, pos + Uij.ncols() - 1 ) );

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
            
            auto  acc_loc = acc( rowis, colis );
            
            // +1 to test for convergence
            acc_loc.set_max_rank( max_rank+1 );
            
            auto  [ W, X ] = approx( U, V, acc_loc );

            if ( W.ncols() <= std::min( max_rank, std::min( D.nrows(), D.ncols() ) / 2 - 1 ) )
            {
                return std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if

        //
        // either not all low-rank or memory gets larger: construct block matrix
        //

        auto  B = std::make_unique< Hpro::TBlockMatrix< value_t > >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zconf ) )
                {
                    if ( is_compressible_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix< value_t > )->compress( *zconf );
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
std::unique_ptr< Hpro::TMatrix< value_t > >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const Hpro::TTruncAcc &          acc,
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
        if ( is_compressible_lowrank( *M ) )
            ptrcast( M.get(), lrmatrix< value_t > )->compress( *zconf );
                
        if ( is_compressible_dense( *M ) )
            ptrcast( M.get(), dense_matrix< value_t > )->compress( *zconf );
    }// if

    return M;
}

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const Hpro::TTruncAcc &          acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const size_t                     max_rank,
                   const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress_topdown( rowis, colis, D, acc, approx, ntile, max_rank, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle SZ/ZFP compression for global dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_compressible_dense( *M ) )
            ptrcast( M.get(), dense_matrix< value_t > )->compress( *zconf );
    }// if

    return M;
}

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown_orig ( const indexset &                 rowis,
                        const indexset &                 colis,
                        const blas::matrix< value_t > &  D,
                        const Hpro::TTruncAcc &          acc,
                        const approx_t &                 approx,
                        const size_t                     ntile,
                        const size_t                     max_rank,
                        const zconfig_t *                zconf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress_topdown_orig( rowis, colis, D, acc, approx, ntile, max_rank, zconf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle SZ/ZFP compression for global dense case
    //
    
    if ( ! is_null( zconf ) )
    {
        if ( is_compressible_dense( *M ) )
            ptrcast( M.get(), dense_matrix< value_t > )->compress( *zconf );
    }// if

    return M;
}

//
// multi-level compression with contributions from all levels
//
namespace detail
{

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
void
compress_ml ( const indexset &           rowis,
              const indexset &           colis,
              blas::matrix< value_t > &  D,
              size_t &                   csize,
              const size_t               lvl_rank,
              const Hpro::TTruncAcc &    acc,
              const approx_t &           approx,
              const size_t               ntile,
              const zconfig_t *          zconf = nullptr )
{
    using namespace hlr::matrix;
    
    //
    // compute lowrank approximation
    //

    if ( std::min( D.nrows(), D.ncols() ) <= ntile )
    {
        auto  Dc       = blas::copy( D );  // do not modify D (!)
        auto  acc_loc  = acc( rowis, colis );
        
        auto  [ U, V ] = approx( Dc, acc_loc );

        if ( U.ncols() < std::min( D.nrows(), D.ncols() ) / 2 )
        {
            // std::cout << rowis.to_string() << " × " << colis.to_string() << " : leaf" << std::endl;
            blas::prod( value_t(1), U, blas::adjoint( V ), value_t(0), D );
            csize += sizeof(value_t) * U.ncols() * ( U.nrows() + V.nrows() );
        }// if
        else
            csize += sizeof(value_t) * D.nrows() * D.ncols();
    }// if
    else
    {
        auto  Dc       = blas::copy( D );  // do not modify D (!)
        auto  acc_loc  = acc( rowis, colis );
        auto  aca      = approx::ACA< value_t >();

        auto  [ U, V ] = aca( Dc, acc_loc );

        csize += sizeof(value_t) * U.ncols() * ( U.nrows() + V.nrows() );
        
        blas::prod( value_t(-1), U, blas::adjoint(V), value_t(1), D );

        const auto  norm_rest = blas::norm_F( D );

        if ( norm_rest <= acc_loc.abs_eps() )
        {
            // std::cout << rowis.to_string() << " × " << colis.to_string() << " : " << norm_rest << " / " << acc_loc.abs_eps() << std::endl;
            blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), D );
            return;
        }// if

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

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                 sub_colis[j] - colis.first() );
                
                compress_ml( sub_rowis[i], sub_colis[j], D_sub, csize, lvl_rank, acc, approx, ntile, zconf );
            }// for
        }// for

        blas::prod( value_t(1), U, blas::adjoint( V ), value_t(1), D );
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t,
           typename zconfig_t >
void
compress_ml ( const indexset &           rowis,
              const indexset &           colis,
              blas::matrix< value_t > &  D,
              size_t &                   csize,
              const size_t               lvl_rank,
              const Hpro::TTruncAcc &    acc,
              const approx_t &           approx,
              const size_t               ntile,
              const zconfig_t *          zconf = nullptr )
{
    using namespace hlr::matrix;

    csize = 0;
    detail::compress_ml< value_t, approx_t, zconfig_t >( rowis, colis, D, csize, lvl_rank, acc, approx, ntile, zconf );
}

//
// import default implementations
//
using hlr::matrix::compress;
using hlr::matrix::decompress;

//
// compress cluster basis data
//
template < typename value_t,
           typename cluster_basis_t >
void
compress ( cluster_basis_t &        cb,
           const Hpro::TTruncAcc &  acc )
{
    using namespace hlr::matrix;

    cb.compress( acc );
    
    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( is_null( cb.son(i) ) )
                continue;
                
            compress< cluster_basis_t >( *cb.son(i), acc );
        }// for
    }// if
}

//
// decompress cluster basis data
//
template < typename value_t,
           typename cluster_basis_t >
void
decompress ( cluster_basis_t &  cb )
{
    using namespace hlr::matrix;

    cb.decompress();
    
    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( is_null( cb.son(i) ) )
                continue;
                
            decompress< cluster_basis_t >( *cb.son(i) );
        }// for
    }// if
}

}// namespace matrix

namespace tensor
{

using hlr::tensor::compress;
using hlr::tensor::decompress;

}// namespace tensor

}}// namespace hlr::seq

#endif // __HLR_SEQ_COMPRESS_HH
