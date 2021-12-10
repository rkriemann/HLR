#ifndef __HLR_TBB_MATRIX_COMPRESS_HH
#define __HLR_TBB_MATRIX_COMPRESS_HH
//
// Project     : HLib
// Module      : tbb/compress
// Description : matrix compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/utils/tensor.hh>

namespace hlr { namespace tbb { namespace matrix {

namespace detail
{

//
// adaptively compress data by low-rank approximation and
// replace original data by approximate data
// - store total size of compressed data in <compressed_size>
//
template < typename value_t, typename approx_t >
std::unique_ptr< hpro::TMatrix >
compress_replace ( const indexset &           rowis,
                   const indexset &           colis,
                   blas::matrix< value_t > &  D,
                   size_t &                   compressed_size,
                   const hpro::TTruncAcc &    acc,
                   const approx_t &           approx,
                   const size_t               ntile,
                   const zfp_config *         zfp_conf = nullptr )
{
    using namespace hlr::matrix;

    // std::cout << "START " << rowis.to_string() << " × " << colis.to_string() << std::endl;

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
            auto  Dc       = blas::copy( D );  // do not modify D directly
            auto  [ U, V ] = approx( Dc, acc( rowis, colis ) );
            
            if ( U.byte_size() + V.byte_size() < Dc.byte_size() )
            {
                return std::make_unique< lrmatrix >( rowis, colis, std::move( U ), std::move( V ) );
            }// if
        }// if
        
        auto  M = std::make_unique< dense_matrix >( rowis, colis, blas::copy( D ) );

        if ( ! is_null( zfp_conf ) )
            M->compress( *zfp_conf );

        // remember compressed size (with or without ZFP compression)
        compressed_size += M->byte_size();

        // uncompress and replace
        if ( M->is_compressed() )
        {
            M->uncompress();
            
            auto  T = M->matrix();
            
            blas::copy( std::get< blas::matrix< value_t > >( T ), D );
        }// if
        
        // and discard matrix
        return std::unique_ptr< hpro::TMatrix >();
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
        size_t      sub_size     = 0;
        auto        mtx          = std::mutex();

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 ),
            [&,ntile] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                         sub_colis[j] - colis.first() );
                        
                        sub_D(i,j) = compress_replace( sub_rowis[i], sub_colis[j], D_sub, compressed_size, acc, approx, ntile, zfp_conf );
                        
                        if ( ! is_null( sub_D(i,j) ) )
                        {
                            auto  lock = std::scoped_lock( mtx );
                            
                            sub_size += sub_D(i,j)->byte_size();
                        }// if
                    }// for
                }// for
            } );

        bool  all_lowrank = true;

        for ( uint  i = 0; i < 2; ++i )
            for ( uint  j = 0; j < 2; ++j )
                if ( is_null( sub_D(i,j) ) || ! is_generic_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;
        
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
            
            auto  [ W, X ] = approx( U, V, acc( rowis, colis ) );

            if ( W.byte_size() + X.byte_size() < smem )
            {
                //
                // larger low-rank block more memory efficient than sum of sub-blocks: keep it
                //

                return std::make_unique< lrmatrix >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // otherwise copy back compressed data and discard low-rank matrices
        //

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                 sub_colis[j] - colis.first() );
                
                if ( ! is_null( sub_D(i,j) ) && is_generic_lowrank( *sub_D(i,j) ) )
                {
                    auto  Rij = ptrcast( sub_D(i,j).get(), lrmatrix );

                    if ( ! is_null( zfp_conf ) )
                    {
                        Rij->compress( *zfp_conf );
                        compressed_size += Rij->byte_size();
                        Rij->uncompress();
                    }// if
                    else
                        compressed_size += Rij->byte_size();

                    auto  DR  = blas::prod( Rij->U< value_t >(), blas::adjoint( Rij->V< value_t >() ) );

                    blas::copy( DR, D_sub );
                }// if

                sub_D(i,j).release();
            }// for
        }// for
            
        return std::unique_ptr< hpro::TMatrix >();
    }// else
}

}// namespace detail

//
// wrapper function handling global low-rank case
//
template < typename value_t, typename approx_t >
void
compress_replace ( const indexset &           rowis,
                   const indexset &           colis,
                   blas::matrix< value_t > &  D,
                   size_t &                   compressed_size,
                   const hpro::TTruncAcc &    acc,
                   const approx_t &           approx,
                   const size_t               ntile,
                   const zfp_config *         zfp_conf = nullptr )
{
    auto  M = detail::compress_replace( rowis, colis, D, compressed_size, acc, approx, ntile, zfp_conf );

    if ( ! is_null( M ) )
    {
        compressed_size += M->byte_size();

        if ( is_generic_lowrank( *M ) )
        {
            auto  R = ptrcast( M.get(), lrmatrix );

            if ( ! is_null( zfp_conf ) )
            {
                R->compress( *zfp_conf );
                compressed_size += R->byte_size();
                R->uncompress();
            }// if
            else
                compressed_size += R->byte_size();

            auto  DR  = blas::prod( R->U< value_t >(), blas::adjoint( R->V< value_t >() ) );

            blas::copy( DR, D );
        }// if
    }// if
}
    
//
// build H-matrix from given dense matrix without reording rows/columns
// starting lowrank approximation at blocks of size ntile × ntile and
// then trying to agglomorate low-rank blocks up to the root
//
namespace detail
{

template < typename value_t,
           typename approx_t >
std::unique_ptr< hpro::TMatrix >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const hpro::TTruncAcc &          acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zfp_config *               zfp_conf = nullptr )
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

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 ),
            [&,ntile] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                               sub_colis[j] - colis.first() );
                        
                        sub_D(i,j) = compress( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zfp_conf );
                        
                        HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );
                    }// for
                }// for
            } );

        bool  all_lowrank = true;
        bool  all_dense   = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
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
        // also: finally compress with zfp
        //

        auto  B = std::make_unique< hpro::TBlockMatrix >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zfp_conf ) )
                {
                    if ( is_generic_lowrank( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), lrmatrix )->compress( *zfp_conf );
                
                    if ( is_generic_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix )->compress( *zfp_conf );
                }// if
                
                B->set_block( i, j, sub_D(i,j).release() );
            }// for
        }// for

        return B;
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
std::unique_ptr< hpro::TMatrix >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const hpro::TTruncAcc &          acc,
           const approx_t &                 approx,
           const size_t                     ntile,
           const zfp_config *               zfp_conf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress( rowis, colis, D, acc, approx, ntile, zfp_conf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle ZFP compression for global lowrank/dense case
    //
    
    if ( ! is_null( zfp_conf ) )
    {
        if ( is_generic_lowrank( *M ) )
            ptrcast( M.get(), lrmatrix )->compress( *zfp_conf );
                
        if ( is_generic_dense( *M ) )
            ptrcast( M.get(), dense_matrix )->compress( *zfp_conf );
    }// if

    return M;
}
    
//
// top-down compression approach: if low-rank approximation is possible
// within given accuracy and maximal rank, stop recursion
//
namespace detail
{

template < typename value_t,
           typename approx_t >
std::unique_ptr< hpro::TMatrix >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const hpro::TTruncAcc &          acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const zfp_config *               zfp_conf = nullptr )
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

            if ( ! is_null( zfp_conf ) )
                ptrcast( R.get(), lrmatrix )->compress( *zfp_conf );

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

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, 2, 0, 2 ),
            [&,ntile] ( const auto &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                               sub_colis[j] - colis.first() );
                        
                        sub_D(i,j) = compress_topdown( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zfp_conf );
                        
                        HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );
                    }// for
                }// for
            } );

        bool  all_dense = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
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
        // also: finally compress with zfp
        //

        auto  B = std::make_unique< hpro::TBlockMatrix >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zfp_conf ) )
                {
                    if ( is_generic_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix )->compress( *zfp_conf );
                }// if
                
                B->set_block( i, j, sub_D(i,j).release() );
            }// for
        }// for

        return B;
    }// else
}

}// namespace detail

template < typename value_t,
           typename approx_t >
std::unique_ptr< hpro::TMatrix >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const hpro::TTruncAcc &          acc,
                   const approx_t &                 approx,
                   const size_t                     ntile,
                   const zfp_config *               zfp_conf = nullptr )
{
    using namespace hlr::matrix;

    auto  M = detail::compress_topdown( rowis, colis, D, acc, approx, ntile, zfp_conf );

    HLR_ASSERT( ! is_null( M ) );

    //
    // handle ZFP compression for global dense case
    //
    
    if ( ! is_null( zfp_conf ) )
    {
        if ( is_generic_dense( *M ) )
            ptrcast( M.get(), dense_matrix )->compress( *zfp_conf );
    }// if

    return M;
}
    
}}}// namespace hlr::tbb::matrix

#endif // __HLR_TBB_MATRIX_COMPRESS_HH
