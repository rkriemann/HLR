#ifndef __HLR_MATRIX_COMPRESS_HH
#define __HLR_MATRIX_COMPRESS_HH
//
// Project     : HLib
// Module      : matrix/compress
// Description : matrix compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/utils/tensor.hh>

namespace hlr { namespace matrix {

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
                   const hpro::TTruncAcc &    acc,
                   const approx_t &           approx,
                   const size_t               ntile,
                   const int                  zfp_rate,
                   size_t &                   compressed_size )
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

        auto  Dc       = blas::copy( D );  // do not modify D directly
        auto  [ U, V ] = approx( Dc, acc );

        if ( U.byte_size() + V.byte_size() < Dc.byte_size() )
        {
            // std::cout << rowis.to_string() << " × " << colis.to_string() << std::endl;

            auto  M = std::make_unique< lrmatrix >( rowis, colis, std::move( U ), std::move( V ) );

            if ( zfp_rate > 0 )
                M->compress( zfp_config_rate( zfp_rate, false ) );
            
            return M;
        }// if
        else
        {
            auto  M = std::make_unique< dense_matrix >( rowis, colis, blas::copy( D ) );

            if ( zfp_rate > 0 )
                M->compress( zfp_config_rate( zfp_rate, false ) );

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
        }// else
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
        size_t      sub_size     = 0;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                 sub_colis[j] - colis.first() );
                
                sub_D(i,j) = compress_replace( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zfp_rate, compressed_size );

                if ( ! is_null( sub_D(i,j) ) )
                    sub_size += sub_D(i,j)->byte_size();
                
                if ( is_null( sub_D(i,j) ) || ! is_generic_lowrank( *sub_D(i,j) ) )
                    all_lowrank = false;
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

                    // uncompress data before further handling
                    if ( Rij->is_compressed() )
                        Rij->uncompress();

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
            
            auto  [ W, X ] = approx( U, V, acc );

            if ( W.byte_size() + X.byte_size() < smem )
            {
                // std::cout << rowis.to_string() << " × " << colis.to_string() << std::endl;
                
                //
                // larger low-rank block more memory efficient than sum of sub-blocks: keep it
                //

                auto  M = std::make_unique< lrmatrix >( rowis, colis, std::move( W ), std::move( X ) );
                
                if ( zfp_rate > 0 )
                    M->compress( zfp_config_rate( zfp_rate, false ) );
                
                return M;
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
                
                if ( is_generic_lowrank( *sub_D(i,j) ) )
                {
                    auto  Rij = ptrcast( sub_D(i,j).get(), lrmatrix );

                    if ( zfp_rate > 0 )
                        Rij->uncompress();

                    auto  DR  = blas::prod( Rij->U< value_t >(), blas::adjoint( Rij->V< value_t >() ) );

                    blas::copy( DR, D_sub );
                }// if

                sub_D(i,j).release();
            }// for
        }// for

        // use size of compressed data below
        compressed_size += sub_size;
        
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
                   const hpro::TTruncAcc &    acc,
                   const approx_t &           approx,
                   const size_t               ntile,
                   const int                  zfp_rate,
                   size_t &                   compressed_size )
{
    auto  M = detail::compress_replace( rowis, colis, D, acc, approx, ntile, zfp_rate, compressed_size );

    if ( ! is_null( M ) )
    {
        compressed_size += M->byte_size();

        if ( is_generic_lowrank( *M ) )
        {
            auto  R = ptrcast( M.get(), lrmatrix );

            if ( zfp_rate > 0 )
                R->uncompress();

            auto  DR  = blas::prod( R->U< value_t >(), blas::adjoint( R->V< value_t >() ) );

            blas::copy( DR, D );
        }// if
    }// if
}
    
}}// namespace hlr::matrix

#endif // __HLR_MATRIX_COMPRESS_HH
