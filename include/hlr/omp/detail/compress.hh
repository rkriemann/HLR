#ifndef __HLR_OMP_MATRIX_DETAIL_COMPRESS_HH
#define __HLR_OMP_MATRIX_DETAIL_COMPRESS_HH
//
// Project     : HLR
// Module      : omp/compress
// Description : matrix compression functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

namespace hlr { namespace omp {

namespace matrix { namespace detail {

//
// build H-matrix from given dense matrix without reording rows/columns
// starting lowrank approximation at blocks of size ntile × ntile and
// then trying to agglomorate low-rank blocks up to the root
//
template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress ( const indexset &                 rowis,
           const indexset &                 colis,
           const blas::matrix< value_t > &  D,
           const accuracy &                 acc,
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
                // std::cout << "R: " << to_string( rowis ) << " x " << to_string( colis ) << ", " << U.ncols() << std::endl;
                return std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( U ), std::move( V ) );
            }// if
        }// if

        // std::cout << "D: " << to_string( rowis ) << " x " << to_string( colis ) << std::endl;
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

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                           sub_colis[j] - colis.first() );
                    
                    sub_D(i,j) = compress( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, zconf );
                    
                    HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );
                }// for
            }// for
        }// omp taskgroup

        bool  all_lowrank = true;
        bool  all_dense   = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
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
                // std::cout << "R: " << to_string( rowis ) << " x " << to_string( colis ) << ", " << W.ncols() << std::endl;
                return std::make_unique< lrmatrix< value_t > >( rowis, colis, std::move( W ), std::move( X ) );
            }// if
        }// if

        //
        // always join dense blocks
        //
        
        if ( all_dense )
        {
            // std::cout << "D: " << to_string( rowis ) << " x " << to_string( colis ) << std::endl;
            return std::make_unique< dense_matrix< value_t > >( rowis, colis, std::move( blas::copy( D ) ) );
        }// if
        
        //
        // either not all low-rank or memory gets larger: construct block matrix
        // also: finally compress with zfp
        //

        // std::cout << "B: " << to_string( rowis ) << " x " << to_string( colis ) << std::endl;

        auto  B = std::make_unique< Hpro::TBlockMatrix< value_t > >( rowis, colis );

        B->set_block_struct( 2, 2 );
        
        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
                if ( ! is_null( zconf ) )
                {
                    if ( is_generic_lowrank( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), lrmatrix< value_t > )->compress( *zconf );
                
                    if ( is_generic_dense( *sub_D(i,j) ) )
                        ptrcast( sub_D(i,j).get(), dense_matrix< value_t > )->compress( *zconf );
                }// if
                
                B->set_block( i, j, sub_D(i,j).release() );
            }// for
        }// for

        return B;
    }// else
}

//
// top-down compression approach: if low-rank approximation is possible
// within given accuracy and maximal rank, stop recursion
//
template < typename value_t,
           typename approx_t,
           typename zconfig_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
compress_topdown ( const indexset &                 rowis,
                   const indexset &                 colis,
                   const blas::matrix< value_t > &  D,
                   const accuracy &                 acc,
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

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                           sub_colis[j] - colis.first() );
                        
                    sub_D(i,j) = compress_topdown( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, max_rank, zconf );
                    
                    HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );
                }// for
            }// for
        }// omp taskgroup

        bool  all_lowrank = true;
        bool  all_dense   = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
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
        // also: finally compress with zfp
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
                        const accuracy &                 acc,
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
        auto  aca     = approx::ACA< value_t >();
        auto  acc_loc = acc( rowis, colis );

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

        #pragma omp taskgroup
        {
            #pragma omp taskloop collapse(2) default(shared)
            for ( uint  i = 0; i < 2; ++i )
            {
                for ( uint  j = 0; j < 2; ++j )
                {
                    const auto  D_sub = D( sub_rowis[i] - rowis.first(),
                                           sub_colis[j] - colis.first() );
                    
                    sub_D(i,j) = compress_topdown_orig( sub_rowis[i], sub_colis[j], D_sub, acc, approx, ntile, max_rank, zconf );
                    
                    HLR_ASSERT( ! is_null( sub_D(i,j).get() ) );
                }// for
            }// for
        }// omp taskgroup

        bool  all_lowrank = true;
        bool  all_dense   = true;

        for ( uint  i = 0; i < 2; ++i )
        {
            for ( uint  j = 0; j < 2; ++j )
            {
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

//
// multi-level compression with contributions from all levels
//
template < typename value_t,
           typename approx_t,
           typename zconfig_t >
void
compress_ml ( const indexset &           rowis,
              const indexset &           colis,
              blas::matrix< value_t > &  D,
              size_t &                   csize,
              const size_t               lvl_rank,
              const accuracy &           acc,
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
            csize += U.ncols() * ( U.nrows() + V.nrows() );
        }// if
        else
            csize += D.nrows() * D.ncols();
    }// if
    else
    {
        auto  Dc       = blas::copy( D );  // do not modify D (!)
        auto  acc_loc  = acc( rowis, colis );
        auto  aca      = approx::ACA< value_t >();

        acc_loc.set_max_rank( lvl_rank );
        
        auto  [ U, V ] = aca( Dc, acc_loc );

        csize += U.ncols() * ( U.nrows() + V.nrows() );
        
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

//
// compress compressible sub-blocks within H-matrix
//
template < typename value_t >
void
compress ( Hpro::TMatrix< value_t > &  A,
           const accuracy &            acc )
{
    using namespace hlr::matrix;

    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        #pragma omp taskloop collapse(2) default(shared)
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( is_null( BA->block( i, j ) ) )
                    continue;
                
                compress( *BA->block( i, j ), acc );
            }// for
        }// for
    }// if
    else if ( compress::is_compressible( A ) )
    {
        dynamic_cast< compress::compressible * >( &A )->compress( acc );
    }// if
}

//
// compress cluster basis data
//
template < typename value_t,
           typename cluster_basis_t >
void
compress ( cluster_basis_t &  cb,
           const accuracy &   acc )
{
    using namespace hlr::matrix;

    cb.compress( acc );
    
    if ( cb.nsons() > 0 )
    {
        #pragma omp taskloop default(shared)
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son(i) ) )
                compress< value_t, cluster_basis_t >( *cb.son(i), acc );
        }// for
    }// if
}

//
// decompress H-matrix
//
template < typename value_t >
void
decompress ( Hpro::TMatrix< value_t > &  A )
{
    using namespace hlr::matrix;

    if ( is_blocked( A ) )
    {
        auto  BA = ptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        #pragma omp taskloop collapse(2) default(shared)
        for ( uint  i = 0; i < BA->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BA->nblock_cols(); ++j )
            {
                if ( is_null( BA->block( i, j ) ) )
                    continue;
                
                decompress( *BA->block( i, j ) );
            }// for
        }// for
    }// if
    else if ( compress::is_compressible( A ) )
    {
        dynamic_cast< compress::compressible * >( &A )->decompress();
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
        #pragma omp taskloop default(shared)
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son(i) ) )
                decompress< value_t, cluster_basis_t >( *cb.son(i) );
        }// for
    }// if
}

}}// namespace matrix

}}// namespace hlr::omp

#endif // __HLR_OMP_MATRIX_DETAIL_COMPRESS_HH
