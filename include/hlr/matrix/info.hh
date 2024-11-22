#ifndef __HLR_MATRIX_INFO_HH
#define __HLR_MATRIX_INFO_HH
//
// Project     : HLR
// Module      : matrix/info
// Description : functions returning specific information about matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hpro/config.h>

#if defined(HPRO_USE_LIC_CHECK)
#define HLR_HAS_H2
#endif

#include <hpro/matrix/TRkMatrix.hh>

#if defined(HLR_HAS_H2)
#include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/lrsvmatrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/matrix/uniform_lr2matrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/h2_lr2matrix.hh>

namespace hlr { namespace matrix {

//
// return data byte sizes of given matrix
// (special and functional versions)
//
template < typename value_t >
size_t
data_byte_size ( const Hpro::TMatrix< value_t > &  M )
{
    return M.data_byte_size();
}

template < typename value_t >
size_t
data_byte_size_dense ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto    B    = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        size_t  size = 0;

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    size += data_byte_size_dense( *B->block( i, j ) );
            }// for
        }// for

        return size;
    }// if
    else if ( matrix::is_dense( M ) || Hpro::is_dense( M ) )
    {
        return  M.data_byte_size();
    }// if

    return 0;
}

template < typename value_t >
size_t
data_byte_size_lowrank ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto    B    = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        size_t  size = 0;

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    size += data_byte_size_lowrank( *B->block( i, j ) );
            }// for
        }// for

        return size;
    }// if
    else if ( matrix::is_lowrank( M )          ||
              matrix::is_lowrank_sv( M )       ||
              matrix::is_uniform_lowrank( M )  ||
              matrix::is_uniform_lowrank2( M ) ||
              matrix::is_h2_lowrank( M )       ||
              matrix::is_h2_lowrank2( M )      ||
              Hpro::is_lowrank( M ) )
    {
        return  M.data_byte_size();
    }// if

    return 0;
}

//
// return min/avg/max rank of given matrix
//
namespace detail
{

template < typename value_t >
std::tuple< uint, size_t, uint, size_t >
rank_info_helper_mat ( const Hpro::TMatrix< value_t > &  M )
{
    if ( is_blocked( M ) )
    {
        auto    B        = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        uint    min_rank = 0;
        uint    max_rank = 0;
        size_t  sum_rank = 0;
        size_t  nnodes   = 0;

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto [ min_ij, sum_ij, max_ij, n_ij ] = rank_info_helper_mat( *B->block( i, j ) );
                
                if      ( min_rank == 0 ) min_rank = min_ij;
                else if ( min_ij   != 0 ) min_rank = std::min( min_rank, min_ij );
                
                max_rank  = std::max( max_rank, max_ij );
                sum_rank += sum_ij;
                nnodes   += n_ij;
            }// for
        }// for

        return { min_rank, sum_rank, max_rank, nnodes };
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::lrmatrix< value_t > );

        return { R->rank(), R->rank(), R->rank(), R->rank() > 0 ? 1 : 0 };
    }// if
    else if ( matrix::is_lowrank_sv( M ) )
    {
        auto  R = cptrcast( &M, matrix::lrsvmatrix< value_t > );

        return { R->rank(), R->rank(), R->rank(), R->rank() > 0 ? 1 : 0 };
    }// if
    else if ( matrix::is_uniform_lowrank( M ) )
    {
        auto  R       = cptrcast( &M, matrix::uniform_lrmatrix< value_t > );
        auto  minrank = std::min( R->row_rank(), R->col_rank() );
        auto  maxrank = std::max( R->row_rank(), R->col_rank() );
        
        return { minrank, ( minrank + maxrank ) / 2, maxrank, minrank > 0 ? 1 : 0 };
    }// if
    else if ( matrix::is_h2_lowrank( &M ) )
    {
        auto  R       = cptrcast( &M, matrix::h2_lrmatrix< value_t > );
        auto  minrank = std::min( R->row_rank(), R->col_rank() );
        auto  maxrank = std::max( R->row_rank(), R->col_rank() );
        
        return { minrank, ( minrank + maxrank ) / 2, maxrank, minrank > 0 ? 1 : 0 };
    }// if
    // else if ( is_generic_lowrank( M ) )
    // {
    //     auto  R = cptrcast( &M, lrmatrix< value_t > );

    //     return { R->rank(), R->rank(), R->rank(), R->rank() > 0 ? 1 : 0 };
    // }// if

    return { 0, 0, 0, 0 };
}

}// namespace detail

template < typename value_t >
std::tuple< uint, uint, uint >
rank_info ( const Hpro::TMatrix< value_t > & M )
{
    auto [ min_rank, sum_rank, max_rank, nnodes ] = detail::rank_info_helper_mat( M );

    return { min_rank, uint( double(sum_rank) / double(nnodes) ), max_rank };
}

//
// determine data memory per level in H-matrix
//
template < typename value_t >
void
print_mem_lvl ( const Hpro::TMatrix< value_t > & H )
{
    using  matrix_t       = Hpro::TMatrix< value_t >;
    using  block_matrix_t = Hpro::TBlockMatrix< value_t >;
    
    auto    current     = std::list< const matrix_t * >{ &H };
    uint    lvl         = 0;
    size_t  size_lr_all = 0;
    size_t  size_d_all  = 0;

    std::cout << boost::format( "%4s" ) % "lvl" << " │ "
              << boost::format( "%10s" ) % "lowrank" << " │ "
              << boost::format( "%10s" ) % "dense" << std::endl
              << "─────┼────────────┼───────────" << std::endl;
    
    while ( current.size() > 0 )
    {
        size_t  size_lr = 0;
        size_t  size_d  = 0;
        auto    next    = decltype( current )();

        for ( auto  M : current )
        {
            if ( is_blocked( M ) )
            {
                auto  B = cptrcast( M, block_matrix_t );

                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        if ( ! is_null( B->block( i, j ) ) )
                            next.push_back( B->block( i, j ) );
            }// if
            else if ( matrix::is_lowrank( M ) )
                size_lr += M->data_byte_size();
            else if ( matrix::is_lowrank_sv( M ) )
                size_lr += M->data_byte_size();
            else if ( matrix::is_dense( M ) )
                size_d += M->data_byte_size();
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// for

        if ( size_d + size_lr > 0 )
            std::cout << boost::format( "%4d" ) % lvl << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_lr ) << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_d ) << std::endl;

        size_lr_all += size_lr;
        size_d_all  += size_d;
        
        lvl++;
        current = std::move( next );
    }// while

    std::cout << "─────┼────────────┼───────────" << std::endl
              << boost::format( "%4s" ) % "all" << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_lr_all ) << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_d_all ) << std::endl;
}

template < typename value_t >
void
print_mem_lvl ( const Hpro::TMatrix< value_t > &                 H,
                const matrix::shared_cluster_basis< value_t > &  rcb,
                const matrix::shared_cluster_basis< value_t > &  ccb )
{
    using  matrix_t        = Hpro::TMatrix< value_t >;
    using  block_matrix_t  = Hpro::TBlockMatrix< value_t >;
    using  cluster_basis_t = matrix::shared_cluster_basis< value_t >;
    
    auto    current_mat = std::list< const matrix_t * >{ &H };
    auto    current_rcb = std::list< const cluster_basis_t * >{ &rcb };
    auto    current_ccb = std::list< const cluster_basis_t * >{ &ccb };
    uint    lvl         = 0;
    size_t  size_cb_all = 0;
    size_t  size_lr_all = 0;
    size_t  size_d_all  = 0;

    std::cout << boost::format( "%4s" ) % "lvl" << " │ "
              << boost::format( "%10s" ) % "bases" << " │ "
              << boost::format( "%10s" ) % "lowrank" << " │ "
              << boost::format( "%10s" ) % "dense" << std::endl
              << "─────┼────────────┼────────────┼───────────" << std::endl;
    
    while ( current_mat.size() > 0 )
    {
        size_t  size_d   = 0;
        size_t  size_lr  = 0;
        size_t  size_cb  = 0;
        auto    next_mat = decltype( current_mat )();
        auto    next_rcb = decltype( current_rcb )();
        auto    next_ccb = decltype( current_ccb )();

        for ( auto  M : current_mat )
        {
            if ( is_blocked( M ) )
            {
                auto  B = cptrcast( M, block_matrix_t );

                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        if ( ! is_null( B->block( i, j ) ) )
                            next_mat.push_back( B->block( i, j ) );
            }// if
            else if ( matrix::is_uniform_lowrank( M ) )
                size_lr += M->data_byte_size();
            else if ( matrix::is_uniform_lowrank2( M ) )
                size_lr += M->data_byte_size();
            else if ( matrix::is_dense( M ) )
                size_d += M->data_byte_size();
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// for

        for ( auto  cb : current_rcb )
        {
            size_cb += sizeof( value_t ) * cb->is().size() * cb->rank();
            
            for ( uint  i = 0; i < cb->nsons(); ++i )
                if ( ! is_null( cb->son( i ) ) )
                    next_rcb.push_back( cb->son( i ) );
        }// for

        for ( auto  cb : current_ccb )
        {
            size_cb += sizeof( value_t ) * cb->is().size() * cb->rank();
            
            for ( uint  i = 0; i < cb->nsons(); ++i )
                if ( ! is_null( cb->son( i ) ) )
                    next_ccb.push_back( cb->son( i ) );
        }// for

        if ( size_d + size_lr > 0 )
            std::cout << boost::format( "%4d" ) % lvl << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_cb ).c_str() << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_lr ).c_str() << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_d ).c_str() <<std::endl;

        size_cb_all += size_cb;
        size_lr_all += size_lr;
        size_d_all  += size_d;
        
        lvl++;
        current_mat = std::move( next_mat );
        current_rcb = std::move( next_rcb );
        current_ccb = std::move( next_ccb );
    }// while

    std::cout << "─────┼────────────┼────────────┼───────────" << std::endl
              << boost::format( "%4s" ) % "all" << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_cb_all ).c_str() << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_lr_all ).c_str() << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_d_all ).c_str() <<std::endl;
}

template < typename value_t >
void
print_mem_lvl ( const Hpro::TMatrix< value_t > &                 H,
                const matrix::nested_cluster_basis< value_t > &  rcb,
                const matrix::nested_cluster_basis< value_t > &  ccb )
{
    using  matrix_t        = Hpro::TMatrix< value_t >;
    using  block_matrix_t  = Hpro::TBlockMatrix< value_t >;
    using  cluster_basis_t = matrix::nested_cluster_basis< value_t >;
    
    auto    current_mat = std::list< const matrix_t * >{ &H };
    auto    current_rcb = std::list< const cluster_basis_t * >{ &rcb };
    auto    current_ccb = std::list< const cluster_basis_t * >{ &ccb };
    uint    lvl         = 0;
    size_t  size_cb_all = 0;
    size_t  size_lr_all = 0;
    size_t  size_d_all  = 0;

    std::cout << boost::format( "%4s" ) % "lvl" << " │ "
              << boost::format( "%10s" ) % "bases" << " │ "
              << boost::format( "%10s" ) % "lowrank" << " │ "
              << boost::format( "%10s" ) % "dense" << std::endl
              << "─────┼────────────┼────────────┼───────────" << std::endl;
    
    while ( current_mat.size() > 0 )
    {
        size_t  size_d   = 0;
        size_t  size_lr  = 0;
        size_t  size_cb  = 0;
        auto    next_mat = decltype( current_mat )();
        auto    next_rcb = decltype( current_rcb )();
        auto    next_ccb = decltype( current_ccb )();

        for ( auto  M : current_mat )
        {
            if ( is_blocked( M ) )
            {
                auto  B = cptrcast( M, block_matrix_t );

                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                    for ( uint  j = 0; j < B->nblock_cols(); ++j )
                        if ( ! is_null( B->block( i, j ) ) )
                            next_mat.push_back( B->block( i, j ) );
            }// if
            else if ( matrix::is_h2_lowrank( M ) )
                size_lr += M->data_byte_size();
            else if ( matrix::is_h2_lowrank2( M ) )
                size_lr += M->data_byte_size();
            else if ( matrix::is_dense( M ) )
                size_d += M->data_byte_size();
            else
                HLR_ERROR( "unsupported matrix type: " + M->typestr() );
        }// for

        for ( auto  cb : current_rcb )
        {
            if ( cb->nsons() == 0 )
            {
                size_cb += sizeof( value_t ) * cb->is().size() * cb->rank();
            }// if
            else
            {
                for ( uint  i = 0; i < cb->nsons(); ++i )
                    if ( ! is_null( cb->son( i ) ) )
                    {
                        size_cb += sizeof( value_t ) * cb->son(i)->rank() * cb->rank();
                        next_rcb.push_back( cb->son( i ) );
                    }// if
            }// else
        }// for

        for ( auto  cb : current_ccb )
        {
            if ( cb->nsons() == 0 )
            {
                size_cb += sizeof( value_t ) * cb->is().size() * cb->rank();
            }// if
            else
            {
                for ( uint  i = 0; i < cb->nsons(); ++i )
                    if ( ! is_null( cb->son( i ) ) )
                    {
                        size_cb += sizeof( value_t ) * cb->son(i)->rank() * cb->rank();
                        next_ccb.push_back( cb->son( i ) );
                    }// if
            }// else
        }// for

        if ( size_d + size_lr > 0 )
            std::cout << boost::format( "%4d" ) % lvl << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_cb ).c_str() << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_lr ).c_str() << " │ "
                      << boost::format( "%10s" ) % Hpro::Mem::to_string( size_d ).c_str() <<std::endl;

        size_cb_all += size_cb;
        size_lr_all += size_lr;
        size_d_all  += size_d;
        
        lvl++;
        current_mat = std::move( next_mat );
        current_rcb = std::move( next_rcb );
        current_ccb = std::move( next_ccb );
    }// while

    std::cout << "─────┼────────────┼────────────┼───────────" << std::endl
              << boost::format( "%4s" ) % "all" << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_cb_all ).c_str() << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_lr_all ).c_str() << " │ "
              << boost::format( "%10s" ) % Hpro::Mem::to_string( size_d_all ).c_str() <<std::endl;
}

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_INFO_HH
