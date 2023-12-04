#ifndef __HLR_TBB_DETAIL_H2_MATRIX_HH
#define __HLR_TBB_DETAIL_H2_MATRIX_HH
//
// Project     : HLR
// Module      : matrix.hh
// Description : H² matrix construction
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TBlockMatrix.hh>

#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>

#include <hlr/tbb/detail/uniform_basis.hh>

namespace hlr { namespace tbb { namespace matrix { namespace detail {

using namespace hlr::matrix;

template < typename value_t >
void
init_cluster_bases ( const Hpro::TMatrix< value_t > &   M,
                     nested_cluster_basis< value_t > &  rowcb,
                     nested_cluster_basis< value_t > &  colcb )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        {
            // auto  lock = std::scoped_lock( rowcb.mutex() );
            
            for ( uint  i = 0; i < B->nblock_rows(); ++i )
            {
                auto  rowcb_i = rowcb.son( i );
            
                for ( uint  j = 0; j < B->nblock_cols(); ++j )
                {
                    auto  M_ij    = B->block( i, j );
                
                    if ( ! is_null( M_ij ) )
                    {
                        if ( is_null( rowcb_i ) )
                        {
                            rowcb_i = new nested_cluster_basis< value_t >( M_ij->row_is() );
                            rowcb.set_son( i, rowcb_i );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( rowcb_i->nsons() == 0 ))
                            rowcb_i->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );

                        continue;
                    }// if
                }// for
            }// for
        }

        {
            // auto  lock = std::scoped_lock( colcb.mutex() );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
            
                for ( uint  i = 0; i < B->nblock_rows(); ++i )
                {
                    auto  M_ij = B->block( i, j );
                
                    if ( ! is_null( M_ij ) )
                    {
                        if ( is_null( colcb_j ) )
                        {
                            colcb_j = new nested_cluster_basis< value_t >( M_ij->col_is() );
                            colcb.set_son( j, colcb_j );
                        }// if
            
                        if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
                            colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );

                        continue;
                    }// if
                }// for
            }// for
        }

        //
        // recurse
        //
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            auto  rowcb_i = rowcb.son( i );
            
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  colcb_j = colcb.son( j );
                
                if ( ! is_null( B->block( i, j ) ) )
                    init_cluster_bases( *B->block( i, j ), *rowcb_i, *colcb_j );
            }// for
        }// for
    }// if
}

//
// collect lowrank matrices per block row/column and corresponding semi-coupling matrices.
// Also set up structure of cluster bases.
// 
template < typename value_t >
using  lr_list_t      = std::list< const hlr::matrix::lrmatrix< value_t > * >;

template < typename value_t >
using  lr_mat_map_t   = std::unordered_map< indexset, lr_list_t< value_t >, indexset_hash >;

template < typename value_t >
using  coupling_map_t = std::unordered_map< const hlr::matrix::lrmatrix< value_t > *, blas::matrix< value_t > >;

template < typename value_t >
void
build_mat_map ( const Hpro::TMatrix< value_t > &   A,
                nested_cluster_basis< value_t > &  rowcb,
                nested_cluster_basis< value_t > &  colcb,
                lr_mat_map_t< value_t > &          row_map,
                coupling_map_t< value_t > &        row_coupling,
                std::mutex &                       row_mtx,
                lr_mat_map_t< value_t > &          col_map,
                coupling_map_t< value_t > &        col_coupling,
                std::mutex &                       col_mtx )
{
    using namespace hlr::matrix;
    
    //
    // decide upon cluster type, how to construct matrix
    //
    
    auto  M = std::unique_ptr< Hpro::TMatrix< value_t > >();
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // compute semi-coupling, e.g. QR factorization of U/V
        // (see "build_cluster_basis" for more details)
        //
        
        auto  R  = cptrcast( &A, hlr::matrix::lrmatrix< value_t > );
        auto  W  = blas::copy( R->U_direct() );
        auto  X  = blas::copy( R->V_direct() );
        auto  Cw = blas::matrix< value_t >();
        auto  Cx = blas::matrix< value_t >();

        ::tbb::parallel_invoke( [&] () { blas::qr( W, Cw, false ); }, // only need R, not Q
                                [&] () { blas::qr( X, Cx, false ); } );
        
        HLR_ASSERT( Cw.ncols() != 0 );
        HLR_ASSERT( Cx.ncols() != 0 );

        // add matrix to block row/column
        // and remember the semi-coupling (see "build_cluster_basis" why the "other")


        {
            auto  lock = std::scoped_lock( row_mtx );
            
            row_map[ A.row_is() ].push_back( R );
            row_coupling[ R ] = std::move( Cx );
        }
            
        {
            auto  lock = std::scoped_lock( col_mtx );
            
            col_map[ A.col_is() ].push_back( R );
            col_coupling[ R ] = std::move( Cw );
        }
    }// if
    else if ( is_blocked( A ) )
    {
        auto  B = cptrcast( &A, Hpro::TBlockMatrix< value_t > );

        //
        // build (empty) cluster bases objects
        //

        // ::tbb::parallel_invoke(
        //     [&,B] ()
        //     {
        //         auto  lock = std::scoped_lock( rowcb.mutex() );
                
        //         for ( uint  i = 0; i < B->nblock_rows(); ++i )
        //         {
        //             auto  rowcb_i = rowcb.son( i );
                    
        //             for ( uint  j = 0; j < B->nblock_cols(); ++j )
        //             {
        //                 auto  M_ij = B->block( i, j );
                        
        //                 if ( ! is_null( M_ij ) )
        //                 {
        //                     if ( is_null( rowcb_i ) )
        //                     {
        //                         rowcb_i = new nested_cluster_basis< value_t >( M_ij->row_is() );
        //                         rowcb.set_son( i, rowcb_i );
        //                     }// if
        
        //                     if ( is_blocked( M_ij ) && ( rowcb_i->nsons() == 0 ))
        //                         rowcb_i->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_rows() );

        //                     continue;
        //                 }// if
        //             }// for
        //         }// for
        //     },

        //     [&,B] ()
        //     {
        //         auto  lock = std::scoped_lock( colcb.mutex() );
                
        //         for ( uint  j = 0; j < B->nblock_cols(); ++j )
        //         {
        //             auto  colcb_j = colcb.son( j );
                    
        //             for ( uint  i = 0; i < B->nblock_rows(); ++i )
        //             {
        //                 auto  M_ij = B->block( i, j );
                        
        //                 if ( ! is_null( M_ij ) )
        //                 {
        //                     if ( is_null( colcb_j ) )
        //                     {
        //                         colcb_j = new nested_cluster_basis< value_t >( M_ij->col_is() );
        //                         colcb.set_son( j, colcb_j );
        //                     }// if
        
        //                     if ( is_blocked( M_ij ) && ( colcb_j->nsons() == 0 ))
        //                         colcb_j->set_nsons( cptrcast( M_ij, Hpro::TBlockMatrix< value_t > )->nblock_cols() );

        //                     continue;
        //                 }// if
        //             }// for
        //         }// for
        //     }
        // );

        //
        // recurse
        //
        
        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,B] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    auto  rowcb_i = rowcb.son( i );
                    
                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  colcb_j = colcb.son( j );
                
                        if ( ! is_null( B->block( i, j ) ) )
                            build_mat_map( *B->block( i, j ), *rowcb_i, *colcb_j,
                                           row_map, row_coupling, row_mtx,
                                           col_map, col_coupling, col_mtx );
                    }// for
                }// for
            }
        );
    }// if
}

//
// build nested cluster basis using precomputed semi-coupling of lowrank matrices
// in block row/columns
//
template < typename value_t >
using  lr_mat_list_t  = std::list< const hlr::matrix::lrmatrix< value_t > * >;

template < typename value_t,
           typename basisapx_t >
blas::matrix< value_t >
build_nested_cluster_basis ( nested_cluster_basis< value_t > &  cb,
                             const basisapx_t &                 basisapx,
                             const accuracy &                   acc,
                             const lr_mat_map_t< value_t > &    lrmat_map,
                             const coupling_map_t< value_t > &  coupling_map,
                             const lr_mat_list_t< value_t > &   parent_matrices,
                             const bool                         transposed )
{
    using  real_t  = Hpro::real_type_t< value_t >;

    const matop_t  op = ( transposed ? apply_transposed : apply_normal );

    //
    // set up list of lowrank matrices contributing to local basis
    //
    
    auto  mat_list = lr_mat_list_t< value_t >( parent_matrices );
    auto  is_sort  = [op] ( auto  M1, auto  M2 ) { return M1->row_is( op ).is_strictly_left_of( M2->row_is( op ) ); };

    if ( lrmat_map.find( cb.is() ) != lrmat_map.end() )
    {
        const auto  local_mats = lrmat_map.at( cb.is() );
        
        mat_list.insert( mat_list.end(), local_mats.begin(), local_mats.end() );
    }// if

    mat_list.sort( is_sort );
    
    // std::cout << cb.is() << " : " << mat_list.size() << std::endl;
    
    //
    // determine local "rank"
    //
    
    uint  ncols = 0;

    // determine total number of columns
    for ( const auto  R_i : mat_list )
    {
        const auto  C_i = coupling_map.at( R_i );
                
        ncols += C_i.nrows();
    }// for
    
    //
    //
    // construct cluster basis
    //

    if ( cb.nsons() == 0 )
    {
        // check for empty basis
        if ( mat_list.empty() )
            return blas::matrix< value_t >();
        
        //
        // compute column basis for block row
        //
        //  ( U₀·V₀'  U₁·V₁'  U₂·V₂'  … )
        //
        // as 
        //
        //   ( U₀·C₀'·Q₀'  U₁·C₁'·Q₁'  U₂'·C₂'·Q₂' … )
        //
        // with QR decomposition V_i = Q_i C_i
        // (precomputed in "build_mat_map" above)
        //
        // As Q_i is orthogonal, it can be neglected in column basis computation!
        //

        // build ( U_0·C_0'  U_1·C_1'  U_2'·C_2' … )
        const uint  nrows = cb.is().size();
        auto        X     = blas::matrix< value_t >( nrows, ncols );
        uint        pos   = 0;

        for ( const auto  M_i : mat_list )
        {   // std::cout << "  " << cb.is() << ", " << M_i->block_is() << ", " << pos << std::endl;
            const auto  C_i   = coupling_map.at( M_i );
            auto        U_i   = M_i->U( op );
            auto        U_sub = blas::matrix< value_t >( U_i, cb.is() - M_i->row_ofs( op ), blas::range::all );
            auto        X_i   = blas::prod( U_sub, blas::adjoint( C_i ) );
            auto        X_sub = blas::matrix( X, blas::range::all, blas::range( pos, pos + C_i.nrows() - 1 ) );

            blas::copy( X_i, X_sub );
            pos += C_i.nrows();
        }// for

        // actually build cluster basis
        auto  Ws = blas::vector< real_t >(); // singular values corresponding to basis vectors
        auto  W  = basisapx.column_basis( X, acc, & Ws );
        auto  R  = blas::prod( blas::adjoint( W ), X );

        cb.set_basis( std::move( W ), std::move( Ws ) );

        return std::move( R );
    }// if
    else
    {
        //
        // recurse
        //

        uint  nrows    = 0;
        auto  son_data = std::vector< std::pair< nested_cluster_basis< value_t > *, blas::matrix< value_t > > >( cb.nsons() );

        // for ( uint  i = 0; i < cb.nsons(); ++i )
        ::tbb::parallel_for< size_t >(
            size_t(0), cb.nsons(),
            [&,transposed] ( const size_t  i )
            {
                if ( ! is_null( cb.son( i ) ) )
                {
                    auto  R_i = build_nested_cluster_basis( *cb.son( i ), basisapx, acc, lrmat_map, coupling_map, mat_list, transposed );
                    
                    nrows += R_i.nrows();
                    son_data[i]  = { cb.son(i), std::move( R_i ) };
                }// if
            } );

        // check for empty basis
        if ( mat_list.empty() )
            return blas::matrix< value_t >();
        
        //
        // compute transfer matrices by joining the R_i
        //

        auto  V   = blas::matrix< value_t >( nrows, ncols );
        uint  pos = 0;

        for ( const auto  [ cb_i, R_i ] : son_data )
        {
            //
            // build mapping from lowrank matrices of son cluster to R_i
            //

            auto  list_i  = lr_mat_list_t< value_t >( mat_list.begin(), mat_list.end() );
            auto  mat_idx = std::unordered_map< const hlr::matrix::lrmatrix< value_t > *, idx_t >();
            
            if ( lrmat_map.find( cb_i->is() ) != lrmat_map.end() )
            {
                auto  map_i = lrmat_map.at( cb_i->is() );
                
                list_i.insert( list_i.end(), map_i.begin(), map_i.end() );
            }// if

            list_i.sort( is_sort );

            // offset is defined by position in son list ...
            uint  idx = 0;
            
            for ( auto  M_j : list_i )
            {
                const auto  C_j = coupling_map.at( M_j );

                mat_idx[ M_j ] = idx;
                idx           += C_j.nrows();
            }// for

            const auto  rrange_R = blas::range( 0, idx_t( R_i.nrows() ) - 1 );
            const auto  rrange_V = rrange_R + pos;
            uint        cpos     = 0;

            // ... but we use only those also within local list
            for ( auto  M_j : mat_list )
            {
                const auto  C_j      = coupling_map.at( M_j );
                const auto  ncols_j  = C_j.nrows(); // always used as C'
                const auto  ofs_j    = mat_idx[ M_j ];
                const auto  crange_R = blas::range( ofs_j, ofs_j + ncols_j - 1 );
                const auto  crange_V = blas::range( cpos, cpos + ncols_j - 1 );
                const auto  R_sub    = blas::matrix< value_t >( R_i, rrange_R, crange_R );
                auto        V_sub    = blas::matrix< value_t >( V,   rrange_V, crange_V );
                
                blas::copy( R_sub, V_sub );
                cpos += ncols_j;
            }// for

            pos += R_i.nrows();
        }// for

        // std::cout << cb.is() << " : |V| = " << blas::norm_F( V ) << std::endl;
        // io::matlab::write( V, "V2" );
        
        auto  [ Q, R ] = blas::factorise_ortho( V, acc );
        auto  E        = std::vector< blas::matrix< value_t > >( cb.nsons() );

        // std::cout << cb.is() << " : |V| = " << blas::norm_F( Q ) << std::endl;
        // std::cout << cb.is() << " : |R| = " << blas::norm_F( R ) << std::endl;
        // io::matlab::write( V, "V2" );
        
        pos = 0;
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            const auto  R_i = son_data[i].second;
            const auto  Q_i = blas::matrix< value_t >( Q,
                                                       blas::range( pos, pos + R_i.nrows() - 1 ),
                                                       blas::range::all );

            // std::cout << cb.is() << " : |E| = " << blas::norm_F( Q_i ) << std::endl;
            E[i] = std::move( blas::copy( Q_i ) );
            pos += R_i.nrows();
        }// for

        cb.set_transfer( std::move( E ) );

        return std::move( R );
    }// else

    return blas::matrix< value_t >();
}

//
// build H² representation of A by converting all lowrank matrices
// into uniform low-rank matrices using given cluster bases rowcb/colcb.
//
template < typename value_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_h2 ( const Hpro::TMatrix< value_t > &   A,
           nested_cluster_basis< value_t > &  rowcb,
           nested_cluster_basis< value_t > &  colcb )
{
    using namespace hlr::matrix;

    //
    // decide upon cluster type, how to construct matrix
    //

    std::unique_ptr< Hpro::TMatrix< value_t > >  M;
    
    if ( hlr::matrix::is_lowrank( A ) )
    {
        //
        // compute coupling matrix as W'·U·(X'·V)'
        // with cluster basis W and X
        //
        
        auto  R = cptrcast( &A, hlr::matrix::lrmatrix< value_t > );
        auto  W = rowcb.transform_forward( R->U_direct() );
        auto  X = colcb.transform_forward( R->V_direct() );
        auto  S = blas::prod( W, blas::adjoint( X ) );

        M = std::make_unique< h2_lrmatrix< value_t > >( A.row_is(), A.col_is(), rowcb, colcb, std::move( S ) );
    }// if
    else if ( is_blocked( A ) )
    {
        auto  BA = cptrcast( &A, Hpro::TBlockMatrix< value_t > );
        
        M = std::make_unique< Hpro::TBlockMatrix< value_t > >();

        auto  B = ptrcast( M.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BA );

        ::tbb::parallel_for(
            ::tbb::blocked_range2d< uint >( 0, B->nblock_rows(),
                                            0, B->nblock_cols() ),
            [&,BA,B] ( const ::tbb::blocked_range2d< uint > &  r )
            {
                for ( auto  i = r.rows().begin(); i != r.rows().end(); ++i )
                {
                    auto  rowcb_i = rowcb.son( i );
                    
                    HLR_ASSERT( ! is_null( rowcb_i ) );

                    for ( auto  j = r.cols().begin(); j != r.cols().end(); ++j )
                    {
                        auto  colcb_j = colcb.son( j );
                        auto  A_ij    = BA->block( i, j );
                
                        HLR_ASSERT( ! is_null( colcb_j ) );

                        if ( ! is_null( A_ij ) )
                        {
                            auto  B_ij = build_h2( *A_ij, *rowcb_i, *colcb_j );
                            
                            B->set_block( i, j, B_ij.release() );
                        }// if
                    }// for
                }// for
            } );
    }// if
    else if ( hlr::matrix::is_dense( A ) )
    {
        auto  D  = cptrcast( &A, hlr::matrix::dense_matrix< value_t > );
        auto  DD = blas::copy( D->mat() );

        return  std::make_unique< dense_matrix< value_t > >( D->row_is(), D->col_is(), std::move( DD ) );
    }// if
    else
    {
        M = A.copy();
    }// else

    M->set_id( A.id() );
    M->set_procs( A.procs() );

    return M;
}

}}}}// namespace hlr::tbb::matrix::detail

#endif // __HLR_TBB_DETAIL_H2_MATRIX_HH
