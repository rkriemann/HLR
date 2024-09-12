#ifndef __HLR_ARITH_DETAIL_H2_HH
#define __HLR_ARITH_DETAIL_H2_HH
//
// Project     : HLR
// Module      : arith/h2
// Description : arithmetic functions for uniform matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hlr/matrix/nested_cluster_basis.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/h2_lrmatrix.hh>
#include <hlr/matrix/h2_lr2matrix.hh>
#include <hlr/vector/uniform_vector.hh>
#include <hlr/vector/scalar_vector.hh>

namespace hlr { namespace h2 { namespace detail {

using hlr::vector::uniform_vector;
using hlr::vector::scalar_vector;

//
// compute mat-vec M·x = y with uniform vectors x,y.
// For dense blocks of M, the actual result is directly updated.
//
// Version traversing the H-matrix
//
template < typename value_t,
           typename cluster_basis_t >
void
mul_vec ( const value_t                              alpha,
          const Hpro::matop_t                        op_M,
          const Hpro::TMatrix< value_t > &           M,
          const uniform_vector< cluster_basis_t > &  x,
          uniform_vector< cluster_basis_t > &        y,
          const scalar_vector< value_t > &           sx,
          scalar_vector< value_t > &                 sy )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, Hpro::TBlockMatrix< value_t > );

        if ( ! (( B->nblock_rows( op_M ) == y.nblocks() ) &&
                ( B->nblock_cols( op_M ) == x.nblocks() )) )
            HLR_ERROR( "matrix/vector block structure incompatible" );
            
        for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
        {
            auto  y_i = y.block( i );
            
            for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
            {
                auto  B_ij = B->block( i, j, op_M );
                auto  x_j  = x.block( j );
            
                if ( ! is_null( B_ij ) )
                {
                    mul_vec( alpha, op_M, *B_ij, *x_j, *y_i, sx, sy );
                }// if
            }// for
        }// for
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else if ( matrix::is_h2_lowrank2( M ) )
    {
        auto  R = cptrcast( &M, matrix::h2_lr2matrix< value_t > );

        R->uni_apply_add( alpha, x.coeffs(), y.coeffs(), op_M );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  x_i = blas::vector< value_t >( blas::vec( sx ), M.col_is( op_M ) - sx.ofs() );
        auto  y_j = blas::vector< value_t >( blas::vec( sy ), M.row_is( op_M ) - sy.ofs() );

        M.apply_add( alpha, x_i, y_j, op_M );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

// //
// // matvec version using special cluster tree like data structure 
// //
// template < typename value_t >
// struct cluster_blocks_t
// {
//     using dense_matrix = matrix::dense_matrix< value_t >;
//     using h2_lrmatrix  = matrix::h2_lrmatrix< value_t >;
//     using u_vector     = uniform_vector< matrix::nested_cluster_basis< value_t > >;
        
//     // corresponding index set of cluster
//     indexset                                    is;
    
//     // list of associated matrices
//     std::list< const dense_matrix * >           D;
//     std::list< std::pair< const h2_lrmatrix *,
//                           const u_vector > >    U;

//     // son matrices (following cluster tree)
//     std::vector< cluster_blocks_t * >           sub_blocks;

//     // ctor
//     cluster_blocks_t ( const indexset &  ais )
//             : is( ais )
//     {}

//     // dtor
//     ~cluster_blocks_t ()
//     {
//         for ( auto  cb : sub_blocks )
//             delete cb;
//     }
// };

// template < typename value_t >
// void
// build_cluster_blocks ( const matop_t                              op_M,
//                        const Hpro::TMatrix< value_t > &           M,
//                        const uniform_vector< matrix::nested_cluster_basis< value_t > > &  x,
//                        cluster_blocks_t< value_t > &              cb )
// {
//     if ( is_blocked( M ) )
//     {
//         auto  B = cptrcast( & M, Hpro::TBlockMatrix< value_t > );

//         if ( cb.sub_blocks.size() == 0 )
//             cb.sub_blocks.resize( B->nblock_rows( op_M ) );
        
//         for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
//         {
//             HLR_ASSERT( ! is_null( B->block( i, 0, op_M ) ) );

//             if ( is_null( cb.sub_blocks[i] ) )
//                 cb.sub_blocks[i] = new cluster_blocks_t< value_t >( B->block( i, 0, op_M )->row_is( op_M ) );
//         }// for
                
//         for ( uint  i = 0; i < B->nblock_rows( op_M ); ++i )
//         {
//             for ( uint  j = 0; j < B->nblock_cols( op_M ); ++j )
//             {
//                 if ( B->block( i, j, op_M ) != nullptr )
//                     build_cluster_blocks( op_M, * B->block( i, j, op_M ), & x.block( j ), * cb.sub_blocks[i] );
//             }// for
//         }// for
//     }// if
//     else if ( matrix::is_h2_lowrank( M ) )
//     {
//         cb.R.push_back( { cptrcast( &M, matrix::h2_lrmatrix< value_t > ), & x } );
//     }// if
//     else if ( matrix::is_dense( M ) )
//     {
//         cb.D.push_back( cptrcast( &M, matrix::dense_matrix< value_t > ) );
//     }// if
//     else
//         HLR_ERROR( "unsupported matrix type: " + M.typestr() );
// }

// template < typename value_t >
// void
// mul_vec_cl ( const value_t                              alpha,
//              const matop_t                              op_M,
//              const cluster_blocks_t< value_t > &        cb,
//              const uniform_vector< matrix::nested_cluster_basis< value_t > > &  x,
//              uniform_vector< matrix::nested_cluster_basis< value_t > > &        y,
//              const vector::scalar_vector< value_t > &   sx,
//              vector::scalar_vector< value_t > &         sy )
// {
//     if ( alpha == value_t(0) )
//         return;

//     //
//     // compute update with all block in current block row
//     //

//     if ( ! cb.R.empty() )
//     {
//         for ( auto  [ R, ux ] : cb.R )
//         {
//             #if defined(HLR_HAS_ZBLAS_DIRECT)
//             if ( R->is_compressed() )
//             {
//                 switch ( op_M )
//                 {
//                     case apply_normal     : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoupling(), ux.coeffs().data(), y.coeffs().data() ); break;
//                     case apply_conjugate  : { HLR_ASSERT( false ); }
//                     case apply_transposed : { HLR_ASSERT( false ); }
//                     case apply_adjoint    : compress::zblas::mulvec( R->row_rank(), R->col_rank(), op_M, alpha, R->zcoupling(), ux.coeffs().data(), y.coeffs().data() ); break;
//                     default               : HLR_ERROR( "unsupported matrix operator" );
//                 }// switch
//             }// if
//             else
//             #endif
//             {
//                 switch ( op_M )
//                 {
//                     case Hpro::apply_normal     : blas::mulvec( alpha, R->coupling(), ux.coeffs(), value_t(1), y.coeffs() ); break;
//                     case Hpro::apply_conjugate  : HLR_ASSERT( false );
//                     case Hpro::apply_transposed : HLR_ASSERT( false );
//                     case Hpro::apply_adjoint    : blas::mulvec( alpha, blas::adjoint( R->coupling() ), ux.coeffs(), value_t(1), y.coeffs() ); break;
//                     default                     : HLR_ERROR( "unsupported matrix operator" );
//                 }// switch
//             }// else
//         }// for
//     }// if

//     if ( ! cb.D.empty() )
//     {
//         auto  y_j = blas::vector< value_t >( blas::vec( y ), cb.is - y.ofs() );
//         auto  yt  = blas::vector< value_t >( y_j.length() );
    
//         for ( auto  M : cb.M )
//         {
//             auto  x_i = blas::vector< value_t >( blas::vec( x ), M->col_is( op_M ) - x.ofs() );
            
//             M->apply_add( 1, x_i, yt, op_M );
//         }// for

//         blas::add( alpha, yt, y_j );
//     }// if

//     //
//     // recurse
//     //
    
//     for ( auto  sub : cb.sub_blocks )
//         mul_vec_cl( alpha, op_M, *sub, x, y );
// }

//
// copy given scalar vector into uniform vector format
//
template < typename value_t,
           typename cluster_basis_t >
std::unique_ptr< uniform_vector< cluster_basis_t > >
scalar_to_uniform ( const cluster_basis_t &           cb,
                    const scalar_vector< value_t > &  v )
{
    auto  u = std::make_unique< uniform_vector< cluster_basis_t > >( cb );

    if ( cb.nsons() == 0 )
    {
        auto  V = cb.basis();
        
        if ( V.nrows() == 0 )
            return u;
        
        //
        // s ≔ V'·v
        //
        
        auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );
        auto  s    = blas::mulvec( blas::adjoint( V ), v_cb );

        u->set_coeffs( std::move( s ) );
    }// if
    else
    {
        //
        // s ≔ V'·v = (∑_i V_i E_i)' v = ∑_i E_i' V_i' v
        //

        auto  s = blas::vector< value_t >( cb.rank() );
        
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            auto  u_i = scalar_to_uniform( *cb.son(i), v );

            if ( cb.rank() > 0 )
                blas::mulvec( blas::adjoint( cb.transfer_mat(i) ), u_i->coeffs(), s );
            
            u->set_block( i, u_i.release() );
        }// for

        u->set_coeffs( std::move( s ) );
    }// else

    return u;
}

//
// add coefficients of uniform vector y to scalar vector y
//
template < typename value_t,
           typename cluster_basis_t >
void
add_uniform_to_scalar ( const uniform_vector< cluster_basis_t > &  u,
                        scalar_vector< value_t > &                 v,
                        blas::vector< value_t > &                  s )
{
    if ( s.length() > 0 )
        blas::add( value_t(1), u.coeffs(), s );
    else
        s = blas::copy( u.coeffs() );

    auto &  cb = u.basis();

    if ( cb.nsons() == 0 )
    {
        auto  v_loc = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );

        blas::mulvec( value_t(1), cb.basis(), s, value_t(1), v_loc );
    }// if
    else
    {
        // shift local coefficients of u to sons and proceed
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            auto  u_i = u.block( i );
            auto  s_i = cb.transfer_to_son( i, s );
            
            add_uniform_to_scalar( *u_i, v, s_i );
        }// for
    }// else
}

////////////////////////////////////////////////////////////////////////////////
//
// block row wise with implicit backward transformation
//
////////////////////////////////////////////////////////////////////////////////

//
// construct mapping id -> clusterbasis
//
template < typename value_t >
void
build_id2cb ( matrix::nested_cluster_basis< value_t > &                   cb,
              std::vector< matrix::nested_cluster_basis< value_t > * > &  idmap )
{
    HLR_ASSERT( cb.id() != -1 );
    HLR_ASSERT( cb.id() < idmap.size() );

    idmap[ cb.id() ] = & cb;

    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
                build_id2cb( * cb.son( i ), idmap );
        }// for
    }// if
}

//
// construct mapping cluster basis id (cluster) -> list of matrix blocks in M
//
template < typename value_t >
void
build_id2blocks ( const matrix::nested_cluster_basis< value_t > &                 cb,
                  const Hpro::TMatrix< value_t > &                                M,
                  std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap,
                  const bool                                                      transposed )
{
    HLR_ASSERT( cb.id() != -1 );
    HLR_ASSERT( cb.id() < blockmap.size() );

    if ( ! is_blocked( M ) )
    {
        blockmap[ cb.id() ].push_back( & M );
    }// else
    else
    {
        auto  op = ( transposed ? apply_transposed : apply_normal );
        auto  B  = cptrcast( & M, Hpro::TBlockMatrix< value_t > );

        HLR_ASSERT( B->nblock_rows( op ) == cb.nsons() );

        for ( uint  i = 0; i < B->nblock_rows( op ); ++i )
        {
            auto  cb_i = cb.son( i );

            HLR_ASSERT( ! is_null( cb_i ) );
            
            for ( uint  j = 0; j < B->nblock_cols( op ); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    build_id2blocks( *cb_i, * B->block( i, j ), blockmap, transposed );
            }// for
        }// if
    }// else
}

//
// no construction of explicit, hierarchical representation of uniform vector
// but just store coefficients in std::vector indexed by cluster basis id
//
template < typename value_t >
void
scalar_to_uniform ( const matrix::nested_cluster_basis< value_t > &   cb,
                    const vector::scalar_vector< value_t > &          v,
                    std::vector< blas::vector< value_t > > &          vcoeff )
{
    if ( cb.nsons() == 0 )
    {
        if ( cb.rank() > 0 )
        {
            auto  v_cb = blas::vector< value_t >( blas::vec( v ), cb.is() - v.ofs() );

            HLR_ASSERT( cb.id() < vcoeff.size() );
                
            vcoeff[ cb.id() ] = std::move( cb.transform_forward( v_cb ) );
        }// if
    }// if
    else
    {
        if ( cb.rank() > 0 )
        {
            //
            // recurse and transfer coefficients from sons
            //
            
            auto  Si = std::vector< blas::vector< value_t > >( cb.nsons() );
        
            for ( uint  i = 0; i < cb.nsons(); ++i )
            {
                auto  cb_i = cb.son(i);

                scalar_to_uniform( *cb_i, v, vcoeff );
                Si[i] = std::move( cb.transfer_from_son( i, vcoeff[ cb_i->id() ] ) );
            }// for

            auto  s = blas::vector< value_t >( cb.rank() );
        
            for ( auto  s_i : Si )
                blas::add( value_t(1), s_i, s );
        
            vcoeff[ cb.id() ] = std::move( s );
        }// if
        else
        {
            //
            // just recurse (no transfer matrices)
            //
            
            for ( uint  i = 0; i < cb.nsons(); ++i )
                scalar_to_uniform( *cb.son(i), v, vcoeff );
        }// else
    }// else
}

template < typename value_t >
void
mul_vec_row ( const value_t                                                         alpha,
              const Hpro::matop_t                                                   op_M,
              const matrix::nested_cluster_basis< value_t > &                       rowcb,
              const std::vector< matrix::nested_cluster_basis< value_t > * > &      colcb,
              const std::vector< std::list< const Hpro::TMatrix< value_t > * > > &  blockmap,
              const std::vector< blas::vector< value_t > > &                        xcoeff,
              const blas::vector< value_t > &                                       y_parent,
              const vector::scalar_vector< value_t > &                              sx,
              vector::scalar_vector< value_t > &                                    sy )
{
    //
    // start with coefficients from parent
    //

    auto  y_loc = blas::copy( y_parent );
    auto  y_son = std::vector< blas::vector< value_t > >( rowcb.nsons() );

    if (( y_loc.length() == 0 ) && ( rowcb.rank() != 0 ))
        y_loc = std::move( blas::vector< value_t >( rowcb.rank() ) );

    HLR_ASSERT( y_loc.length() == rowcb.rank() );
    
    //
    // perform matvec with all blocks in block row
    //

    auto &  blockrow = blockmap[ rowcb.id() ];

    if ( blockrow.size() > 0 )
    {
        auto  rowis  = rowcb.is();
        auto  y_j    = blas::vector< value_t >( blas::vec( sy ), rowis - sy.ofs() );
        auto  t_j    = blas::vector< value_t >( y_j.length() );
        bool  update = false;
            
        for ( auto  M : blockrow )
        {
            if ( matrix::is_h2_lowrank( M ) )
            {
                auto    R       = cptrcast( M, matrix::h2_lrmatrix< value_t > );
                auto &  colcb_R = R->col_cb();
                auto    ux      = xcoeff[ colcb_R.id() ];

                R->uni_apply_add( alpha, ux, y_loc, op_M );
            }// if
            else if ( matrix::is_h2_lowrank2( M ) )
            {
                auto    R       = cptrcast( M, matrix::h2_lr2matrix< value_t > );
                auto &  colcb_R = R->col_cb();
                auto    ux      = xcoeff[ colcb_R.id() ];

                R->uni_apply_add( alpha, ux, y_loc, op_M );
            }// if
            else if ( matrix::is_dense( M ) )
            {
                auto  x_i = blas::vector< value_t >( blas::vec( sx ), M->col_is( op_M ) - sx.ofs() );
                        
                M->apply_add( alpha, x_i, t_j, op_M );
                update = true;
            }// if
            else
                HLR_ERROR( "unsupported matrix type : " + M->typestr() );
        }// for

        //
        // handle local uniform part
        //
        
        if ( y_loc.length() > 0 )
        {
            if ( rowcb.nsons() == 0 )
            {
                // apply to local update
                rowcb.transform_backward( y_loc, t_j );
                update = true;
            }// if
            else
            {
                // transfer local part to sons
                for ( uint  i = 0; i < rowcb.nsons(); ++i )
                    y_son[i] = std::move( rowcb.transfer_to_son( i, y_loc ) );
            }// else
        }// if

        //
        // finally add local update to destination
        //

        if ( update )
            blas::add( value_t(1), t_j, y_j );
    }// if

    //
    // recurse
    //

    if ( rowcb.nsons() > 0 )
    {
        for ( uint  i = 0; i < rowcb.nsons(); ++i )
        {
            if ( ! is_null( rowcb.son( i ) ) )
                mul_vec_row( alpha, op_M, * rowcb.son( i ), colcb, blockmap, xcoeff, y_son[i], sx, sy );
        }// for
    }// if
}

////////////////////////////////////////////////////////////////////////////////
//
// FLOPs
//
////////////////////////////////////////////////////////////////////////////////

//
// return FLOPs needed to convert vector into uniform basis
//
template < typename cluster_basis_t >
flops_t
scalar_to_uniform_flops ( const cluster_basis_t &  cb )
{
    flops_t  flops = 0;
    
    if ( cb.nsons() == 0 )
    {
        if constexpr ( Hpro::is_complex_type_v< Hpro::value_type_t< cluster_basis_t > > )
            flops += FLOPS_ZGEMV( cb.rank(), cb.is().size() );
        else
            flops += FLOPS_DGEMV( cb.rank(), cb.is().size() );
    }// if
    else
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            flops += scalar_to_uniform_flops( *cb.son(i) );

            if ( cb.rank() > 0 )
            {
                const auto  E_i = cb.transfer_mat(i);
                
                if constexpr ( Hpro::is_complex_type_v< Hpro::value_type_t< cluster_basis_t > > )
                    flops += FLOPS_ZGEMV( E_i.ncols(), E_i.nrows() );
                else
                    flops += FLOPS_DGEMV( E_i.ncols(), E_i.nrows() );
            }// for
        }// for
    }// if

    return flops;
}

//
// return FLOPs needed for computing y = y + α op( M ) x
// with M in uniform-H format
//
template < typename value_t >
flops_t
mul_vec_flops ( const Hpro::matop_t               op_M,
                const Hpro::TMatrix< value_t > &  M )
{
    using namespace hlr::matrix;
    
    if ( is_blocked( M ) )
    {
        auto        B       = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        const auto  row_ofs = M.row_is( op_M ).first();
        const auto  col_ofs = M.col_is( op_M ).first();
        flops_t     flops   = 0;
    
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                auto  B_ij = B->block( i, j );
            
                if ( ! is_null( B_ij ) )
                    flops += mul_vec_flops( op_M, *B_ij );
            }// for
        }// for

        return flops;
    }// if
    else if ( matrix::is_h2_lowrank( M ) )
    {
        const auto  R = cptrcast( &M, matrix::h2_lrmatrix< value_t > );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( R->row_rank( op_M ), R->col_rank( op_M ) );
        else
            return FLOPS_DGEMV( R->row_rank( op_M ), R->col_rank( op_M ) );
    }// if
    else if ( matrix::is_h2_lowrank2( M ) )
    {
        const auto  R = cptrcast( &M, matrix::h2_lr2matrix< value_t > );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( R->row_rank( op_M ), R->rank() ) + FLOPS_ZGEMV( R->col_rank( op_M ), R->rank() );
        else
            return FLOPS_DGEMV( R->row_rank( op_M ), R->rank() ) + FLOPS_DGEMV( R->col_rank( op_M ), R->rank() );
    }// if
    else if ( matrix::is_dense( M ) )
    {
        const auto  nrows = M.nrows( op_M );
        const auto  ncols = M.ncols( op_M );
        
        if constexpr ( Hpro::is_complex_type_v< value_t > )
            return FLOPS_ZGEMV( nrows, ncols );
        else
            return FLOPS_DGEMV( nrows, ncols );
    }// if
    else
        HLR_ERROR( "unsupported matrix type: " + M.typestr() );

    return 0;
}

//
// return FLOPs for adding vector in uniform cluster basis to scalar vector
//
template < typename cluster_basis_t >
flops_t
add_uniform_to_scalar_flops ( const cluster_basis_t &  cb )
{
    flops_t  flops = 0;
    
    if ( cb.nsons() == 0 )
    {
        if constexpr ( Hpro::is_complex_type_v< Hpro::value_type_t< cluster_basis_t > > )
            flops += FLOPS_ZGEMV( cb.is().size(), cb.rank() );
        else
            flops += FLOPS_DGEMV( cb.is().size(), cb.rank() );
    }// if
    else
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( cb.rank() > 0 )
            {
                const auto  E_i = cb.transfer_mat(i);

                if constexpr ( Hpro::is_complex_type_v< Hpro::value_type_t< cluster_basis_t > > )
                    flops += FLOPS_ZGEMV( E_i.nrows(), E_i.ncols() );
                else
                    flops += FLOPS_DGEMV( E_i.nrows(), E_i.ncols() );
            }// if
            
            flops += add_uniform_to_scalar_flops( *cb.son(i) );
        }// for
    }// if

    return flops;
}

}}}// namespace hlr::h2::detail

#endif // __HLR_ARITH_DETAIL_H2_HH
