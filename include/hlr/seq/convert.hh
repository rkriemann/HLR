#ifndef __HLR_SEQ_CONVERT_HH
#define __HLR_SEQ_CONVERT_HH
//
// Project     : HLib
// Module      : matrix/convert
// Description : matrix conversion functions
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#if defined(HAS_UNIVERSAL)
#include <universal/number/posit/posit.hpp>
#endif

#include <hlib-config.h>

#if defined(USE_LIC_CHECK)
#define HAS_H2
#include <hpro/matrix/TUniformMatrix.hh>
#endif

#include <hlr/matrix/convert.hh>
#include <hlr/matrix/lrmatrix.hh>
#include <hlr/matrix/dense_matrix.hh>
#include <hlr/matrix/uniform_lrmatrix.hh>
#include <hlr/utils/compression.hh>

namespace hlr { namespace seq { namespace matrix {

using hlr::matrix::convert_to_lowrank;
using hlr::matrix::convert_to_dense;

//
// convert matrix between different floating point precisions
// - return storage used with destination precision
//
template < typename T_value_dest,
           typename T_value_src >
size_t
convert_prec ( hpro::TMatrix &  M )
{
    if constexpr( std::is_same_v< T_value_dest, T_value_src > )
        return M.byte_size();
    
    if ( is_blocked( M ) )
    {
        auto    B = ptrcast( &M, hpro::TBlockMatrix );
        size_t  s = sizeof(hpro::TBlockMatrix);

        s += B->nblock_rows() * B->nblock_cols() * sizeof(hpro::TMatrix *);
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    s += convert_prec< T_value_dest, T_value_src >( * B->block( i, j ) );
            }// for
        }// for

        return s;
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  R = ptrcast( &M, hpro::TRkMatrix );
        auto  U = blas::copy< T_value_dest >( blas::mat_U< T_value_src >( R ) );
        auto  V = blas::copy< T_value_dest >( blas::mat_V< T_value_src >( R ) );

        blas::copy< T_value_dest, T_value_src >( U, blas::mat_U< T_value_src >( R ) );
        blas::copy< T_value_dest, T_value_src >( V, blas::mat_V< T_value_src >( R ) );

        return R->byte_size() - sizeof(T_value_src) * R->rank() * ( R->nrows() + R->ncols() ) + sizeof(T_value_dest) * R->rank() * ( R->nrows() + R->ncols() ); 
    }// if
    else if ( hlr::matrix::is_generic_lowrank( M ) )
    {
        auto  R = ptrcast( &M, hlr::matrix::lrmatrix );

        std::visit(
            [R] ( auto &&  UV )
            {
                auto  U = blas::copy< T_value_dest, T_value_src >( UV.U );
                auto  V = blas::copy< T_value_dest, T_value_src >( UV.V );

                R->set_lrmat( std::move( U ), std::move( V ) );
            },
            R->factors() );

        return R->byte_size(); 
    }// if
    else if ( is_dense( M ) )
    {
        auto  D  = ptrcast( &M, hpro::TDenseMatrix );
        auto  DD = blas::copy< T_value_dest >( blas::mat< T_value_src >( D ) );

        blas::copy< T_value_dest, T_value_src >( DD, blas::mat< T_value_src >( D ) );

        return D->byte_size() - sizeof(T_value_src) * D->nrows() * D->ncols() + sizeof(T_value_dest) * D->nrows() * D->ncols();
    }// if
    else if ( matrix::is_generic_dense( M ) )
    {
        auto  D  = ptrcast( &M, matrix::dense_matrix );

        std::visit(
            [D] ( auto &&  M )
            {        
                auto  M2 = blas::copy< T_value_dest, T_value_src >( M );

                D->set_matrix( std::move( M2 ) );
            },
            D->matrix() );

        return D->byte_size();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );

    return 0;
}

#if defined(HAS_ZFP)
//
// compress data using ZFP and return memory consumption
//
template < typename value_t >
size_t
convert_zfp ( hpro::TMatrix &  A,
              zfp_config &     config )
{
    if ( is_blocked( A ) )
    {
        auto    B = ptrcast( &A, hpro::TBlockMatrix );
        size_t  s = sizeof(hpro::TBlockMatrix);

        s += B->nblock_rows() * B->nblock_cols() * sizeof(hpro::TMatrix *);
        
        for ( uint i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint j = 0; j < B->nblock_cols(); ++j )
            {
                s += convert_zfp< value_t >( * B->block(i,j), config );
            }// for
        }// for

        return s;
    }// if
    else if ( is_dense( A ) )
    {
        auto    D = ptrcast( &A, hpro::TDenseMatrix );
        size_t  s = A.byte_size() - sizeof(value_t) * D->nrows() * D->ncols() - sizeof(blas::matrix< value_t >);

        auto    u = zfp::const_array2< value_t >( D->nrows(), D->ncols(), config, 0, 0 );

        u.set( blas::mat< value_t >( D ).data() );

        const size_t  mem_dense = sizeof(value_t) * D->nrows() * D->ncols();
        const size_t  mem_zfp   = u.compressed_size();

        if ( mem_zfp < mem_dense )
        {
            s += u.compressed_size();
            s += sizeof(zfp::const_array2< value_t >);

            // write back compressed data
            u.get( blas::mat< value_t >( D ).data() );
        }// if
        else
            return A.byte_size();

        return s;
    }// if
    else if ( matrix::is_generic_dense( A ) )
    {
        auto    D = ptrcast( &A, matrix::dense_matrix );
        size_t  s = 0;

        std::visit(
            [&s,D,config] ( auto &&  M )
            {
                using  m_value_t = typename std::decay_t< decltype(M) >::value_t;
                using  m_real_t  = typename hpro::real_type_t< m_value_t >;

                uint   factor    = sizeof(m_value_t) / sizeof(m_real_t);

                s = D->byte_size() - sizeof(m_value_t) * D->nrows() * D->ncols() - sizeof(blas::matrix< m_value_t >);

                if constexpr( std::is_same_v< m_value_t, m_real_t > )
                {
                    auto  u = zfp::const_array2< m_value_t >( D->nrows(), D->ncols(), config, 0, 0 );

                    u.set( M.data() );

                    const size_t  mem_dense = sizeof(m_value_t) * D->nrows() * D->ncols();
                    const size_t  mem_zfp   = u.compressed_size();

                    if ( mem_zfp < mem_dense )
                    {
                        s += u.compressed_size();
                        s += sizeof(zfp::const_array2< m_value_t >);

                        u.get( M.data() );
                    }// if
                    else
                        s = D->byte_size();
                }// if
                else
                {
                    auto  u = zfp::const_array3< m_real_t >( D->nrows(), D->ncols(), factor, config, 0, 0 );

                    u.set( (m_real_t*) M.data() );

                    const size_t  mem_dense = sizeof(m_value_t) * D->nrows() * D->ncols();
                    const size_t  mem_zfp   = u.compressed_size();

                    if ( mem_zfp < mem_dense )
                    {
                        s += u.compressed_size();
                        s += sizeof(zfp::const_array3< m_real_t >);

                        u.get( (m_real_t*) M.data() );
                    }// if
                    else
                        s = D->byte_size();
                }// else
            },
            D->matrix()
        );
        
        return s;
    }// if
    else if ( is_lowrank( A ) )
    {
        auto    R = ptrcast( &A, hpro::TRkMatrix );
        size_t  s = A.byte_size() - sizeof(value_t) * R->rank() * ( R->nrows() + R->ncols() ) - 2*sizeof(blas::matrix< value_t >);

        const size_t  mem_lr  = sizeof(value_t) * R->rank() * ( R->nrows() + R->ncols() );
        size_t        mem_zfp = 0;
        auto          U_zfp   = std::vector< value_t >( R->rank() * R->nrows() );
        auto          V_zfp   = std::vector< value_t >( R->rank() * R->ncols() );

        {
            auto  uU = zfp::const_array2< value_t >( R->nrows(), R->rank(), config, 0, 0 );
        
            uU.set( blas::mat_U< value_t >( R ).data() );

            mem_zfp += uU.compressed_size();

            uU.get( U_zfp.data() );
        }

        {
            auto    uV = zfp::const_array2< value_t >( R->ncols(), R->rank(), config, 0, 0 );

            uV.set( blas::mat_V< value_t >( R ).data() );

            mem_zfp += uV.compressed_size();

            uV.get( V_zfp.data() );
        }
        
        if ( mem_zfp < mem_lr )
        {
            s += mem_zfp;
            s += 2*sizeof(zfp::const_array2< value_t >);

            // write back compressed data
            memcpy( blas::mat_U< value_t >( R ).data(), U_zfp.data(), sizeof(value_t) * U_zfp.size() );
            memcpy( blas::mat_V< value_t >( R ).data(), V_zfp.data(), sizeof(value_t) * V_zfp.size() );
        }// if
        else
            return A.byte_size();

        return s;
    }// if
    else if ( hlr::matrix::is_generic_lowrank( A ) )
    {
        auto    R = ptrcast( &A, matrix::lrmatrix );
        size_t  s = 0;

        std::visit(
            [R,&s,&config] ( auto &&  UV )
            {
                using  uv_value_t = typename std::decay_t< decltype(UV) >::value_t;
                using  uv_real_t  = typename hpro::real_type_t< uv_value_t >;

                uint   factor     = sizeof(uv_value_t) / sizeof(uv_real_t);
                
                s = R->byte_size() - sizeof(uv_value_t) * R->rank() * ( R->nrows() + R->ncols() ) - 2*sizeof(blas::matrix< uv_value_t >);

                const size_t  mem_lr  = sizeof(uv_value_t) * R->rank() * ( R->nrows() + R->ncols() );
                size_t        mem_zfp = 0;
                auto          U_zfp   = std::vector< uv_real_t >( R->rank() * R->nrows() * factor );
                auto          V_zfp   = std::vector< uv_real_t >( R->rank() * R->ncols() * factor );

                {
                    auto  uU = zfp::const_array2< uv_real_t >( R->nrows(), R->rank() * factor, config, 0, 0 );
        
                    uU.set( (uv_real_t*) UV.U.data() );

                    mem_zfp += uU.compressed_size();

                    uU.get( U_zfp.data() );
                }

                {
                    auto    uV = zfp::const_array2< uv_real_t >( R->ncols(), R->rank() * factor, config, 0, 0 );

                    uV.set( (uv_real_t*) UV.V.data() );

                    mem_zfp += uV.compressed_size();

                    uV.get( V_zfp.data() );
                }
        
                if ( mem_zfp < mem_lr )
                {
                    // std::cout << "R : " << mem_zfp << " , " << mem_lr << std::endl;
        
                    s += mem_zfp;
                    s += 2*sizeof(zfp::const_array2< uv_real_t >);

                    memcpy( (uv_real_t*) UV.U.data(), U_zfp.data(), sizeof(uv_real_t) * U_zfp.size() );
                    memcpy( (uv_real_t*) UV.V.data(), V_zfp.data(), sizeof(uv_real_t) * V_zfp.size() );
                }// if
                else
                    s = R->byte_size();
            },
            R->factors()
        );

        return s;
    }// if
    #if defined(HAS_H2)
    else if ( is_uniform( &A ) )
    {
        auto          R       = ptrcast( &A, hpro::TUniformMatrix );
        const size_t  mem_lr  = sizeof(value_t) * R->row_rank() * R->col_rank();
        size_t        s       = A.byte_size() - mem_lr - sizeof(blas::matrix< value_t >);
        auto          zC      = zfp::const_array2< value_t >( R->row_rank(), R->col_rank(), config, 0, 0 );
        
        zC.set( hpro::coeff< value_t >( R ).data() );

        const size_t  mem_zfp = zC.compressed_size();

        if ( mem_zfp < mem_lr )
        {
            s += mem_zfp;
            s += sizeof(zfp::const_array2< value_t >);

            // write back compressed data
            zC.get( hpro::coeff< value_t >( R ).data() );

            return s;
        }// if
        else
            return A.byte_size();
    }// if
    #endif
    else if ( is_uniform_lowrank( A ) )
    {
        auto          R       = ptrcast( &A, uniform_lrmatrix< value_t > );
        const size_t  mem_lr  = sizeof(value_t) * R->row_rank() * R->col_rank();
        size_t        s       = A.byte_size() - mem_lr - sizeof(blas::matrix< value_t >);
        auto          zC      = zfp::const_array2< value_t >( R->row_rank(), R->col_rank(), config, 0, 0 );
        
        zC.set( R->coeff().data() );

        const size_t  mem_zfp = zC.compressed_size();

        if ( mem_zfp < mem_lr )
        {
            s += mem_zfp;
            s += sizeof(zfp::const_array2< value_t >);

            // write back compressed data
            zC.get( R->coeff().data() );

            return s;
        }// if
        else
            return A.byte_size();
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    return 0;
}

template < typename value_t >
size_t
convert_zfp ( cluster_basis< value_t > &  cb,
              zfp_config &                config )
{
    //
    // convert local basis
    //

    size_t  s = sizeof( cluster_basis< value_t > );
    
    if ( cb.rank() > 0 )
    {
        auto  C  = cb.basis();
        auto  zC = zfp::const_array2< value_t >( C.nrows(), C.ncols(), config, 0, 0 );
        
        zC.set( C.data() );

        const size_t  mem_dense = sizeof(value_t) * C.nrows() * C.ncols();
        const size_t  mem_zfp   = zC.compressed_size();

        if ( mem_zfp < mem_dense )
        {
            s += mem_zfp + sizeof(zfp::const_array2< value_t >) - sizeof(blas::matrix< value_t >);
            
            // write back compressed data
            zC.get( C.data() );
        }// if
        else
        {
            s += mem_dense;
        }// else
    }// if
    
    if ( cb.nsons() > 0 )
    {
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
            {
                s += convert_zfp< value_t >( * cb.son(i), config );
            }// if
        }// for
    }// if

    return s;
}

#if defined(HAS_H2)
template < typename value_t >
size_t
convert_zfp ( hpro::TClusterBasis< value_t > &  cb,
              zfp_config &                      config )
{
    //
    // convert local basis
    //

    size_t  s = sizeof( cluster_basis< value_t > );
    
    if ( cb.nsons() == 0 )
    {
        auto  C  = cb.basis();
        auto  zC = zfp::const_array2< value_t >( C.nrows(), C.ncols(), config, 0, 0 );
        
        zC.set( C.data() );

        const size_t  mem_dense = sizeof(value_t) * C.nrows() * C.ncols();
        const size_t  mem_zfp   = zC.compressed_size();

        if ( mem_zfp < mem_dense )
        {
            s += mem_zfp + sizeof(zfp::const_array2< value_t >) - sizeof(blas::matrix< value_t >);
            
            // write back compressed data
            zC.get( C.data() );
        }// if
        else
        {
            s += mem_dense;
        }// else
    }// if
    else 
    {
        //
        // recurse
        //
        
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
            {
                s += convert_zfp< value_t >( * cb.son(i), config );
            }// if
        }// for

        //
        // compress transfer matrices
        //

        if ( cb.rank() > 0 )
        {
            for ( uint  i = 0; i < cb.nsons(); ++i )
            {
                auto  T  = cb.transfer_mat( i );
                auto  zT = zfp::const_array2< value_t >( T.nrows(), T.ncols(), config, 0, 0 );
        
                zT.set( T.data() );

                const size_t  mem_dense = sizeof(value_t) * T.nrows() * T.ncols();
                const size_t  mem_zfp   = zT.compressed_size();

                if ( mem_zfp < mem_dense )
                {
                    s += mem_zfp + sizeof(zfp::const_array2< value_t >) - sizeof(blas::matrix< value_t >);
            
                    // write back compressed data
                    zT.get( T.data() );
                }// if
                else
                {
                    s += mem_dense;
                }// else
            }// for
        }// if
    }// if

    return s;
}
#endif

#endif

#if defined(HAS_UNIVERSAL)
//
// compress data using ZFP and return memory consumption
//
template < uint bitsize,
           uint expsize >
size_t
convert_posit ( hpro::TMatrix &  A )
{
    using  posit_t = sw::universal::posit< bitsize, expsize >;
    
    if ( is_blocked( A ) )
    {
        auto    B = ptrcast( &A, hpro::TBlockMatrix );
        size_t  s = sizeof(hpro::TBlockMatrix);

        s += B->nblock_rows() * B->nblock_cols() * sizeof(hpro::TMatrix *);
        
        for ( uint i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint j = 0; j < B->nblock_cols(); ++j )
            {
                s += convert_posit< bitsize, expsize >( * B->block(i,j) );
            }// for
        }// for

        return s;
    }// if
    else if ( is_dense( A ) )
    {
        auto    D = ptrcast( &A, hpro::TDenseMatrix );
        size_t  s = D->byte_size();

        if ( D->is_complex() )
        {
            using  value_t = hpro::complex;
            using  real_t  = hpro::real;
            
            auto  M = blas::mat< value_t >( D );
        
            s -= ( sizeof(value_t) * D->nrows() * D->ncols() );
            s += size_t( std::ceil( ( bitsize * D->nrows() * D->ncols() ) / 8.0 ) );

            for ( uint  j = 0; j < M.ncols(); ++j )
            {
                for ( uint  i = 0; i < M.nrows(); ++i )
                {
                    auto  pr_ij = posit_t( std::real( M(i,j) ) );
                    auto  pi_ij = posit_t( std::imag( M(i,j) ) );
                    
                    M(i,j) = value_t( real_t( pr_ij ), real_t( pi_ij ) );
                }// for
            }// for
        }// if
        else
        {
            using  value_t = hpro::real;
            
            auto  M = blas::mat< value_t >( D );
        
            s -= ( sizeof(value_t) * D->nrows() * D->ncols() );
            s += size_t( std::ceil( ( bitsize * D->nrows() * D->ncols() ) / 8.0 ) );

            for ( uint  j = 0; j < M.ncols(); ++j )
            {
                for ( uint  i = 0; i < M.nrows(); ++i )
                {
                    auto  p_ij = posit_t( M(i,j) );
                    
                    M(i,j) = value_t(p_ij);
                }// for
            }// for
        }// else
        
        return s;
    }// if
    else if ( matrix::is_generic_dense( A ) )
    {
        auto    D = ptrcast( &A, matrix::dense_matrix );
        size_t  s = 0;

        std::visit(
            [&s,D] ( auto &&  M )
            {
                using  value_t = typename std::decay_t< decltype(M) >::value_t;

                if constexpr( ! hpro::is_complex_type_v< value_t > )
                {
                    s  = D->byte_size();
                    s -= ( sizeof(value_t) * D->nrows() * D->ncols() );
                    s += size_t( std::ceil( ( bitsize * D->nrows() * D->ncols() ) / 8.0 ) );

                    for ( uint  j = 0; j < M.ncols(); ++j )
                    {
                        for ( uint  i = 0; i < M.nrows(); ++i )
                        {
                            auto  p_ij = posit_t( M(i,j) );
                            
                            M(i,j) = double(p_ij);
                        }// for
                    }// for
                }// if
            },
            D->matrix()
        );
        
        return s;
    }// if
    else if ( is_lowrank( A ) )
    {
        auto    R = ptrcast( &A, hpro::TRkMatrix );
        size_t  s = R->byte_size();

        if ( R->is_complex() )
        {
            using  value_t = hpro::complex;
            using  real_t  = hpro::real;

            auto  U = blas::mat_U< value_t >( R );
            auto  V = blas::mat_V< value_t >( R );
        
            s -= sizeof(value_t) * R->rank() * ( R->nrows() + R->ncols() );
            s += size_t( std::ceil( bitsize * R->rank() * ( R->nrows() + R->ncols() ) / 8.0 ) );
                
            for ( uint  k = 0; k < U.ncols(); ++k )
            {
                for ( uint  i = 0; i < U.nrows(); ++i )
                {
                    auto  pr_ij = posit_t( std::real( U(i,k) ) );
                    auto  pi_ij = posit_t( std::imag( U(i,k) ) );
                            
                    U(i,k) = value_t( real_t( pr_ij ), real_t( pi_ij ) );
                }// for
            }// for

            for ( uint  k = 0; k < V.ncols(); ++k )
            {
                for ( uint  i = 0; i < V.nrows(); ++i )
                {
                    auto  pr_ij = posit_t( std::real( V(i,k) ) );
                    auto  pi_ij = posit_t( std::imag( V(i,k) ) );
                            
                    V(i,k) = value_t( real_t( pr_ij ), real_t( pi_ij ) );
                }// for
            }// for
        }// if
        else
        {
            using  value_t = hpro::real;
            
            auto  U = blas::mat_U< value_t >( R );
            auto  V = blas::mat_V< value_t >( R );
        
            s -= sizeof(value_t) * R->rank() * ( R->nrows() + R->ncols() );
            s += size_t( std::ceil( bitsize * R->rank() * ( R->nrows() + R->ncols() ) / 8.0 ) );
                
            for ( uint  k = 0; k < U.ncols(); ++k )
            {
                for ( uint  i = 0; i < U.nrows(); ++i )
                {
                    auto  p_ij = posit_t( U(i,k) );
                            
                    U(i,k) = value_t(p_ij);
                }// for
            }// for

            for ( uint  k = 0; k < V.ncols(); ++k )
            {
                for ( uint  i = 0; i < V.nrows(); ++i )
                {
                    auto  p_ij = posit_t( V(i,k) );
                            
                    V(i,k) = value_t(p_ij);
                }// for
            }// for
        }// else
        
        
        return s;
    }// if
    else if ( hlr::matrix::is_generic_lowrank( A ) )
    {
        auto    R = ptrcast( &A, matrix::lrmatrix );
        size_t  s = 0;

        std::visit(
            [R,&s] ( auto &&  UV )
            {
                using  value_t = typename std::decay_t< decltype(UV) >::value_t;
                
                if constexpr( ! hpro::is_complex_type_v< value_t > )
                {
                    s  = R->byte_size();
                    s -= sizeof(value_t) * R->rank() * ( R->nrows() + R->ncols() );
                    s += size_t( std::ceil( bitsize * R->rank() * ( R->nrows() + R->ncols() ) / 8.0 ) );
                
                    for ( uint  k = 0; k < UV.U.ncols(); ++k )
                    {
                        for ( uint  i = 0; i < UV.U.nrows(); ++i )
                        {
                            auto  p_ij = posit_t( UV.U(i,k) );
                            
                            UV.U(i,k) = double(p_ij);
                        }// for
                    }// for

                    for ( uint  k = 0; k < UV.V.ncols(); ++k )
                    {
                        for ( uint  i = 0; i < UV.V.nrows(); ++i )
                        {
                            auto  p_ij = posit_t( UV.V(i,k) );
                            
                            UV.V(i,k) = double(p_ij);
                        }// for
                    }// for
                }// if
            },
            R->factors()
        );

        return s;
    }// if
    #if defined(HAS_H2)
    else if ( is_uniform( &A ) )
    {
        auto    R = ptrcast( &A, hpro::TUniformMatrix );
        size_t  s = A.byte_size();

        if ( R->is_complex() )
        {
            using  value_t = hpro::complex;
            using  real_t  = hpro::real;

            auto  C = hpro::coeff< value_t >( R );
            
            s -= sizeof(value_t) * R->row_rank() * R->col_rank();
            s += size_t( std::ceil( bitsize * R->row_rank() * R->col_rank() / 8.0 ) );
        
            for ( uint  j = 0; j < C.ncols(); ++j )
            {
                for ( uint  i = 0; i < C.nrows(); ++i )
                {
                    auto  pr_ij = posit_t( std::real( C(i,j) ) );
                    auto  pi_ij = posit_t( std::imag( C(i,j) ) );
                    
                    C(i,j) = value_t( real_t( pr_ij ), real_t( pi_ij ) );
                }// for
            }// for
        }// if
        else
        {
            using  value_t = hpro::real;
            
            auto  C = hpro::coeff< value_t >( R );
            
            s -= sizeof(value_t) * R->row_rank() * R->col_rank();
            s += size_t( std::ceil( bitsize * R->row_rank() * R->col_rank() / 8.0 ) );
        
            for ( uint  j = 0; j < C.ncols(); ++j )
            {
                for ( uint  i = 0; i < C.nrows(); ++i )
                {
                    auto  p_ij = posit_t( std::real( C(i,j) ) );
                    
                    C(i,j) = value_t( p_ij );
                }// for
            }// for
        }// else
        
        return s;
    }// if
    #endif
    else if ( is_uniform_lowrank( A ) )
    {
        // TODO: support float/complex< float >
        if ( A.is_complex() )
        {
            using  value_t = std::complex< double >;
            using  real_t  = double;
            
            auto    R = ptrcast( &A, uniform_lrmatrix< value_t > );
            size_t  s = A.byte_size();
            auto    C = R->coeff();

            s -= sizeof(value_t) * R->row_rank() * R->col_rank();
            s += size_t( std::ceil( bitsize * R->row_rank() * R->col_rank() / 8.0 ) );
        
            for ( uint  j = 0; j < C.ncols(); ++j )
            {
                for ( uint  i = 0; i < C.nrows(); ++i )
                {
                    auto  pr_ij = posit_t( std::real( C(i,j) ) );
                    auto  pi_ij = posit_t( std::imag( C(i,j) ) );
                            
                    C(i,j) = value_t( real_t( pr_ij ), real_t( pi_ij ) );
                }// for
            }// for

            return s;
        }// if
        else
        {
            using  value_t = double;

            auto    R = ptrcast( &A, uniform_lrmatrix< value_t > );
            size_t  s = A.byte_size();
            auto    C = R->coeff();

            s -= sizeof(value_t) * R->row_rank() * R->col_rank();
            s += size_t( std::ceil( bitsize * R->row_rank() * R->col_rank() / 8.0 ) );
        
            for ( uint  j = 0; j < C.ncols(); ++j )
            {
                for ( uint  i = 0; i < C.nrows(); ++i )
                {
                    auto  p_ij = posit_t( C(i,j) );
                            
                    C(i,j) = value_t(p_ij);
                }// for
            }// for

            return s;
        }// else
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    return 0;
}

template < uint bitsize,
           uint expsize,
           typename value_t >
size_t
convert_posit ( cluster_basis< value_t > &  cb )
{
    using  posit_t = sw::universal::posit< bitsize, expsize >;

    //
    // convert local basis
    //

    size_t  s = sizeof( cluster_basis< value_t > );
    
    if ( cb.rank() > 0 )
    {
        auto  C = cb.basis();

        s += size_t( std::ceil( ( bitsize * C.nrows() * C.ncols() ) / 8.0 ) );

        for ( uint  j = 0; j < C.ncols(); ++j )
        {
            for ( uint  i = 0; i < C.nrows(); ++i )
            {
                auto  p_ij = posit_t( C(i,j) );
                    
                C(i,j) = value_t(p_ij);
            }// for
        }// for
    }// if
    
    if ( cb.nsons() > 0 )
    {
        auto  mtx = std::mutex();
        
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
            {
                auto  s_i = convert_posit< bitsize, expsize >( * cb.son(i) );
                
                {
                    auto  lock = std::scoped_lock( mtx );
                    
                    s += s_i;
                }
            }// if
        }// for
    }// if

    return s;
}

#if defined(HAS_H2)
template < uint bitsize,
           uint expsize,
           typename value_t >
size_t
convert_posit ( hpro::TClusterBasis< value_t > &  cb )
{
    using  posit_t = sw::universal::posit< bitsize, expsize >;

    //
    // convert local basis
    //

    size_t  s = sizeof( cluster_basis< value_t > );
    
    if ( cb.nsons() == 0 )
    {
        auto  C  = cb.basis();

        s += size_t( std::ceil( ( bitsize * C.nrows() * C.ncols() ) / 8.0 ) );

        for ( uint  j = 0; j < C.ncols(); ++j )
        {
            for ( uint  i = 0; i < C.nrows(); ++i )
            {
                auto  p_ij = posit_t( C(i,j) );
                    
                C(i,j) = value_t(p_ij);
            }// for
        }// for
    }// if
    else 
    {
        //
        // recurse
        //
        
        for ( uint  i = 0; i < cb.nsons(); ++i )
        {
            if ( ! is_null( cb.son( i ) ) )
            {
                s += convert_posit< bitsize, expsize >( * cb.son(i) );
            }// if
        }// for

        //
        // compress transfer matrices
        //

        if ( cb.rank() > 0 )
        {
            for ( uint  i = 0; i < cb.nsons(); ++i )
            {
                auto  T = cb.transfer_mat( i );

                s += sizeof( T );
                s += size_t( std::ceil( ( bitsize * T.nrows() * T.ncols() ) / 8.0 ) );

                for ( uint  j = 0; j < T.ncols(); ++j )
                {
                    for ( uint  i = 0; i < T.nrows(); ++i )
                    {
                        auto  p_ij = posit_t( T(i,j) );
                    
                        T(i,j) = value_t(p_ij);
                    }// for
                }// for
            }// for
        }// if
    }// if

    return s;
}
#endif

#endif

}}}// namespace hlr::seq::matrix

#endif // __HLR_SEQ_CONVERT_HH
