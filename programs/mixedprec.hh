//
// Project     : HLR
// Program     : mixedprec
// Description : testing mixed precision for H
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2021. All Rights Reserved.
//

#include <fstream>
#include <limits>

#if defined(HAS_ZFP)
#include <zfpcarray2.h>
#include <zfpcarray3.h>
#endif

#include "hlr/arith/norm.hh"
#include "hlr/bem/aca.hh"
#include <hlr/matrix/print.hh>
#include <hlr/utils/io.hh>

#include <hlr/utils/eps_printer.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

#if defined(HAS_ZFP)
template < typename value_t >
size_t
convert_zfp ( hpro::TMatrix &  A,
              zfp_config &     config,
              uint             cache_size );
#endif

//
// return copy of matrix with data converted to given prevision
//
template < typename T_value_dest,
           typename T_value_src >
size_t
convert_prec ( hpro::TMatrix &  M );

//
// replace TRkMatrix by lrmatrix
//
void
convert_generic ( hpro::TMatrix &  M );

//
// print matrix <M> to file <filename>
//
void
print_prec ( const hpro::TMatrix &  M,
             const double           tol );


struct local_accuracy : public hpro::TTruncAcc
{
    local_accuracy ( const double  abs_eps )
            : hpro::TTruncAcc( 0.0, abs_eps )
    {}
    
    virtual const TTruncAcc  acc ( const indexset &  rowis,
                                   const indexset &  colis ) const
    {
        return hpro::absolute_prec( abs_eps() * std::sqrt( double(rowis.size() * colis.size()) ) );
    }
};

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    auto  acc     = gen_accuracy();
    auto  coeff   = problem->coeff_func();
    auto  pcoeff  = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );

    std::cout << "  " << term::bullet << term::bold << "nearfield" << term::reset << std::endl;
    
    auto  tic     = timer::now();
    auto  A_nf    = impl::matrix::build_nearfield( bct->root(), *pcoeff, nseq );
    auto  toc     = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A_nf->nrows() << " × " << A_nf->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A_nf->byte_size() ) << std::endl;

    // auto  norm_nf  = norm::spectral( *A_nf );
    auto  norm_nf  = norm::frobenius( *A_nf );

    std::cout << "    |A_nf| = " << format_norm( norm_nf ) << std::endl;

    auto  delta   = 10.0 * norm_nf * hlr::cmdline::eps / A_nf->nrows();
    // auto  acc2    = hpro::absolute_prec( delta );
    auto  acc2    = local_accuracy( delta );

    std::cout << "  " << term::bullet << term::bold << "H-matrix, ε = " << delta << term::reset << std::endl;
    
    auto  lrapx   = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );

    tic = timer::now();
    
    auto  A       = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc2, nseq );
    
    toc = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    print_prec( *A, acc2.abs_eps() );

    std::cout << "  " << term::bullet << term::bold << "exact matrix" << term::reset << std::endl;

    auto  acc3    = hpro::fixed_prec( 1e-12 );
    auto  dense   = std::make_unique< hpro::TSVDLRApx< value_t > >( pcoeff.get() );

    tic = timer::now();
    
    auto  A_full  = impl::matrix::build( bct->root(), *pcoeff, *dense, acc3, nseq );

    toc = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A_full->byte_size() ) << std::endl;

    auto  norm_A  = norm::spectral( *A_full );
    auto  diff    = matrix::sum( value_t(1), *A, value_t(-1), *A_full );
    auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );

    std::cout << "    error = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;

    convert_generic( *A );

    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    
    // convert_prec< float, value_t >( *A );
    
    // std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;
    
    // auto  norm_A = hlr::norm::spectral( *A, true, 1e-4 );

    std::cout << "    |A|   = " << format_norm( norm_A ) << std::endl;
    std::cout << "    |A|   = " << format_norm( hlr::norm::frobenius( *A ) ) << std::endl;


    //
    // standard single and half compression
    //
    
    {
        std::cout << "    " << term::bullet << term::bold << "single precision" << term::reset << std::endl;

        using single_t = math::decrease_precision_t< value_t >;

        auto  A2   = impl::matrix::copy( *A );
        auto  mem2 = convert_prec< single_t, value_t >( *A2 );
            
        std::cout << "      mem    = " << format_mem( mem2 ) << std::endl;
            
        auto  diff  = matrix::sum( value_t(1), *A_full, value_t(-1), *A2 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "      error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;
    }

    #if defined(HAS_HALF)
    {
        std::cout << "    " << term::bullet << term::bold << "half precision" << term::reset << std::endl;

        using single_t = math::decrease_precision_t< value_t >;
        using half_t   = math::decrease_precision_t< single_t >;

        auto  A2   = impl::matrix::copy( *A );
        auto  mem2 = convert_prec< half_t, value_t >( *A2 );
            
        std::cout << "      mem    = " << format_mem( mem2 ) << std::endl;
            
        auto  diff  = matrix::sum( value_t(1), *A_full, value_t(-1), *A2 );
        auto  error = hlr::norm::spectral( *diff, true, 1e-4 );
        
        std::cout << "      error  = " << format_error( error ) << " / " << format_error( error / norm_A ) << std::endl;
    }
    #endif

    // {
    //     auto  A2 = impl::matrix::copy( *A );

    //     convert_generic( *A2 );

    //     std::cout << "    |A|   = " << format_norm( hlr::norm::spectral( *A2, true, 1e-4 ) ) << std::endl;
    //     std::cout << "    |A|   = " << format_norm( hlr::norm::frobenius( *A2 ) ) << std::endl;
    //     std::cout << "    mem   = " << format_mem( A2->byte_size() ) << std::endl;
        
    //     auto  x = A2->col_vector();
    //     auto  y = A2->row_vector();

    //     x->fill_rand( 0 );
    //     x->scale( 1.0 / x->norm2() );

    //     A->mul_vec( 1.0, *x, 0.0, *y, apply_normal );
    //     io::matlab::write( *y, "x1" );
        
    //     A2->mul_vec( 1.0, *x, 0.0, *y, apply_normal );
    //     io::matlab::write( *y, "x2" );

    //     A->mul_vec( 1.0, *x, 0.0, *y, apply_adjoint );
    //     io::matlab::write( *y, "y1" );
        
    //     A2->mul_vec( 1.0, *x, 0.0, *y, apply_adjoint );
    //     io::matlab::write( *y, "y2" );

    //     return;
    // }

    //
    // ZFP compression
    //

    #if defined(HAS_ZFP)
    
    std::cout << "    " << term::bullet << term::bold << "ZFP compression" << term::reset << std::endl;

    // for ( uint  rate = 64; rate >= 8; rate -= 2 )
    for ( double  rate = 1e-32; rate <= 1e-2; rate *= 10 )
    {
        auto  A_zfp   = impl::matrix::copy( *A );
        // auto  config  = zfp_config_reversible();
        // auto  config  = zfp_config_rate( rate, false );
        // auto  config  = zfp_config_precision( rate );
        auto  config  = zfp_config_accuracy( rate );

        auto  mem_zfp = convert_zfp< value_t >( *A_zfp, config, 0 );
        auto  diff    = matrix::sum( value_t(1), *A_full, value_t(-1), *A_zfp );
        auto  error   = hlr::norm::spectral( *diff, true, 1e-4 );
    
        // std::cout << "      " << boost::format( "%2d" ) % rate << " / "
        std::cout << "      " << boost::format( "%.1e" ) % rate << " / "
                  << format_error( error ) << " / "
                  << format_error( error / norm_A ) << " / "
                  << format_mem( mem_zfp ) << std::endl;
    }// for
    
    #endif
}

#if defined(HAS_ZFP)
//
// compress data using ZFP and return memory consumption
//
template < typename value_t >
size_t
convert_zfp ( hpro::TMatrix &  A,
              zfp_config &     config,
              uint             cache_size )
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
                s += convert_zfp< value_t >( * B->block(i,j), config, cache_size );
            }// for
        }// for

        return s;
    }// if
    else if ( is_dense( A ) )
    {
        auto    D = ptrcast( &A, hpro::TDenseMatrix );
        size_t  s = A.byte_size() - sizeof(value_t) * D->nrows() * D->ncols() - sizeof(blas::matrix< value_t >);

        auto    u = zfp::const_array2< value_t >( D->nrows(), D->ncols(), config, 0, cache_size );

        u.set( blas::mat< value_t >( D ).data() );

        const size_t  mem_dense = sizeof(value_t) * D->nrows() * D->ncols();
        const size_t  mem_zfp   = u.compressed_size();

        if ( mem_zfp < mem_dense )
        {
            // std::cout << "D : " << mem_zfp << " , " << mem_dense << std::endl;
            
            s += u.compressed_size();
            s += sizeof(zfp::const_array2< value_t >);

            //
            // write back compressed data
            //

            // auto  T = blas::copy( blas::mat< value_t >( D ) );

            u.get( blas::mat< value_t >( D ).data() );

            // io::matlab::write( T, "T1" );
            // io::matlab::write( blas::mat< value_t >( D ), "T2" );
            
            // blas::add( value_t(-1), blas::mat< value_t >( D ), T );

            // std::cout << A.id() << " : " << blas::norm_F( T ) << std::endl;
            
            // hlr::breakpoint();
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
            [&s,D,config,cache_size] ( auto &&  M )
            {
                using  m_value_t = typename std::decay_t< decltype(M) >::value_t;
                using  m_real_t  = typename hpro::real_type_t< m_value_t >;

                uint   factor    = sizeof(m_value_t) / sizeof(m_real_t);

                s = D->byte_size() - sizeof(m_value_t) * D->nrows() * D->ncols() - sizeof(blas::matrix< m_value_t >);

                if constexpr( std::is_same_v< m_value_t, m_real_t > )
                {
                    auto  u = zfp::const_array2< m_value_t >( D->nrows(), D->ncols(), config, 0, cache_size );

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
                    auto  u = zfp::const_array3< m_real_t >( D->nrows(), D->ncols(), factor, config, 0, cache_size );

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
            auto  uU = zfp::const_array2< value_t >( R->nrows(), R->rank(), config, 0, cache_size );
        
            uU.set( blas::mat_U< value_t >( R ).data() );

            mem_zfp += uU.compressed_size();

            uU.get( U_zfp.data() );
        }

        {
            auto    uV = zfp::const_array2< value_t >( R->ncols(), R->rank(), config, 0, cache_size );

            uV.set( blas::mat_V< value_t >( R ).data() );

            mem_zfp += uV.compressed_size();

            uV.get( V_zfp.data() );
        }
        
        if ( mem_zfp < mem_lr )
        {
            // std::cout << "R : " << mem_zfp << " , " << mem_lr << std::endl;
        
            s += mem_zfp;
            s += 2*sizeof(zfp::const_array2< value_t >);

            // auto  TU = blas::copy( blas::mat_U< value_t >( R ) );
            // auto  TV = blas::copy( blas::mat_V< value_t >( R ) );

            // io::matlab::write( TU, "TU" );
            
            memcpy( blas::mat_U< value_t >( R ).data(), U_zfp.data(), sizeof(value_t) * U_zfp.size() );
            memcpy( blas::mat_V< value_t >( R ).data(), V_zfp.data(), sizeof(value_t) * V_zfp.size() );

            // io::matlab::write( blas::mat_U< value_t >( R ), "CU" );
            
            // blas::add( value_t(-1), blas::mat_U< value_t >( R ), TU );
            // blas::add( value_t(-1), blas::mat_V< value_t >( R ), TV );

            // std::cout << A.id() << " : " << blas::norm_F( TU ) << ", " << blas::norm_F( TV ) << std::endl;

            // hlr::breakpoint();
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
            [R,&s,&config,cache_size] ( auto &&  UV )
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
                    auto  uU = zfp::const_array2< uv_real_t >( R->nrows(), R->rank() * factor, config, 0, cache_size );
        
                    uU.set( (uv_real_t*) UV.U.data() );

                    mem_zfp += uU.compressed_size();

                    uU.get( U_zfp.data() );
                }

                {
                    auto    uV = zfp::const_array2< uv_real_t >( R->ncols(), R->rank() * factor, config, 0, cache_size );

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
    // else if ( is_uniform_lowrank( A ) )
    // {
    //     auto    U = ptrcast( &A, matrix::uniform_lrmatrix< value_t > );
    //     size_t  s = A.byte_size() - sizeof(value_t) * U->row_rank() * U->col_rank();

    // }// if
    else
        HLR_ERROR( "unsupported matrix type : " + A.typestr() );

    return 0;
}
#endif

//
// return copy of matrix with data converted to given prevision
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

//
// replace TRkMatrix by lrmatrix
//
void
convert_generic ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    convert_generic( *B->block( i, j ) );
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  DM = ptrcast( &M, hpro::TDenseMatrix );
        auto  D  = std::make_unique< matrix::dense_matrix >( M.row_is(), M.col_is() );

        if ( M.is_complex() )
            D->set_matrix( std::move( blas::mat< hpro::complex >( *DM ) ) );
        else
            D->set_matrix( std::move( blas::mat< hpro::real >( *DM ) ) );

        DM->parent()->replace_block( DM, D.release() );
    }// if
    else if ( is_lowrank( M ) )
    {
        auto  RM = ptrcast( &M, hpro::TRkMatrix );
        auto  R  = std::make_unique< matrix::lrmatrix >( M.row_is(), M.col_is() );

        if ( M.is_complex() )
            R->set_lrmat( std::move( blas::mat_U< hpro::complex >( *RM ) ),
                          std::move( blas::mat_V< hpro::complex >( *RM ) ) );
        else
            R->set_lrmat( std::move( blas::mat_U< hpro::real >( *RM ) ),
                          std::move( blas::mat_V< hpro::real >( *RM ) ) );

        RM->parent()->replace_block( RM, R.release() );
    }// if
}

//
// actual print function
//
void
print_prec ( const hpro::TMatrix &  M,
             eps_printer &          prn,
             const double           tol )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );

        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( ! is_null( B->block( i, j ) ) )
                    print_prec( * B->block( i, j ), prn, tol );
            }// for
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        prn.set_rgb( 85,87,83 );
        
        prn.fill_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );

        // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
    }// if
    else
    {
        // auto  norm_M = norm::spectral( M );
        auto  norm_M = norm::frobenius( M );

        if      ( norm_M <= tol / 4e-3 )   prn.set_rgb(  52, 101, 164 ); // bfloat16
        else if ( norm_M <= tol / 5e-4 )   prn.set_rgb(  15, 210,  22 ); // fp16
        else if ( norm_M <= tol / 6e-8 )   prn.set_rgb( 252, 175,  62 ); // fp32
        else if ( norm_M <= tol / 1e-16 )  prn.set_rgb( 239,  41,  41 ); // fp64
        else                               prn.set_rgb( 164,   0,   0 ); // fp128
        
        prn.fill_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );

        // draw frame
        prn.set_gray( 0 );
        prn.draw_rect( M.col_ofs(),
                       M.row_ofs(),
                       M.col_ofs() + M.ncols(),
                       M.row_ofs() + M.nrows() );
    }// else
}

//
// print matrix <M> to file <filename>
//
void
print_prec ( const hpro::TMatrix &  M,
             const double           tol )
{
    std::ofstream  out( "prec.eps" );
    eps_printer    prn( out );

    const auto   max_size = std::max( std::max( M.nrows(), M.ncols() ), size_t(1) );
    const auto   min_size = std::max( std::min( M.nrows(), M.ncols() ), size_t(1) );
    const auto   width    = ( M.ncols() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    const auto   height   = ( M.nrows() == max_size ? 500 : 500 * double(min_size) / double(max_size) );
    
    prn.begin( width, height );
    prn.scale( double(width)  / double(M.ncols()),
               double(height) / double(M.nrows()) );
    prn.translate( - double(M.col_ofs()),
                   - double(M.row_ofs()) );
    prn.set_line_width( 0.1 );
    print_prec( M, prn, tol );
    prn.end();
}
    
