//
// Project     : HLR
// Program     : dpt
// Description : testing DPT eigenvalue algorithmus
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <tbb/parallel_for.h>

#include <hpro/algebra/mat_mul_diag.hh>
#include <hpro/algebra/mat_add.hh>

#include <hlr/utils/likwid.hh>

#include "hlr/bem/aca.hh"
#include "hlr/seq/norm.hh"
#include <hlr/vector/scalar_vector.hh>
#include <hlr/matrix/print.hh>
#include <hlr/approx/svd.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// return copy of (block-wise/element-wise) diagonal of matrix
//
template < typename value_t >
void
extract_diag ( const hpro::TMatrix &      M,
               blas::vector< value_t > &  diag )
{
    if ( is_blocked( M ) )
    {
        auto  B = cptrcast( &M, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( B->block( i, i ) != nullptr )
                extract_diag( * B->block( i, i ), diag );
        }// for
    }// if
    else
    {
        if ( is_dense( M ) )
        {
            auto        D      = cptrcast( &M, hpro::TDenseMatrix );
            auto        blas_D = hpro::blas_mat< value_t >( D );
            const auto  nrows  = D->nrows();
            const auto  ncols  = D->ncols();
        
            for ( size_t  i = 0; i < std::min( nrows, ncols ); ++i )
                diag( M.row_ofs() + i ) = blas_D( i, i );
        }// if
        else
            HLR_ERROR( "only dense format supported" );
    }// else
}

//
// add given diagonal to matrix
//
template < typename value_t >
void
add_diag ( hpro::TMatrix &                  M,
           const blas::vector< value_t > &  diag )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( B->block( i, i ) != nullptr )
                add_diag( * B->block( i, i ), diag );
        }// for
    }// if
    else
    {
        if ( is_dense( M ) )
        {
            auto        D      = ptrcast( &M, hpro::TDenseMatrix );
            auto        blas_D = hpro::blas_mat< value_t >( D );
            const auto  nrows  = D->nrows();
            const auto  ncols  = D->ncols();
        
            for ( size_t  i = 0; i < std::min( nrows, ncols ); ++i )
                blas_D( i, i ) += diag( M.row_ofs() + i );
        }// if
        else
            HLR_ERROR( "only dense format supported" );
    }// else
}

template < typename T_value >
class theta_coeff_fn : public hpro::TCoeffFn< T_value >
{
public:
    using  value_t = T_value;

private:
    const blas::vector< value_t >  v_row;
    const blas::vector< value_t >  v_col;
        
public:
    theta_coeff_fn ( const blas::vector< value_t > &  av_row,
                     const blas::vector< value_t > &  av_col )
            : v_row( av_row )
            , v_col( av_col )
    {}

    virtual ~theta_coeff_fn () {}

    //
    // evaluation of coefficient
    //
    virtual void eval  ( const hpro::TIndexSet &  rowis,
                         const hpro::TIndexSet &  colis,
                         value_t *                matrix ) const
    {
        const auto  row_ofs = rowis.first();
        const auto  col_ofs = colis.first();
        
        for ( auto  j : colis )
            for ( auto  i : rowis )
                matrix[ (j - col_ofs) * rowis.size() + (i - row_ofs) ] = value_t(1) / ( v_row(i) - v_col(j) );
    }

    virtual void eval ( const std::vector< idx_t > &  rowis,
                        const std::vector< idx_t > &  colis,
                        value_t *                     matrix ) const
    {
        for ( auto  j : colis )
            for ( auto  i : rowis )
                matrix[ j * rowis.size() + i ] = value_t(1) / ( v_row(i) - v_col(j) );
    }
};

//
// construct Θ with Θ_ij = 1 / ( v_i - v_j ) using
// block structure as defined by M
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
construct_theta ( const hpro::TMatrix &            M,
                  const blas::vector< value_t > &  v,
                  const hpro::TTruncAcc &          acc )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = construct_theta( * BM->block( i, j ), v, acc );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_dense( M ) )
    {
        auto  D      = M.copy();
        auto  blas_D = hpro::blas_mat< value_t >( ptrcast( D.get(), hpro::TDenseMatrix ) );

        for ( idx_t  j = 0; j < idx_t(D->ncols()); ++j )
            for ( idx_t  i = 0; i < idx_t(D->nrows()); ++i )
            {
                if ( i + D->row_ofs() != j + D->col_ofs() )
                    blas_D( i, j ) = value_t(1) / ( v( D->row_ofs() + i ) - v( D->col_ofs() + j ) );
                else
                    blas_D( i, j ) = value_t(0);
            }// for
        
        return D;
    }// else
    else if ( is_lowrank( M ) )
    {
        //
        // extract vectors corresponding to rowis(M) and colis(M)
        // and define coefficient function based on these to be
        // used by ACA
        //

        blas::vector      v_row( v, M.row_is() );
        blas::vector      v_col( v, M.col_is() );
        theta_coeff_fn    theta_coeff( v_row, v_col );
        auto              op = operator_wrapper( hpro::bis( hpro::is( 0, M.nrows()-1 ),
                                                            hpro::is( 0, M.ncols()-1 ) ),
                                                 theta_coeff );
        auto              pivot_search = approx::aca_pivot( op );
        

        auto [ U, V ] = approx::aca( op, pivot_search, acc, nullptr );
            
        return std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is(), std::move( U ), std::move( V ) );
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// construct identity matrix with same diagonal structure as given matrix
// - all low-rank matrices have zero rank
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
construct_identity ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = construct_identity< value_t >( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_dense( M ) )
    {
        auto  D      = M.copy();
        auto  blas_D = hpro::blas_mat< value_t >( ptrcast( D.get(), hpro::TDenseMatrix ) );

        D->scale( 0 );

        if ( M.row_is() == M.col_is() )
        {
            for ( idx_t  i = 0; i < std::min< idx_t >( D->nrows(), D->ncols() ); ++i )
                blas_D( i, i ) = value_t(1);
        }// if
        
        return D;
    }// else
    else if ( is_lowrank( M ) )
    {
        auto  R = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is() );

        R->set_complex( M.is_complex() );

        return R;
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// construct zero matrix with same diagonal structure as <M>
//
inline
std::unique_ptr< hpro::TMatrix >
construct_zero ( const hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = construct_zero( * BM->block( i, j ) );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_dense( M ) )
    {
        auto  D = M.copy();

        D->scale( 0 );
        
        return D;
    }// else
    else if ( is_lowrank( M ) )
    {
        auto  R = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is() );

        R->set_complex( M.is_complex() );

        return R;
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// construct diagonal matrix with same diagonal structure as <M>
// and diagonal defined by <diag>
// - all low-rank matrices have zero rank
//
template < typename value_t >
std::unique_ptr< hpro::TMatrix >
construct_diag ( const hpro::TMatrix &            M,
                 const blas::vector< value_t > &  diag )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, hpro::TBlockMatrix );
        auto  N  = std::make_unique< hpro::TBlockMatrix >();
        auto  B  = ptrcast( N.get(), hpro::TBlockMatrix );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = construct_diag< value_t >( * BM->block( i, j ), diag );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for
        
        return N;
    }// if
    else if ( is_dense( M ) )
    {
        auto  D      = M.copy();
        auto  blas_D = hpro::blas_mat< value_t >( ptrcast( D.get(), hpro::TDenseMatrix ) );

        D->scale( 0 );

        if ( M.row_is() == M.col_is() )
        {
            for ( idx_t  i = 0; i < std::min< idx_t >( D->nrows(), D->ncols() ); ++i )
                blas_D( i, i ) = diag( M.row_ofs() + i );
        }// if
        
        return D;
    }// else
    else if ( is_lowrank( M ) )
    {
        auto  R = std::make_unique< hpro::TRkMatrix >( M.row_is(), M.col_is() );

        R->set_complex( M.is_complex() );

        return R;
    }// if
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// set diagonal to zero
//
template < typename value_t >
void
zero_diag ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( B->block( i, i ) != nullptr )
                zero_diag< value_t >( *B->block( i, i ) );
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D      = ptrcast( &M, hpro::TDenseMatrix );
        auto  blas_D = hpro::blas_mat< value_t >( D );

        for ( idx_t  i = 0; i < std::min< idx_t >( D->nrows(), D->ncols() ); ++i )
            blas_D( i, i ) = value_t(0);
    }// else
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// set diagonal to zero
//
template < typename value_t >
void
add_identity ( hpro::TMatrix &  M )
{
    if ( is_blocked( M ) )
    {
        auto  B = ptrcast( &M, hpro::TBlockMatrix );
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( B->block( i, i ) != nullptr )
                add_identity< value_t >( *B->block( i, i ) );
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto  D      = ptrcast( &M, hpro::TDenseMatrix );
        auto  blas_D = hpro::blas_mat< value_t >( D );

        for ( idx_t  i = 0; i < std::min< idx_t >( D->nrows(), D->ncols() ); ++i )
            blas_D( i, i ) += value_t(1);
    }// else
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// add α,α+1,...,α+n to m_ii
//
template < typename value_t >
void
add_iota_diag ( hpro::TMatrix &  M,
                const double     alpha,
                const double     ofs = 0.0 )
{
    if ( is_blocked( M ) )
    {
        auto    B = ptrcast( &M, hpro::TBlockMatrix );
        double  f = ofs;
        
        for ( uint  i = 0; i < std::min( B->nblock_rows(), B->nblock_cols() ); ++i )
        {
            if ( B->block( i, i ) != nullptr )
            {
                add_iota_diag< value_t >( *B->block( i, i ), alpha, f );

                f += alpha * std::min( B->block( i, i )->nrows(),
                                       B->block( i, i )->ncols() );
            }// if
        }// for
    }// if
    else if ( is_dense( M ) )
    {
        auto    D      = ptrcast( &M, hpro::TDenseMatrix );
        auto    blas_D = hpro::blas_mat< value_t >( D );
        double  f      = ofs;

        for ( idx_t  i = 0; i < std::min< idx_t >( D->nrows(), D->ncols() ); ++i, f += alpha )
            blas_D( i, i ) += f;
    }// else
    else
        HLR_ERROR( "unsupported matrix type : " + M.typestr() );
}

//
// return norm | M V · V E | / ( |M|·|V| )
//
template < typename value_t >
typename hpro::real_type< value_t >::type_t
everror ( const hpro::TMatrix &            M,
          const hpro::TMatrix &            V,
          const blas::vector< value_t > &  E,
          const hpro::TTruncAcc &          acc )
{
    using real_t = typename hpro::real_type< value_t >::type_t;

    const auto  normM = real_t(1);
    const auto  normV = real_t(1);
    auto        T     = construct_zero( M );
    auto        apx   = approx::SVD< value_t >();

    impl::multiply( value_t(1),
                    hpro::apply_normal, M,
                    hpro::apply_normal, V,
                    *T, acc, apx );

    auto  EM = construct_diag( M, E );
    
    impl::multiply( value_t(-1),
                    hpro::apply_normal, V,
                    hpro::apply_normal, *EM,
                    *T, acc, apx );

    return seq::norm::frobenius( *T ) / ( normM * normV );
}

//
// compute eigen values of given matrix M using DPT iteration
//
//   - stop iteration if |V_i - V_i-1| < tol or i > max_it
//   - if tol == -1, then machine precision w.r.t. value_t is chosen
//   - if max_it == 0, then max_it = 100 is set
// 
template < typename value_t >
std::pair< blas::vector< value_t >,
           std::unique_ptr< hpro::TMatrix > >
dpteigen ( hpro::TMatrix &      M,
           const hpro::TTruncAcc &    acc,
           const double         tol        = -1,
           const size_t         max_it     = 0,
           const std::string &  error_type = "frobenius",
           const int            verbosity  = 1 )
{
    using real_t = typename hpro::real_type< value_t >::type_t;
    
    auto  tic = timer::now();

    //
    // Θ_ij = 1 / ( M_ii - M_jj )
    //

    // hpro::DBG::write( M, "M.mat", "M" );
    
    auto  diag_M = blas::vector< value_t >( M.nrows() );

    extract_diag< value_t >( M, diag_M );

    // hpro::DBG::write( diag_M, "diagM.mat", "diagM" );
    
    auto  Theta = construct_theta< value_t >( M, diag_M, acc );

    // hpro::DBG::write( *Theta, "Theta.mat", "Theta" );
    
    //
    // Δ = M - diag(M)
    //
    
    auto  Delta = impl::matrix::copy( M );

    zero_diag< value_t >( *Delta );

    // hpro::DBG::write( Delta.get(), "Delta.mat", "Delta" );
    
    //
    // V = I
    //

    auto  V   = construct_identity< value_t >( M );

    // hpro::DBG::write( *V, "V.mat", "V" );
    
    auto  toc = timer::since( tic );

    if ( verbosity >= 1 )
        std::cout << "  time for setup = " << boost::format( "%.3es" ) % toc << std::endl;
    
    //
    // iteration
    //

    auto           diag_T    = hpro::TScalarVector( M.row_is() );
    const real_t   precision = ( tol < 0
                                 ? real_t(10) * std::numeric_limits< real_t >::epsilon()
                                 : tol );
    const size_t   max_steps = ( max_it == 0 ? 100 : max_it );
    auto           T         = impl::matrix::copy( M );
    size_t         nsteps    = 0;
    real_t         old_error = 0;
    auto           apx       = hlr::approx::SVD< value_t >();

    tic = timer::now();

    do
    {
        auto  tic2 = timer::now();
        
        // T = Δ·V
        // hpro::DBG::write( Delta.get(), "Delta.mat", "Delta" );
        // hpro::DBG::write( V.get(), "V.mat", "V" );
        auto  tic3 = timer::now();

        hpro::multiply( value_t(1), hpro::apply_normal, Delta.get(), hpro::apply_normal, V.get(), value_t(0), T.get(), acc );

        auto  toc3 = timer::since( tic3 );

        std::cout << "      gemm     in " << format_time( toc3 ) << std::endl;

        // hpro::DBG::write( T.get(), "T1.mat", "T1" );
        
        // T = Δ·V - V·diag(Δ·V) = T - V·diag(T)
        auto  Vc = impl::matrix::copy( *V );
        
        extract_diag< value_t >( *T, hpro::blas_vec< value_t >( diag_T ) );
        hpro::mul_diag_right( Vc.get(), diag_T );
        // hpro::DBG::write( Vc.get(), "Vc.mat", "Vc" );
        hpro::add( value_t(-1), Vc.get(), value_t(1), T.get(), acc );
        // hpro::DBG::write( T.get(), "T2.mat", "T2" );

        // // I - Θ ∗ (Δ·V - V·diag(Δ·V)) = I - Θ ∗ T
        tic3 = timer::now();
        impl::multiply_hadamard( value_t(-1), *T, *Theta, acc, apx );
        toc3 = timer::since( tic3 );
        std::cout << "      hadamard in " << format_time( toc3 ) << std::endl;
        // hpro::DBG::write( T.get(), "T3.mat", "T3" );
        add_identity< value_t >( *T );
        // hpro::DBG::write( T.get(), "T4.mat", "T4" );
        // hpro::DBG::write( V.get(), "V.mat", "V" );

        //
        // compute error ||V-T||_F
        //

        real_t  error = 0;

        if (( error_type == "frobenius" ) || ( error_type == "fro" ))
        {
            error = seq::norm::frobenius( -1.0, *T, 1.0, *V );
        }// if
        else
            throw std::runtime_error( "unknown error type" );

        //
        // test stop criterion
        //

        std::swap( T, V );
        // impl::matrix::copy_to( *T, *V );

        auto  toc2 = timer::since( tic2 );
            
        if ( verbosity >= 1 )
        {
            std::cout << "    step " << boost::format( "%03d" ) % nsteps
                      << " : error = " << boost::format( "%.4e" ) % error;

            if ( nsteps > 0 )
                std::cout << ", reduction = " << boost::format( "%.4e" ) % ( error / old_error );
            
            std::cout << ", time = " << format_time( toc2 )
                      << ", size(V) = " << format_mem( V->byte_size() ) << std::endl;
        }// if
        
        old_error = error;
        
        ++nsteps;
        
        if ( error < precision )
            break;

        if ( ! std::isnormal( error ) )
            break;
        
    } while ( nsteps < max_steps );

    toc = timer::since( tic );

    if ( verbosity >= 1 )
        std::cout << "  time for iteration = " << boost::format( "%.3es" ) % toc << std::endl;

    //
    // eigenvalues  : diag( M + Δ·V )
    // eigenvectors : V
    //

    blas::vector< value_t >  E( M.nrows() );

    tic = timer::now();

    hpro::multiply( value_t(1), hpro::apply_normal, Delta.get(), hpro::apply_normal, V.get(), value_t(1), & M, acc );
    extract_diag< value_t >( M, E );

    toc = timer::since( tic );

    if ( verbosity >= 1 )
        std::cout << "  time for postproc  = " << boost::format( "%.3es" ) % toc << std::endl;
    
    return { std::move( E ), std::move( V ) };
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    {
        std::cout << term::bullet << term::bold << "dense DPT eigen iteration ( " << impl_name
                  << " )" << term::reset << std::endl;

        blas::eigen_stat  stat;
        
        for ( size_t  n = 2560; n <= 4096; n += 512 )
        {
            std::mutex  mtx;
            uint        nsweeps_min = 0;
            uint        nsweeps_jac = 0;

            ::tbb::parallel_for( uint(0), uint(10),
                                 [&,n] ( const uint )
                                 {
                                     auto  R  = blas::random< value_t >( n, n );
                                     auto  M  = blas::prod( value_t(1), R, blas::adjoint(R) );
                                     auto  Mc = blas::copy( M );

                                     for ( uint nsweeps = 0; nsweeps < n; ++nsweeps )
                                     {
                                         {
                                             auto  [ E, V ] = blas::eigen_jac( M, 1, 1e-14 );
                                         }
                                         
                                         auto  T = blas::copy( M );
                                         
                                         auto  [ E, V ] = blas::eigen_dpt( T, 0, 1e-8, "fro", 0, & stat );
                                         
                                         if ( stat.converged )
                                         {
                                             // converged
                                             std::scoped_lock  lock( mtx );
                                             
                                             nsweeps_min = std::max( nsweeps_min, nsweeps+1 );
                                             break;
                                         }// if
                                     }// for

                                     auto  [ E, V ] = blas::eigen_jac( Mc, 100, 1e-14, & stat );

                                     {
                                         std::scoped_lock  lock( mtx );
                                         
                                         nsweeps_jac = std::max( nsweeps_jac, stat.nsweeps );
                                     }
                                 } );

            std::cout << "n = " << n << "   " << nsweeps_min << "    " << nsweeps_jac << std::endl;
        }// for

        return;
    }
    
    {
        std::cout << term::bullet << term::bold << "dense DPT eigen iteration ( " << impl_name
                  << " )" << term::reset << std::endl;

        auto  R = blas::random< value_t >( cmdline::n, cmdline::n );
        auto  M = blas::prod( value_t(1), R, blas::adjoint(R) );
        
        hpro::DBG::write( M, "M.mat", "M" );
        
        // {
        //     auto  [ E, V ] = blas::eigen_jac( M, 100, cmdline::eps );

        //     hpro::DBG::write( E, "E1.mat", "E1" );
        //     hpro::DBG::write( V, "V1.mat", "V1" );
        // }

        blas::make_diag_dom( M, 10000, cmdline::eps );
        
        hpro::DBG::write( M, "M2.mat", "M2" );
        
        auto  [ E, V ] = blas::eigen_dpt( M, 0, 1e-8, "fro", 2 );

        hpro::DBG::write( E, "E.mat", "E" );
        hpro::DBG::write( V, "V.mat", "V" );

        return;
    }
    
    auto  tic     = timer::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if

    blas::reset_flops();
    
    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
        matrix::print_eps( *A, "A" );

    //////////////////////////////////////////////////////////////////////
    //
    // DPT eigenvalue iteration
    //
    //////////////////////////////////////////////////////////////////////

    {
        std::cout << term::bullet << term::bold << "DPT eigen iteration ( " << impl_name
                  << ", " << acc.to_string()
                  << " )" << term::reset << std::endl;
        
        // hpro::DBG::write( *A, "A.mat", "A" );
        
        auto  B       = impl::matrix::copy( *A );

        add_iota_diag< value_t >( *B, 1e-3 );

        if ( B->nrows() <= 2048 )
            hpro::DBG::write( *B, "M.mat", "M" );

        tic = timer::now();
        
        auto [ E, V ] = dpteigen< value_t >( *B, acc, 1e-7, 100 );

        toc = timer::since( tic );
        
        std::cout << "    done in " << format_time( toc ) << std::endl;
        std::cout << "    mem   = " << format_mem( V->byte_size() ) << std::endl;

        impl::matrix::copy_to( *A, *B );
        add_iota_diag< value_t >( *B, 1e-3 );
        std::cout << "    error = " << format_error( everror( *B, *V, E, acc ) ) << std::endl;
        
        if ( B->nrows() <= 2048 )
        {
            hpro::DBG::write( E, "E.mat", "E" );
            hpro::DBG::write( *V, "V.mat", "V" );
        }// if
    }
}
