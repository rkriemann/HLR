#ifndef __HLR_SEQ_IPT_HH
#define __HLR_SEQ_IPT_HH
//
// Project     : HLR
// Module      : ipt.hh
// Description : IPT eigenvalue/vector computation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <hlr/approx/traits.hh>
#include <hlr/approx/accuracy.hh>
#include <hlr/arith/multiply.hh>
#include <hlr/arith/norm.hh>
#include <hlr/seq/matrix.hh>

namespace hlr { namespace seq {

namespace detail
{

//
// compute matrix G for IPT iteration
//
template < typename value_t,
           approx::approximation_type approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_G ( const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   d,
          const accuracy &                  acc,
          const approx_t &                  apx );

}// namespace detail

//
// compute eigenvalues and eigenvectors of matrix M
// up to given tolerance via IPT iteration
// - assumption: M is near diagonal
//
template < typename value_t,
           approx::approximation_type  approx_t >
std::pair< std::unique_ptr< Hpro::TMatrix< value_t > >,
           blas::vector< value_t > >
ipt ( const Hpro::TMatrix< value_t > &  M,
      const double                      tolerance,
      const accuracy &                  acc,
      const approx_t &                  apx,
      const uint                        verbosity = 0 )
{
    using  real_t = real_type_t< value_t >;

    namespace impl = hlr::seq;
    
    //
    // initial setup:
    //
    //   D = diag(M)
    //
    //   Δ = M - D
    //
    //       ⎧ 1 / ( D_ii - D_jj ) , i ≠ j
    //   G = ⎨ 
    //       ⎩ 0                   , i = j
    //
    //   Z = I
    //

    auto  d = impl::matrix::diagonal( M );
    auto  Δ = impl::matrix::copy( M );
    auto  G = detail::build_G( M, d, acc, apx );
    auto  Z = impl::matrix::identity( M );
    auto  T = impl::matrix::copy_struct( M );

    blas::scale( value_t(-1), d );
    hlr::add_diag( *Δ, d );
        
    //
    // iteration
    //

    uint  sweep      = 0;
    uint  max_sweeps = M.nrows();
    auto  old_error  = real_t(1);
        
    do
    {
        //
        // iteration step:
        //
        //   F(Z) := I + G ⊗ ( Z·diag(Δ·Z) - Δ·Z )
        //         = I + G ⊗ ( Z·diag(T) - T )   with T = Δ·Z
        //
            
        // T = Δ·Z
        impl::multiply( value_t(1),
                        apply_normal, *Δ,
                        apply_normal, *Z,
                        *T, acc, apx );

        // {
        //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
        //     auto  DT = matrix::convert_to_dense( *T );

        //     io::matlab::write( DT->mat(), Hpro::to_string( "Ra%d", sweep ) );
        // }
        // {
        //     auto  T1 = matrix::convert_to_dense( *T );

        //     io::matlab::write( T1->mat(), "T" );
        // }

        auto  dT = impl::matrix::diagonal( *T );

        //
        // T := Z·diag(T) - T
        //
            
        // • ZT := Z·diag(T)
        auto  ZT = impl::matrix::copy( *Z );
            
        hlr::multiply_diag( *ZT, dT );
            
        // {
        //     auto  T1 = matrix::convert_to_dense( *ZT );

        //     io::matlab::write( T1->mat(), "ZT" );
        // }

        // • T := - T + ZT
        T->scale( value_t(-1) );
        impl::add( value_t(1), *ZT, *T, acc, apx );

        // {
        //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
        //     auto  DT = matrix::convert_to_dense( *T );

        //     io::matlab::write( DT->mat(), Hpro::to_string( "Rb%d", sweep ) );
        // }

        // {
        //     auto  T1 = matrix::convert_to_dense( *T );

        //     io::matlab::write( T1->mat(), "T" );
        // }

        //
        // T := I + G ⊗ T
        //
            
        // • T := G ⊗ T
        impl::multiply_hadamard( value_t(1), *T, *G, acc, apx );
                                     
        // {
        //     auto  T1 = matrix::convert_to_dense( *T );

        //     io::matlab::write( T1->mat(), "T" );
        // }

        // • T := I - T
        hlr::add_identity( *T, value_t(1) );

        // {
        //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
        //     auto  DT = matrix::convert_to_dense( *T );

        //     io::matlab::write( DT->mat(), Hpro::to_string( "Rc%d", sweep ) );
        // }

        // {
        //     auto  T1 = matrix::convert_to_dense( *T );

        //     io::matlab::write( T1->mat(), "T" );
        // }

        //
        // compute error ||Z-T||_F
        //

        impl::add( value_t(-1), *T, *Z, acc, apx );

        auto  error = norm::frobenius( *Z );

        //
        // test stop criterion
        //

        impl::matrix::copy_to( *T, *Z );

        // {
        //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
        //     auto  DZ = matrix::convert_to_dense( *Z );

        //     io::matlab::write( DZ->mat(), Hpro::to_string( "Z%d", sweep ) );
        // }

        if ( verbosity >= 1 )
        {
            std::cout << "    sweep " << sweep << " : error = " << error;

            if ( sweep > 0 )
                std::cout << ", reduction = " << error / old_error;
            
            std::cout << std::endl;
        }// if

        if (( sweep > 0 ) && ( error / old_error > real_t(10) ))
            break;
        
        old_error = error;
        
        ++sweep;

        if ( error < tolerance )
            break;

        if ( ! std::isnormal( error ) )
            break;
        
    } while ( sweep < max_sweeps );

    auto  T2 = matrix::copy( M );
            
    multiply( value_t(1),
                    apply_normal, *Δ,
                    apply_normal, *Z,
                    *T2, acc, apx );

    auto  E = impl::matrix::diagonal( *T2 );

    return { std::move( Z ), std::move( E ) };
}

namespace detail
{

//
// compute matrix G for IPT iteration
//
template < typename value_t,
           approx::approximation_type approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_G ( const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   d,
          const accuracy &                  acc,
          const approx_t &                  apx )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = build_G( * BM->block( i, j ), d, acc, apx );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  N  = M.copy();
        auto  D  = ptrcast( N.get(), matrix::dense_matrix< value_t > );
        auto  DD = D->mat();
            
        HLR_ASSERT( ! D->is_compressed() );
            
        for ( uint  j = 0; j < DD.ncols(); ++j )
        {
            auto  pj  = M.col_ofs() + j;
            auto  d_j = d( pj );
            
            for ( uint  i = 0; i < DD.nrows(); ++i )
            {
                auto  pi  = M.row_ofs() + i;
                auto  d_i = d( pi );

                if ( pi == pj )
                    DD(i,j) = value_t(0);
                else
                    DD(i,j) = value_t(1) / ( d_i - d_j );
            }// for
        }// for
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        HLR_ASSERT( M.row_is() != M.col_is() );
        
        auto  N = M.copy_struct();
        auto  R = ptrcast( N.get(), matrix::lrmatrix< value_t > );

        // TODO: make efficient!
        auto  D = blas::matrix< value_t >( M.nrows(), M.ncols() );
        
        for ( uint  j = 0; j < D.ncols(); ++j )
        {
            auto  d_j = d( M.col_ofs() + j );
            
            for ( uint  i = 0; i < D.nrows(); ++i )
            {
                auto  d_i = d( M.row_ofs() + i );
                
                D(i,j) = value_t(1) / ( d_i - d_j );
            }// for
        }// for

        auto  [ U, V ] = apx( D, acc );

        R->set_lrmat( std::move( U ), std::move( V ) );

        return  N;
    }// if
    else
        HLR_ERROR( "todo" );
}

}// namespace detail

}}// namespace hlr::seq

#endif // __HLR_SEQ_IPT_HH
