//
// Project     : HLR
// Module      : magma.cc
// Description : program for testing MAGMA based arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/io/TClusterBasisVis.hh>

#include <hlr/seq/norm.hh>
#include <hlr/seq/arith_uniform.hh>
#include <hlr/matrix/print.hh>

#include "hlr/tf/matrix.hh"
#include "hlr/tf/approx.hh"
#include "hlr/tf/dag.hh"

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< hpro::TMatrix >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis  bc_vis;
        
        print_ps( ct->root(), "ct" );
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    
    A = hlr::tf::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // assign clusters since needed for cluster bases
    hlr::tf::matrix::assign_cluster( *A, *bct->root() );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    //////////////////////////////////////////////////////////////////
    //
    // compute QR
    //
    //////////////////////////////////////////////////////////////////

    blas::magma::init();
    
    // assuming HODLR adm., so use first off-diagonal block as low-rank matrix
    auto  A01 = ptrcast( ptrcast( A.get(), hpro::TBlockMatrix )->block( 0, 1 ), hpro::TRkMatrix );
    auto  U   = blas::copy( hpro::blas_mat_A< value_t >( *A01 ) );
    // auto  V   = blas::copy( hpro::blas_mat_B< value_t >( *A01 ) );
        
    //
    // using default blas
    //

    {
        auto                     U1 = blas::copy( U );
        blas::matrix< value_t >  R1( U.ncols(), U.ncols() );
        
        blas::qr( U1, R1 );
        
        hpro::DBG::write( U1, "U1.mat", "U1" );
        hpro::DBG::write( R1, "R1.mat", "R1" );
    }
    
    //
    // using MAGMA
    //

    {
        auto                     U2 = blas::copy( U );
        blas::matrix< value_t >  R2( U.ncols(), U.ncols() );
    
        blas::magma::init();
        blas::magma::qr( U2, R2 );

        hpro::DBG::write( U2, "U2.mat", "U2" );
        hpro::DBG::write( R2, "R2.mat", "R2" );
    }

    //////////////////////////////////////////////////////////////////
    //
    // compute QR with column pivoting
    //
    //////////////////////////////////////////////////////////////////

    //
    // using default blas
    //

    {
        auto                     U1 = blas::copy( U );
        blas::matrix< value_t >  R1( U.ncols(), U.ncols() );
        std::vector< int >       P1( U.ncols() );
        
        blas::qrp( U1, R1, P1 );
        
        hpro::DBG::write( U1, "U1p.mat", "U1p" );
        hpro::DBG::write( R1, "R1p.mat", "R1p" );
    }
    
    //
    // using MAGMA
    //

    {
        auto                     U2 = blas::copy( U );
        blas::matrix< value_t >  R2( U.ncols(), U.ncols() );
        std::vector< int >       P2( U.ncols() );
    
        blas::magma::qrp( U2, R2, P2 );
        
        hpro::DBG::write( U2, "U2p.mat", "U2p" );
        hpro::DBG::write( R2, "R2p.mat", "R2p" );
    }
}

template < typename problem_t >
void
framework_main ()
{
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );

    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
