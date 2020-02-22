//
// Project     : HLR
// File        : tile-hodlr.hh
// Description : geeric code for tile-based HODLR LU
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/algebra/mul_vec.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/h.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"
#include "hlr/vector/tiled_scalarvector.hh"
#include "hlr/seq/norm.hh"
#include "hlr/seq/arith_tiled_v2.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    
    std::vector< double >  runtime;
    
    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< hpro::TMatrix >();

    if ( matrixfile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = cluster::h::cluster( *coord, ntile );
        auto  bct     = cluster::h::blockcluster( *ct, *ct );
    
        if ( hpro::verbose( 3 ) )
        {
            hpro::TPSBlockClusterVis   bc_vis;
        
            bc_vis.id( true ).print( bct->root(), "bct" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );

        A = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    }// if
    else
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = hpro::read_matrix( matrixfile );

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// else

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////

    {
        matrix::tile_storage< value_t >  X;
        
        auto  Y = seq::tiled2::multiply( value_t(1), hpro::apply_normal, *A, X );
    }
    
    //////////////////////////////////////////////////////////////////////
    //
    // vector multiplication
    //
    //////////////////////////////////////////////////////////////////////

    if ( true )
    {
        std::cout << term::bullet << term::bold << "Vector Multiplication" << term::reset << std::endl;

        double  t_seq = 0;
        
        {
            hpro::TScalarVector  x( A->col_is(), A->value_type() );
            hpro::TScalarVector  y( A->row_is(), A->value_type() );

            x.fill( 1 );

            runtime.clear();
                
            for ( int  i = 0; i < nbench; ++i )
            {
                tic = timer::now();

                for ( int  j = 0; j < 20; ++j )
                    A->apply_add( 1, & x, & y );
        
                toc = timer::since( tic );

                std::cout << "  gemv in    " << format_time( toc ) << std::endl;

                runtime.push_back( toc.seconds() );
            }// for

            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            t_seq = median( runtime );
        }

        double  t_par = 0;

        {
            hpro::TScalarVector  x( A->col_is(), A->value_type() );
            hpro::TScalarVector  y( A->row_is(), A->value_type() );

            x.fill( 1 );

            runtime.clear();

            for ( int  i = 0; i < nbench; ++i )
            {
                tic = timer::now();

                for ( int  j = 0; j < 20; ++j )
                    impl::mul_vec( hpro::real(1), hpro::apply_normal, *A, x.blas_rvec(), y.blas_rvec() );

                toc = timer::since( tic );

                std::cout << "  gemv in    " << format_time( toc ) << std::endl;

                runtime.push_back( toc.seconds() );
            }// for

            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            t_par = median( runtime );

            std::cout << "  speedup  = " << t_seq / t_par << std::endl;
        }

        // convert to tiled format
        A = impl::matrix::copy_tiled< double >( *A, ntile );

        double  t_tiled = 0;
        
        {
            vector::tiled_scalarvector< hpro::real >  x( A->col_is(), ntile );
            vector::tiled_scalarvector< hpro::real >  y( A->row_is(), ntile );

            x.fill( 1 );
            
            runtime.clear();
                
            for ( int  i = 0; i < nbench; ++i )
            {
                tic = timer::now();

                for ( int  j = 0; j < 20; ++j )
                    impl::tiled2::mul_vec( hpro::real(1), hpro::apply_normal, *A, x, y );
        
                toc = timer::since( tic );

                std::cout << "  gemv in    " << format_time( toc ) << std::endl;

                runtime.push_back( toc.seconds() );
            }// for

            if ( nbench > 1 )
                std::cout << "  runtime  = "
                          << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                          << std::endl;

            t_tiled = median( runtime );

            std::cout << "  speedup  = " << t_seq / t_tiled << std::endl;
        }

        return;
    }// if
}
