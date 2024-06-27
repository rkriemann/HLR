//
// Project     : HLR
// Module      : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hpro/config.h>
#include <hpro/bem/TBFCoeffFn.hh>

#include <hlr/bem/aca.hh>
#include <hlr/approx/randsvd.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

//
// construct grid with given number of 1d grid points
//
std::unique_ptr< Hpro::TGrid >
make_square ( size_t  n )
{
    using triangle_t = Hpro::TGrid::triangle_t;

    const double                  h = 1.0 / (n-1);
    std::vector< Hpro::T3Point >  vertices( n*n );

    for ( size_t  j = 0; j < n; ++j )
        for ( size_t  i = 0; i < n; ++i )
            vertices[j*n+i] = Hpro::T3Point( i*h, j*h, 0 );

    std::vector< triangle_t >  triangles( 2*(n-1)*(n-1) );
    size_t                     pos = 0;
    
    for ( size_t  j = 0; j < n-1; ++j )
    {
        for ( size_t  i = 0; i < n-1; ++i )
        {
            idx_t  v0 = j*n+i;       // ll
            idx_t  v1 = j*n+i+1;     // lr
            idx_t  v2 = (j+1)*n+i+1; // ur
            idx_t  v3 = (j+1)*n+i;   // ul

            triangles[ pos++ ] = triangle_t{ v0, v2, v3 };
            triangles[ pos++ ] = triangle_t{ v0, v1, v2 };
        }// for
    }// for
            
    return std::make_unique< Hpro::TGrid >( vertices, triangles );
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    using real_t  = Hpro::real_type_t< value_t >;

    auto  runtime = std::vector< double >();
    auto  tic     = timer::now();
    auto  toc     = timer::since( tic );

    {
        using  fnspace_t = Hpro::TLinearFnSpace< value_t >;

        auto  n1      = size_t(cmdline::n);
        auto  n2      = n1*n1;
        auto  grid    = make_square( n1 ); // Hpro::make_grid( "square-2" );
        auto  fnspace = fnspace_t( grid.get() );
        auto  bf      = Hpro::TLaplaceSLPBF< fnspace_t, fnspace_t >( & fnspace, & fnspace, 6u );
        auto  coord   = fnspace.build_coord();

        std::cout << "set up " << coord->ncoord() / 16 << " coordinates" << std::endl;
        
        auto  part    = Hpro::TGeomBSPPartStrat( Hpro::adaptive_split_axis );
        auto  ctbuild = Hpro::TBSPCTBuilder( & part, n2 / 16 );

        auto  ct      = ctbuild.build( coord.get() );

        // ┌───┬───┬───┬───┐
        // │   │   │   │   │
        // ├───┼───┼───┼───┤
        // │   │   │   │   │
        // ├───┼───┼───┼───┤
        // │   │   │ 2 │   │
        // ├───┼───┼───┼───┤
        // │   │ 0 │ 1 │ 3 │
        // └───┴───┴───┴───┘
        
        const Hpro::TGeomCluster *  cl0 = nullptr;
        const Hpro::TGeomCluster *  cl1 = nullptr;
        const Hpro::TGeomCluster *  cl2 = nullptr;
        const Hpro::TGeomCluster *  cl3 = nullptr;

        // collect sons on level 3
        auto  cls = std::list< const Hpro::TGeomCluster * >();
        
        for ( uint  i0 = 0; i0 < 2; ++i0 )
            for ( uint  i1 = 0; i1 < 2; ++i1 )
                for ( uint  i2 = 0; i2 < 2; ++i2 )
                    for ( uint  i3 = 0; i3 < 2; ++i3 )
                        cls.push_back( cptrcast( ct->root()->son(i0)->son(i1)->son(i2)->son(i3), Hpro::TGeomCluster ) );

        // look for cl0..3
        auto  p0 = Hpro::TPoint( 0.375, 0.125, 0 );
        auto  p1 = Hpro::TPoint( 0.625, 0.125, 0 );
        auto  p2 = Hpro::TPoint( 0.625, 0.375, 0 );
        auto  p3 = Hpro::TPoint( 0.875, 0.125, 0 );
        
        for ( auto  cl : cls ) if ( cl->bbox().is_inside( p0 ) ) { cl0 = cl; break; }
        for ( auto  cl : cls ) if ( cl->bbox().is_inside( p1 ) ) { cl1 = cl; break; }
        for ( auto  cl : cls ) if ( cl->bbox().is_inside( p2 ) ) { cl2 = cl; break; }
        for ( auto  cl : cls ) if ( cl->bbox().is_inside( p3 ) ) { cl3 = cl; break; }
        
        {
            io::vtk::print( *grid, "grid" );
            io::vtk::print( *coord, "coord" );

            // {
            //     auto  label = std::vector< uint >( coord->ncoord(), 0 );

            //     for ( uint  i = ct->root()->son(0)->son(0)->first(); i <= ct->root()->son(0)->son(0)->last(); ++i )
            //         label[ct->perm_i2e()->permute(i)] = 1;

            //     for ( uint  i = ct->root()->son(1)->son(0)->first(); i <= ct->root()->son(1)->son(0)->last(); ++i )
            //         label[ct->perm_i2e()->permute(i)] = 2;
        
            //     io::vtk::print( *coord, label, "coord1" );
            // }

            // {
            //     auto  label = std::vector< uint >( coord->ncoord(), 0 );

            //     for ( uint  i = ct->root()->son(0)->son(0)->first(); i <= ct->root()->son(0)->son(0)->last(); ++i )
            //         label[ct->perm_i2e()->permute(i)] = 1;

            //     for ( uint  i = ct->root()->son(1)->son(1)->first(); i <= ct->root()->son(1)->son(1)->last(); ++i )
            //         label[ct->perm_i2e()->permute(i)] = 2;
        
            //     io::vtk::print( *coord, label, "coord2" );
            // }

            {
                auto  label = std::vector< uint >( coord->ncoord(), 0 );

                for ( uint  i = cl0->first(); i <= cl0->last(); ++i ) label[ct->perm_i2e()->permute(i)] = 1;
                for ( uint  i = cl1->first(); i <= cl1->last(); ++i ) label[ct->perm_i2e()->permute(i)] = 2;
        
                io::vtk::print( *coord, label, "coord1" );
            }

            {
                auto  label = std::vector< uint >( coord->ncoord(), 0 );

                for ( uint  i = cl0->first(); i <= cl0->last(); ++i ) label[ct->perm_i2e()->permute(i)] = 1;
                for ( uint  i = cl2->first(); i <= cl2->last(); ++i ) label[ct->perm_i2e()->permute(i)] = 2;
        
                io::vtk::print( *coord, label, "coord2" );
            }

            {
                auto  label = std::vector< uint >( coord->ncoord(), 0 );

                for ( uint  i = cl0->first(); i <= cl0->last(); ++i ) label[ct->perm_i2e()->permute(i)] = 1;
                for ( uint  i = cl3->first(); i <= cl3->last(); ++i ) label[ct->perm_i2e()->permute(i)] = 2;
        
                io::vtk::print( *coord, label, "coord3" );
            }
        }// if

        //
        // build matrix blocks
        //

        std::cout << "setting up matrices ... " << std::flush;
        
        tic = timer::now();
        
        auto  coeff  = Hpro::TBFCoeffFn< decltype( bf ) >( & bf ); 
        auto  pcoeff = hpro::TPermCoeffFn< value_t >( &coeff, ct->perm_i2e(), ct->perm_i2e() );
        auto  A0     = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  A1     = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  A2     = std::unique_ptr< Hpro::TMatrix< value_t > >();
        auto  A3     = std::unique_ptr< Hpro::TMatrix< value_t > >();
            
        ::tbb::parallel_invoke(
            [&] () { A0 = pcoeff.build( *cl0, *cl0 ); std::cout << "0, " << std::flush; },
            [&] () { A1 = pcoeff.build( *cl0, *cl1 ); std::cout << "1, " << std::flush; },
            [&] () { A2 = pcoeff.build( *cl0, *cl2 ); std::cout << "2, " << std::flush; },
            [&] () { A3 = pcoeff.build( *cl0, *cl3 ); std::cout << "3, " << std::flush; }
        );
        
        toc = timer::since( tic );
    
        std::cout << "done in " << format_time( toc ) << std::endl;
        
        auto  B0 = cptrcast( A0.get(), Hpro::TDenseMatrix< value_t > )->blas_mat();
        auto  B1 = cptrcast( A1.get(), Hpro::TDenseMatrix< value_t > )->blas_mat();
        auto  B2 = cptrcast( A2.get(), Hpro::TDenseMatrix< value_t > )->blas_mat();
        auto  B3 = cptrcast( A3.get(), Hpro::TDenseMatrix< value_t > )->blas_mat();

        std::cout << "computing singular values ... " << std::flush;

        io::matlab::write( B2, "B2" );
        
        tic = timer::now();
        
        auto  S0 = blas::vector< real_t >();
        auto  S1 = blas::vector< real_t >();
        auto  S2 = blas::vector< real_t >();
        auto  S3 = blas::vector< real_t >();
        auto  S4 = blas::vector< real_t >();
    
        tic = timer::now();

        ::tbb::parallel_invoke(
            [&] () { S0 = std::move( blas::sv( B0 ) ); std::cout << "0, " << std::flush; },
            [&] () { S1 = std::move( blas::sv( B1 ) ); std::cout << "1, " << std::flush; },
            [&] () { S2 = std::move( blas::sv( B2 ) ); std::cout << "2, " << std::flush; },
            [&] () { S3 = std::move( blas::sv( B3 ) ); std::cout << "3, " << std::flush; }
        );

        toc = timer::since( tic );
    
        std::cout << "done in " << format_time( toc ) << std::endl;

        auto  acc = fixed_prec( 1e-6 );

        std::cout << "ranks : " << acc.trunc_rank( S1 ) << " / " << acc.trunc_rank( S2 ) << " / " << acc.trunc_rank( S3 ) << std::endl;
        
        io::matlab::write( S0, "laplace_S0" );
        io::matlab::write( S1, "laplace_S1" );
        io::matlab::write( S2, "laplace_S2" );
        io::matlab::write( S3, "laplace_S3" );
    }
    
    // auto  problem = gen_problem< problem_t >();
    // auto  coord   = problem->coordinates();
    // auto  ct      = gen_ct( *coord );
    // auto  bct     = gen_bct( *ct, *ct );
    
    // if ( hpro::verbose( 3 ) )
    // {
    //     io::vtk::print( *coord, "coord" );

    //     auto  label = std::vector< uint >( coord->ncoord() );

    //     for ( uint  son = 0; son < 2; ++son )
    //         for ( uint  i = ct->root()->son(son)->first(); i <= ct->root()->son(son)->last(); ++i )
    //             label[ct->perm_i2e()->permute(i)] = son;
        
    //     io::vtk::print( *coord, label, "coord1" );

    //     for ( uint  son = 0; son < 4; ++son )
    //         for ( uint  i = ct->root()->son(son/2)->son(son%2)->first(); i <= ct->root()->son(son/2)->son(son%2)->last(); ++i )
    //             label[ct->perm_i2e()->permute(i)] = son;
        
    //     io::vtk::print( *coord, label, "coord2" );
    // }// if
    
    // auto  coeff  = problem->coeff_func();
    // auto  pcoeff = hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    // auto  lrapx  = bem::aca_lrapx( pcoeff );

    // tic = timer::now();

    // auto  A00 = impl::matrix::build_dense( bct->root()->son(0,0), pcoeff );
    // auto  A01 = impl::matrix::build_dense( bct->root()->son(0,1), pcoeff );
    // auto  A10 = impl::matrix::build_dense( bct->root()->son(1,0), pcoeff );
    
    // toc = timer::since( tic );
    
    // std::cout << "    done in  " << format_time( toc ) << std::endl;
    // std::cout << "    dims   = " << term::bold << A->nrows() << " × " << A->ncols() << term::reset << std::endl;
    // std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    // if ( hpro::verbose( 3 ) )
    // {
    //     io::eps::print( *A, "A", "noid" );
    // }// if
}
