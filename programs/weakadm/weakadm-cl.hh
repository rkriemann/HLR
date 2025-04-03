//
// Project     : HLR
// Module      : weakadm
// Description : program for testing weak admissibility
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <tbb/parallel_invoke.h>

#include <hlr/matrix/radial.hh>
#include <hlr/apps/radial.hh>
#include <hlr/cluster/weakadm.hh>
#include <hlr/cluster/build.hh>

#include "common.hh"
#include "common-main.hh"

using namespace hlr;

std::vector< Hpro::TPoint >
make_vertices_nd ( const uint    dim,
                   const size_t  nsize )
{
    using  point_t = Hpro::TPoint;
    
    auto          vertices = std::vector< Hpro::TPoint >();
    const double  h        = 1.0 / double(nsize-1);
    
    if ( dim == 1 )
    {
        vertices.reserve( nsize );
            
        for ( size_t  i0 = 0; i0 < nsize; ++i0 )
            vertices.push_back( Hpro::TPoint( i0*h ) );
    }// if
    else if ( dim == 2 )
    {
        vertices.reserve( nsize * nsize );
            
        for ( size_t  i1 = 0; i1 < nsize; ++i1 )
            for ( size_t  i0 = 0; i0 < nsize; ++i0 )
                vertices.push_back( Hpro::TPoint( i0*h, i1*h ) );
    }// if
    else if ( dim == 3 )
    {
        vertices.reserve( nsize * nsize * nsize );
            
        for ( size_t  i2 = 0; i2 < nsize; ++i2 )
            for ( size_t  i1 = 0; i1 < nsize; ++i1 )
                for ( size_t  i0 = 0; i0 < nsize; ++i0 )
                    vertices.push_back( Hpro::TPoint( i0*h, i1*h, i2*h ) );
    }// if
    else if ( dim == 4 )
    {
        vertices.reserve( nsize * nsize * nsize * nsize );
            
        for ( size_t  i3 = 0; i3 < nsize; ++i3 )
            for ( size_t  i2 = 0; i2 < nsize; ++i2 )
                for ( size_t  i1 = 0; i1 < nsize; ++i1 )
                    for ( size_t  i0 = 0; i0 < nsize; ++i0 )
                        vertices.push_back( point_t( i0*h, i1*h, i2*h, i3*h ) );
    }// if
    else if ( dim == 5 )
    {
        vertices.reserve( nsize * nsize * nsize * nsize * nsize );
            
        for ( size_t  i4 = 0; i4 < nsize; ++i4 )
            for ( size_t  i3 = 0; i3 < nsize; ++i3 )
                for ( size_t  i2 = 0; i2 < nsize; ++i2 )
                    for ( size_t  i1 = 0; i1 < nsize; ++i1 )
                        for ( size_t  i0 = 0; i0 < nsize; ++i0 )
                            vertices.push_back( point_t( i0*h, i1*h, i2*h, i3*h, i4*h ) );
    }// if
    else
        HLR_ERROR( "unsupported dimension" );

    return vertices;
}

//
// determine number of blocks with face/edge/vertex/strong admissibility
//
size_t
nadmblocks ( const uint              anoverlap,
             const cluster::block &  bc )
{
    if ( bc.is_adm() )
    {
        auto  rowcl = cptrcast( bc.rowcl(), Hpro::TGeomCluster );
        auto  colcl = cptrcast( bc.colcl(), Hpro::TGeomCluster );

        if ( rowcl == colcl )
            return 0;

        const uint  dim      = rowcl->bbox().min().dim();
        uint        noverlap = 0;
    
        const auto  rbbox = rowcl->bbox();
        const auto  cbbox = colcl->bbox();
        
        for ( uint  i = 0; i < dim; ++i )
        {
            const auto  rmin   = rbbox.min()[i];
            const auto  rmax   = rbbox.max()[i];
            
            const auto  cmin   = cbbox.min()[i];
            const auto  cmax   = cbbox.max()[i];
            
            if (( rmax <= cmin ) ||   // ├── τ ──┤├── σ ──┤
                ( cmax <= rmin ))     // ├── σ ──┤├── τ ──┤
            {
                // no overlap
            }// if
            else
                noverlap++;
        }// for

        // filter out strong adm.
        if ( std::min( rbbox.diameter(), cbbox.diameter() ) <= ( 2.0 * rbbox.distance( cbbox ) ) )
        {
            if ( anoverlap == dim )// signals strong adm. is requested
                return 1;
            else
                return 0;
        }//
        else if ( noverlap == anoverlap ) // weak admissibility
            return 1;
        else
            return 0;
    }// if
    else
    {
        size_t  n = 0;
        
        for ( uint  i = 0; i < bc.nsons(); ++i )
            n += nadmblocks( anoverlap, * bc.son(i) );

        return n;
    }// else
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

    //
    // ND coordinates
    //

    auto  vertices = make_vertices_nd( cmdline::ndim, cmdline::n );
    auto  coord    = cluster::coordinates( vertices );
    
    if ( hpro::verbose( 4 ) )
        io::vtk::print( coord, "coord" );
    
    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    " << kernel
              << Hpro::to_string( ", n = %d/%d", cmdline::n, coord.ncoord() )
              << ", ntile = " << ntile
              << ( eps > 0 ? Hpro::to_string( ", ε = %.2e", eps ) : Hpro::to_string( ", k = %d", k ) )
              << std::endl;
    
    auto  part         = Hpro::TGeomBSPPartStrat( Hpro::adaptive_split_axis );
    auto  [ ct, pe2i ] = cluster::build_cluster_tree( coord, part, cmdline::ntile );

    auto  strong  = cluster::strong_adm();
    auto  bcts    = cluster::build_block_tree( *ct, *ct, strong );

    if ( hpro::verbose( 3 ) )
        io::eps::print( *bcts, "bcts" );
        
    std::cout << term::bold << "  adm. blocks:" << term::reset << std::endl
              << "    adm  │ ";
    for ( uint  a = 0; a < cmdline::ndim; ++a )
        std::cout << Hpro::to_string( "   adm%d │", a );
    std::cout << "   std  │ c_sp " << std::endl;
    
    std::cout << "   ──────┼─";
    for ( uint  a = 0; a <= cmdline::ndim; ++a )
        std::cout << "────────┼";
    std::cout << "──────" << std::endl;
    
    std::cout << "    std  │ ";
    for ( uint  a = 0; a <= cmdline::ndim; ++a )
        std::cout << boost::format( "%7d │" ) % nadmblocks( a, *bcts );
    std::cout << boost::format( " %3d" ) % Hpro::compute_c_sp( *bcts, true, true ) << std::endl;
    
    for ( uint  a = 0; a < cmdline::ndim; ++a )
    {
        auto  adm = cluster::weak_adm( a );
        auto  bct = cluster::build_block_tree( *ct, *ct, adm );

        if ( hpro::verbose( 3 ) )
            io::eps::print( *bct, Hpro::to_string( "bct%d", a ) );

        std::cout << boost::format( "    adm%d │ " ) % a;
        for ( uint  a = 0; a <= cmdline::ndim; ++a )
            std::cout << boost::format( "%7d │" ) % nadmblocks( a, *bct );
        std::cout << boost::format( " %3d" ) % Hpro::compute_c_sp( *bct, true, true ) << std::endl;
    }// if

    // tic = timer::now();

    // auto  logr      = matrix::log_function< value_t >();
    // auto  newton    = matrix::newton_function< value_t >();
    // auto  exp       = matrix::exponential_function< value_t >( value_t(1) );
    // auto  gaussian  = matrix::gaussian_function< value_t >( value_t(1) );
    // auto  mquadric  = matrix::multiquadric_function< value_t >( value_t(1) );
    // auto  imquadric = matrix::inverse_multiquadric_function< value_t >( value_t(1) );
    // auto  tps       = matrix::thin_plate_spline_function< value_t >( value_t(1) );
    // auto  ratquad   = matrix::rational_quadratic_function< value_t >( value_t(1), value_t(1) );
    // auto  matcov    = matrix::matern_covariance_function< value_t >( value_t(1), value_t(1.0/3.0), value_t(1) );

    // if ( cmdline::kernel == "log" )
    // {
    //     auto  kernel = radial_kernel( logr );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "newton" )
    // {
    //     auto  kernel = radial_kernel( newton );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "exp" )
    // {
    //     auto  kernel = radial_kernel( exp );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "gaussian" )
    // {
    //     auto  kernel = radial_kernel( gaussian );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "mquadric" )
    // {
    //     auto  kernel = radial_kernel( mquadric );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "imquadric" )
    // {
    //     auto  kernel = radial_kernel( imquadric );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "tps" )
    // {
    //     auto  kernel = radial_kernel( tps );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "ratquad" )
    // {
    //     auto  kernel = radial_kernel( ratquad );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else if ( cmdline::kernel == "matcov" )
    // {
    //     auto  kernel = radial_kernel( matcov );
        
    //     std::tie( M0, M1, M2, M3, M4 ) = build_blocks( kernel, n3, C0, C1, C2, C3, C4 );
    // }// if
    // else
    //     HLR_ERROR( "unsupported radial function : " + cmdline::kernel );
    
    // toc = timer::since( tic );
    
    // std::cout << "done in " << format_time( toc ) << std::endl;

    // //
    // // computing singular values
    // //

    // std::cout << "computing singular values ... " << std::flush;

    // auto  S0 = blas::vector< real_t >();
    // auto  S1 = blas::vector< real_t >();
    // auto  S2 = blas::vector< real_t >();
    // auto  S3 = blas::vector< real_t >();
    // auto  S4 = blas::vector< real_t >();
    
    // tic = timer::now();

    // ::tbb::parallel_invoke(
    //     [&] () { S0 = std::move( blas::sv( M0 ) ); std::cout << "0, " << std::flush; },
    //     [&] () { S1 = std::move( blas::sv( M1 ) ); std::cout << "1, " << std::flush; },
    //     [&] () { S2 = std::move( blas::sv( M2 ) ); std::cout << "2, " << std::flush; },
    //     [&] () { S3 = std::move( blas::sv( M3 ) ); std::cout << "3, " << std::flush; },
    //     [&] () { S4 = std::move( blas::sv( M4 ) ); std::cout << "4, " << std::flush; }
    // );
    
    // toc = timer::since( tic );
    
    // std::cout << "done in " << format_time( toc ) << std::endl;

    // auto  acc = fixed_prec( cmdline::eps );

    // std::cout << "ranks : " << acc.trunc_rank( S1 ) << " / " << acc.trunc_rank( S2 ) << " / " << acc.trunc_rank( S3 ) << " / " << acc.trunc_rank( S4 ) << std::endl;
    
    // io::matlab::write( S0, cmdline::kernel + "_S0" );
    // io::matlab::write( S1, cmdline::kernel + "_S1" );
    // io::matlab::write( S2, cmdline::kernel + "_S2" );
    // io::matlab::write( S3, cmdline::kernel + "_S3" );
    // io::matlab::write( S4, cmdline::kernel + "_S4" );
}
