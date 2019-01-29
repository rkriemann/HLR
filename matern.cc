//
// Project     : HLib
// File        : matern.cc
// Description : use matern kernel to fill matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cmath>
#include <vector>
#include <random>

using namespace std;

#include "matern.hh"

using namespace HLIB;

using real_t = HLIB::real;

namespace Matern
{

using  point_t = T3Point;

//
// set up coordinates
//
std::unique_ptr< TCoordinate >
Problem::build_coord ( const size_t  n )
{
    // std::random_device                rd{};
    // std::mt19937                      generator{ rd() };
    std::mt19937_64                   generator{ 1 };
    std::uniform_real_distribution<>  distr{ 0, 1 };

    vertices.resize( n );
    
    for ( size_t i = 0; i < n; i++ )
        vertices[i] = spherical( 2 * M_PI * distr( generator ),
                                 2 * M_PI * distr( generator ) - M_PI,
                                 1.0 ); // point_t( distr( generator ), distr( generator ) );
    
    auto  coord = make_unique< TCoordinate >( vertices );
    
    if ( verbose( 3 ) )
        print_vtk( coord.get(), "matern_coord" );

    return coord;
}
//
// build matrix
//
std::unique_ptr< TMatrix >
Problem::build_matrix ( const TBlockClusterTree *  bct,
                        const TTruncAcc &          acc )
{
    // unique_ptr< TProgressBar >    progress( verbose(2) ? new TConsoleProgressBar( std::cout ) : nullptr );
    TMaternCovCoeffFn< point_t >  matern_coeff( 1.0, 1.29, 0.325, vertices );
    TPermCoeffFn< real_t >        coefffn( & matern_coeff, bct->row_ct()->perm_i2e(), bct->row_ct()->perm_i2e() );
    TACAPlus< real_t >            aca( & coefffn );
    TDenseMBuilder< real_t >      h_builder( & coefffn, & aca );
    TPSMatrixVis                  mvis;
    
    return h_builder.build( bct, unsymmetric, acc );
}

// //
// // build matrix using Matern kernel function
// //
// std::unique_ptr< TMatrix >
// matern ( const size_t         n,
//          const size_t         ntile,
//          const size_t         k,
//          const std::string &  arith )
// {
    
//     //
//     // build cluster tree and block cluster tree
//     //
    
//     unique_ptr< TClusterTree >       ct;
//     unique_ptr< TBlockClusterTree >  bct;

//     if ( arith == "std" )
//     {
//         TCardBSPPartStrat    part_strat;
//         TBSPCTBuilder        ct_builder( & part_strat, ntile );

//         ct = ct_builder.build( coord.get() );

//         TWeakStdGeomAdmCond  adm_cond( 4.0 );
//         TBCBuilder           bct_builder;

//         bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
//     }// if
//     else if ( arith.substr( 0, 6 ) == "hodlr-" )
//     {
//         TCardBSPPartStrat    part_strat;
//         TBSPCTBuilder        ct_builder( & part_strat, ntile );

//         ct = ct_builder.build( coord.get() );

//         TOffDiagAdmCond      adm_cond;
//         TBCBuilder           bct_builder;

//         bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
//     }// if
//     else if ( arith.substr( 0, 4 ) == "tlr-" )
//     {
//         TCardBSPPartStrat    part_strat;
//         TMBLRCTBuilder       ct_builder( 1, & part_strat, ntile );

//         ct = ct_builder.build( coord.get() );

//         TWeakStdGeomAdmCond  adm_cond( 4.0 );
//         TBCBuilder           bct_builder;

//         bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );
//     }// if
//     else
//         throw "unknown arithmetic";
    
//     if ( verbose( 3 ) )
//     {
//         TPSClusterVis        c_vis;
//         TPSBlockClusterVis   bc_vis;
        
//         c_vis.print( ct->root(), "matern_ct" );
//         bc_vis.id( true ).print( bct->root(), "matern_bct" );
//     }// if
    
//     //
//     // build matrix
//     //
    
//     std::cout << "━━ building H-matrix ( k = " << k << " )" << std::endl;
    
// }

}// namespace Matern
