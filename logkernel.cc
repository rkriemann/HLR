//
// Project     : HLib
// File        : hodlr-lu.cc
// Description : functions for logarithmic kernel function
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include <cmath>
#include <vector>

using namespace std;

#include <boost/mpi.hpp>

#include "logkernel.hh"

using namespace HLIB;

using real_t = HLIB::real;

namespace LogKernel
{

//
// coefficient function for log|x-y| in [0,1]
//
class TLogCoeffFn : public TCoeffFn< real_t >
{
private:
    // stepwidth
    const double  _h;

public:
    // constructor
    TLogCoeffFn ( const double  h )
            : _h(h)
    {}

    //
    // coefficient evaluation
    //
    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        for ( size_t  j = 0; j < m; ++j )
        {
            const idx_t  idx1 = colidxs[ j ];
            
            for ( size_t  i = 0; i < n; ++i )
            {
                const idx_t  idx0 = rowidxs[ i ];
                double       value;

                if ( idx0 == idx1 ) 
                    value = -1.5*_h*_h + _h*_h*std::log(_h);
                else
                {
                    const double dist = _h * ( std::abs( double( idx0 - idx1 ) ) - 1.0 );
                    const double t1   = dist+1.0*_h;
                    const double t2   = dist+2.0*_h;
            
                    value = ( - 1.5*_h*_h + 0.5*t2*t2*std::log(t2) - t1*t1*std::log(t1) );
            
                    if ( std::abs(dist) > 1e-8 )
                        value += 0.5*dist*dist*std::log(dist);
                }
        
                matrix[ j*n + i ] = real_t(-value);
            }// for
        }// for
    }
    using TCoeffFn< real_t >::eval;

    //
    // return format of matrix, e.g. symmetric or hermitian
    //
    virtual matform_t  matrix_format  () const { return symmetric; }
    
};

//
// set up coordinates
//
std::unique_ptr< TCoordinate >
Problem::build_coord ( const size_t  n )
{
    h = 1.0 / double(n);
    
    vector< double * >  vertices( n, nullptr );
    vector< double * >  bbmin( n, nullptr );
    vector< double * >  bbmax( n, nullptr );

    for ( size_t i = 0; i < n; i++ )
    {
        vertices[i]    = new double;
        vertices[i][0] = h * double(i) + ( h / 2.0 ); // center of [i/h,(i+1)/h]

        // set bounding box (support) to [i/h,(i+1)/h]
        bbmin[i]       = new double;
        bbmin[i][0]    = h * double(i);
        bbmax[i]       = new double;
        bbmax[i][0]    = h * double(i+1);
    }// for

    return make_unique< TCoordinate >( vertices, 1, bbmin, bbmax, copy_coord_data );
}

//
// build matrix
//
std::unique_ptr< TMatrix >
Problem::build_matrix ( const TBlockClusterTree *  bct,
                        const TTruncAcc &          acc )
{
    // unique_ptr< TProgressBar >    progress( ( verbose(2) && my_proc == 0 ) ? new TConsoleProgressBar( cout ) : nullptr );
    TLogCoeffFn                   log_coeff( h );
    TPermCoeffFn< real_t >        coefffn( & log_coeff, bct->row_ct()->perm_i2e(), bct->row_ct()->perm_i2e() );
    TACAPlus< real_t >            aca( & coefffn );
    TDenseMBuilder< real_t >      h_builder( & coefffn, & aca );

    h_builder.set_build_ghosts( true );
    
    return h_builder.build( bct, unsymmetric, acc );
}

// //
// // build matrix using Matern kernel function
// //
// std::unique_ptr< TMatrix >
// logkernel ( const size_t         n,
//             const size_t         ntile,
//             const size_t         k,
//             const std::string &  arith )
// {
//     boost::mpi::communicator  world;
//     const auto                my_proc = world.rank();
//     const auto                nprocs  = world.size();

        
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
//         TBCBuilder           bct_builder( 3 );

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
    
//     if ( arith.find( "-mpi" ) != string::npos )
//     {
//         TBlockCyclicDistrBC  distr;

//         distr.distribute( nprocs, bct->root(), nullptr );
    
//         if (( my_proc == 0 ) && verbose( 3 ))
//         {
//             TPSClusterVis        c_vis;
//             TPSBlockClusterVis   bc_vis;
            
//             c_vis.print( ct->root(), "logkernel_ct" );
//             bc_vis.id( true ).procs( true ).print( bct->root(), "logkernel_bct" );
//         }// if
//     }// if
    
//     //
//     // build matrix
//     //

//     if ( my_proc == 0 )
//         std::cout << "━━ building H-matrix ( k = " << k << " )" << std::endl;

// }

}// namespace LogKernel
