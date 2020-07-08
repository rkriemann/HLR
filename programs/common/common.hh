
#include <tbb/global_control.h>

#include <iostream>
#include <string>
#include <algorithm>

using std::string;

#include <boost/format.hpp>

using boost::format;

#include <hpro/base/init.hh>
#include <hpro/base/System.hh>
#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/algebra/mat_norm.hh>
#include <hpro/algebra/mat_conv.hh>
#include <hpro/io/TMatrixIO.hh>
#include <hpro/io/TMatrixVis.hh>
#include <hpro/io/TClusterVis.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>

#include <hlr/cluster/tlr.hh>
#include <hlr/cluster/mblr.hh>
#include <hlr/cluster/tileh.hh>
#include <hlr/cluster/h.hh>

namespace hpro = HLIB;
//namespace blas = HLIB::BLAS;

#include "cmdline.hh"
#include "gen_problem.hh"

// using namespace hlr;

//
// default formating
//

// return main memory usage as a string
inline
std::string
mem_usage ()
{
    return hlr::term::yellow( " [" + hpro::Mem::to_string( hpro::Mem::usage() ) + "]" );
}

// return default formated memory string
inline
std::string
format_mem ( const size_t  m )
{
    return hlr::term::black( hpro::Mem::to_string( m ) ) + mem_usage();
}

template < typename ... T_size >
std::string
format_mem ( const size_t    m,
             const T_size... ms )
{
    return hlr::term::black( hpro::Mem::to_string( m ) ) + " / " + format_mem( ms... );
}

// return default formated timing string
std::string
format_time ( const double  t )
{
    return hlr::term::cyan( str( format( "%.3e s" ) % t ) );
}

template < typename duration_t >
std::string
format_time ( const duration_t  t )
{
    return format_time( t.seconds() );
}

template < typename... T >
std::string
format_time ( const double  t, const T... ts )
{
    return hlr::term::cyan( str( format( "%.3e s" ) % t ) ) + " / " + format_time( ts... );
}

// return default formated error string
inline
std::string
format_error ( const double  e )
{
    return hlr::term::red( str( format( "%.4e" ) % e ) );
}

// return default formated norm string
inline
std::string
format_norm ( const double  e )
{
    return hlr::term::italic( str( format( "%.4e" ) % e ) );
}

// return default formated string for FLOPs
inline
std::string
format_flops ( const double  f,
               const double  t )
{
    return str( format( "%.0f / %.2f GFlops" ) % f % ( f / ( 1e9 * t ) ) );
}

//
// timing
//

namespace timer = hpro::Time::Wall;

//
// return min/max/median of elements in container
//

template < typename T > T  min ( const std::vector< T > &  vec ) { return *std::min_element( vec.begin(), vec.end() ); }
template < typename T > T  max ( const std::vector< T > &  vec ) { return *std::max_element( vec.begin(), vec.end() ); }

template < typename T >
T
median ( const std::vector< T > &  vec )
{
    std::vector< T >  tvec( vec );

    std::sort( tvec.begin(), tvec.end() );

    if ( tvec.size() % 2 == 1 )
        return tvec[ tvec.size() / 2 ];
    else
        return T( ( tvec[ tvec.size() / 2 - 1 ] + tvec[ tvec.size() / 2 ] ) / T(2) );
}

//
// generate accuracy
//
inline
hpro::TTruncAcc
gen_accuracy ()
{
    if ( hlr::cmdline::eps < 0 ) return hpro::fixed_rank( hlr::cmdline::k );
    else                         return hpro::fixed_prec( hlr::cmdline::eps );
}

//
// test verbosity
//
inline
bool
verbose ( const int  lvl )
{
    return hpro::verbose( lvl );
}

//
// cluser given coordinate set
//
std::unique_ptr< hpro::TClusterTree >
gen_ct ( hpro::TCoordinate &  coord )
{
    using  hlr::cmdline::cluster;
    
    if (( cluster == "tlr" ) || ( cluster == "blr" ))
    {
        return hlr::cluster::tlr::cluster( coord, hlr::cmdline::ntile );
    }// if
    else if (( cluster == "mblr" ) ||
             (( cluster.size() >= 6 ) && ( cluster.substr( 0, 5 ) == "mblr-" )))
    {
        if ( cluster.size() >= 6 )
            hlr::cmdline::nlvl = std::stoi( cluster.substr( 5, string::npos ) );
            
        return hlr::cluster::mblr::cluster( coord, hlr::cmdline::ntile, hlr::cmdline::nlvl );
    }// if
    else if (( cluster == "tileh" ) ||
             (( cluster.size() >= 7 ) && ( cluster.substr( 0, 6 ) == "tileh-" )))
    {
        if ( cluster.size() >= 7 )
            hlr::cmdline::nlvl = std::stoi( cluster.substr( 6, string::npos ) );
        
        return hlr::cluster::tileh::cluster( coord, hlr::cmdline::ntile, hlr::cmdline::nlvl );
    }// if
    else if (( cluster == "bsp" ) || ( cluster == "h" ))
    {
        return hlr::cluster::h::cluster( coord, hlr::cmdline::ntile );
    }// if
    else
        HLR_ERROR( "unsupported clustering : " + cluster );
}

std::unique_ptr< hpro::TBlockClusterTree >
gen_bct ( hpro::TClusterTree &  rowct,
          hpro::TClusterTree &  colct )
{
    hpro::TBCBuilder  bct_builder;

    if ( hlr::cmdline::adm == "std" )
    {
        hpro::TStdGeomAdmCond  adm_cond( 2.0, hpro::use_min_diam );
        
        return bct_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "weak" )
    {
        hpro::TWeakStdGeomAdmCond  adm_cond;
        
        return bct_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "hodlr" or hlr::cmdline::adm == "offdiag" )
    {
        hpro::TOffDiagAdmCond  adm_cond;
        
        return bct_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "hilo" )
    {
        hpro::THiLoFreqGeomAdmCond  adm_cond( hlr::cmdline::kappa, 10 );
        
        return bct_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else
        HLR_ERROR( "unsupported admissibility : " + hlr::cmdline::adm );
}
