
#include <hpro/config.h>

#if HPRO_USE_TBB == 1
#include <tbb/global_control.h>
#endif

#include <iostream>
#include <string>
#include <algorithm>

using std::string;

#include <boost/format.hpp>

using boost::format;

#include <hpro/base/init.hh>
#include <hpro/base/System.hh>
#include <hpro/algebra/TLowRankApx.hh>
#include <hpro/io/TMatrixIO.hh>
#include <hpro/io/TClusterVis.hh>
#include <hpro/cluster/TBCBuilder.hh>
#include <hpro/cluster/TGeomAdmCond.hh>
#include <hpro/cluster/TAlgPartStrat.hh>
#include <hpro/cluster/TAlgCTBuilder.hh>
#include <hpro/cluster/TAlgAdmCond.hh>

using Hpro::verbose;

#include <hlr/cluster/tlr.hh>
#include <hlr/cluster/mblr.hh>
#include <hlr/cluster/tileh.hh>
#include <hlr/cluster/h.hh>
#include <hlr/cluster/sfc.hh>

#include <hlr/utils/timer.hh>

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
    return hlr::term::yellow( " [" + Hpro::Mem::to_string( Hpro::Mem::usage() ) + "]" );
}

// return default formated memory string
inline
std::string
format_mem_base ( const size_t  m )
{
    return hlr::term::blue( Hpro::Mem::to_string( m ) );
}

inline
std::string
format_mem ( const size_t  m )
{
    return format_mem_base( m ) + "\t" + mem_usage();
}

template < typename ... T_size >
std::string
format_mem ( const size_t    m,
             const T_size... ms )
{
    return format_mem_base( m ) + " / " + format_mem( ms... );
}

// return default formated timing string
std::string
format_time ( const double  t )
{
    return hlr::term::cyan( str( boost::format( "%.3e s" ) % t ) );
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
    return hlr::term::cyan( str( boost::format( "%.3e s" ) % t ) ) + " / " + format_time( ts... );
}

// return default formated error string
inline
std::string
format_error ( const double  e )
{
    return hlr::term::red( str( boost::format( "%.4e" ) % e ) );
}

template < typename... T >
std::string
format_error ( const double  e, const T... es )
{
    return hlr::term::red( str( boost::format( "%.4e" ) % e ) ) + " / " + format_error( es... );
}

// return default formated norm string
inline
std::string
format_norm ( const double  e )
{
    return hlr::term::italic( str( boost::format( "%.4e" ) % e ) );
}

template < typename... T >
std::string
format_norm ( const double  e, const T... es )
{
    return hlr::term::italic( str( boost::format( "%.4e" ) % e ) ) + " / " + format_norm( es... );
}

// return default formated string for FLOPs
inline
std::string
format_flops ( const double  f )
{
    return str( boost::format( "%.2f MFlop" ) % ( f / 1e6 ) );
}

inline
std::string
format_flops ( const double  f,
               const double  t )
{
    return str( boost::format( "%.2f MFlop/s" ) % ( f / ( 1e6 * t ) ) );
}

inline
std::string
format_rate ( const double  r )
{
    return hlr::term::italic( str( boost::format( "%.02fx" ) % r ) );
}

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
Hpro::TTruncAcc
gen_accuracy ()
{
    if ( hlr::cmdline::eps < 0 ) return Hpro::fixed_rank( hlr::cmdline::k );
    else                         return Hpro::fixed_prec( hlr::cmdline::eps );
}

//
// test verbosity
//
inline
bool
verbose ( const int  lvl )
{
    return Hpro::verbose( lvl );
}

//
// cluser given coordinate set
//
std::unique_ptr< Hpro::TBSPPartStrat >
gen_part_strat ()
{
    using  hlr::cmdline::cluster;

    //
    // choose partitioning strategy
    //

    auto  part = std::unique_ptr< Hpro::TBSPPartStrat >();

    if      ( hlr::cmdline::part == "bsp"      ) part = std::make_unique< Hpro::TCardBSPPartStrat >( Hpro::adaptive_split_axis );
    else if ( hlr::cmdline::part == "bsp-card" ) part = std::make_unique< Hpro::TCardBSPPartStrat >( Hpro::adaptive_split_axis );
    else if ( hlr::cmdline::part == "bsp-vol"  ) part = std::make_unique< Hpro::TGeomBSPPartStrat >( Hpro::adaptive_split_axis );
    else if ( hlr::cmdline::part == "pca"      ) part = std::make_unique< Hpro::TPCABSPPartStrat >( true );
    else if ( hlr::cmdline::part == "pca-card" ) part = std::make_unique< Hpro::TPCABSPPartStrat >( true );
    else if ( hlr::cmdline::part == "pca-vol"  ) part = std::make_unique< Hpro::TPCABSPPartStrat >( false );
    else
        HLR_ERROR( "unsupported partitioning strategy: " + hlr::cmdline::part );

    return part;
}

//
// cluser given coordinate set
//
std::unique_ptr< Hpro::TClusterTree >
gen_ct ( Hpro::TCoordinate &  coord )
{
    using  hlr::cmdline::cluster;

    auto  part = gen_part_strat();

    if (( cluster == "tlr" ) || ( cluster == "blr" ))
    {
        return hlr::cluster::tlr::cluster( coord, *part, hlr::cmdline::ntile );
    }// if
    else if (( cluster == "mblr" ) ||
             (( cluster.size() >= 6 ) && ( cluster.substr( 0, 5 ) == "mblr-" )))
    {
        if ( cluster.size() >= 6 )
            hlr::cmdline::nlvl = std::stoi( cluster.substr( 5, string::npos ) );
            
        return hlr::cluster::mblr::cluster( coord, *part, hlr::cmdline::ntile, hlr::cmdline::nlvl );
    }// if
    else if (( cluster == "tileh" ) ||
             (( cluster.size() >= 7 ) && ( cluster.substr( 0, 6 ) == "tileh-" )))
    {
        if ( cluster.size() >= 7 )
            hlr::cmdline::nlvl = std::stoi( cluster.substr( 6, string::npos ) );
        
        return hlr::cluster::tileh::cluster( coord, *part, hlr::cmdline::ntile, hlr::cmdline::nlvl );
    }// if
    else if (( cluster == "bsp" ) || ( cluster == "h" ))
    {
        return hlr::cluster::h::cluster( coord, *part, hlr::cmdline::ntile );
    }// if
    else if (( cluster == "sfc" ) ||
             (( cluster.size() > 3 ) && ( cluster.substr( 0, 4 ) == "sfc-" )))
    {
        auto  part_type = hlr::cluster::sfc::binary;

        if  ( cluster.size() >= 4 )
        {
            if      ( cluster.substr( 3, cluster.size() ) == "-binary" ) part_type = hlr::cluster::sfc::binary;
            else if ( cluster.substr( 3, cluster.size() ) == "-blr"    ) part_type = hlr::cluster::sfc::blr;
        }// if
        
        return hlr::cluster::sfc::cluster( part_type, coord, hlr::cmdline::ntile );
    }// if
    else
        HLR_ERROR( "unsupported clustering : " + cluster );
}

std::unique_ptr< Hpro::TBlockClusterTree >
gen_bct ( Hpro::TClusterTree &  rowct,
          Hpro::TClusterTree &  colct )
{
    Hpro::TBCBuilder  bt_builder;

    if (( hlr::cmdline::adm == "std" ) || ( hlr::cmdline::adm == "strong" ))
    {
        Hpro::TStdGeomAdmCond  adm_cond( hlr::cmdline::eta, Hpro::use_min_diam );
        
        return bt_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "vertex" )
    {
        Hpro::TWeakStdGeomAdmCond  adm_cond;
        
        return bt_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "weak" )
    {
        Hpro::TWeakGeomAdmCond  adm_cond;
        
        return bt_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if (( hlr::cmdline::adm == "hodlr" ) || ( hlr::cmdline::adm == "offdiag" ))
    {
        Hpro::TOffDiagAdmCond  adm_cond;
        
        return bt_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "hilo" )
    {
        Hpro::THiLoFreqGeomAdmCond  adm_cond( hlr::cmdline::kappa, 10 );
        
        return bt_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else if ( hlr::cmdline::adm == "none" )
    {
        Hpro::TStdGeomAdmCond  adm_cond( 0.0, Hpro::use_min_diam );
        
        return bt_builder.build( & rowct, & colct, & adm_cond );
    }// if
    else
        HLR_ERROR( "unsupported admissibility : " + hlr::cmdline::adm );
}
