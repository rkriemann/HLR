#ifndef __HLR_GEN_PROBLEM_HH
#define __HLR_GEN_PROBLEM_HH

#include <memory>

#include "hlr/apps/log_kernel.hh"
#include "hlr/apps/matern_cov.hh"
#include "hlr/apps/laplace.hh"
#include "hlr/apps/helmholtz.hh"
#include "hlr/apps/exp.hh"

#include "hlr/utils/term.hh"

namespace hlr
{

using namespace cmdline;

void
print_problem_desc ( const std::string &  name )
{
    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    " << name
              << ( appl == "logkernel" ? hpro::to_string( ", n = %d", n ) : ", grid = " + gridfile )
              << ", ntile = " << ntile
              << ( eps > 0 ? hpro::to_string( ", ε = %.2e", eps ) : hpro::to_string( ", k = %d", k ) )
              << std::endl;
}

template < typename problem_t >
std::unique_ptr< problem_t >
gen_problem ()
{
    HLR_ASSERT( "undefined problem chosen" );
}

template <>
std::unique_ptr< hlr::apps::log_kernel >
gen_problem< hlr::apps::log_kernel > ()
{
    print_problem_desc( "LogKernel" );

    return std::make_unique< hlr::apps::log_kernel >( n );
}

template <>
std::unique_ptr< hlr::apps::matern_cov >
gen_problem< hlr::apps::matern_cov > ()
{
    print_problem_desc( "Matern Covariance" );

    const auto  dashpos = gridfile.find( '-' );

    if ( dashpos != std::string::npos )
    {
        const auto  basename = gridfile.substr( 0, dashpos );
        const auto  size     = gridfile.substr( dashpos+1, gridfile.length() );
        const auto  nsize    = std::atoi( size.c_str() );

        if (( basename == "randsphere" ) || ( basename == "randcube" ))
            return std::make_unique< hlr::apps::matern_cov >( basename, nsize );
        else
            return std::make_unique< hlr::apps::matern_cov >( gridfile );
    }// if
    else
        return std::make_unique< hlr::apps::matern_cov >( gridfile );
}

template <>
std::unique_ptr< hlr::apps::laplace_slp >
gen_problem< hlr::apps::laplace_slp > ()
{
    print_problem_desc( "Laplace SLP" );

    assert( gridfile != "" );
    
    return std::make_unique< hlr::apps::laplace_slp >( gridfile );
}

template <>
std::unique_ptr< hlr::apps::helmholtz_slp >
gen_problem< hlr::apps::helmholtz_slp > ()
{
    print_problem_desc( "Helmholtz SLP" );

    assert( gridfile != "" );
    
    return std::make_unique< hlr::apps::helmholtz_slp >( kappa, gridfile );
}

template <>
std::unique_ptr< hlr::apps::exp >
gen_problem< hlr::apps::exp > ()
{
    print_problem_desc( "Exp" );

    assert( gridfile != "" );
    
    return std::make_unique< hlr::apps::exp >( gridfile );
}

}// namespace hlr

#endif // __HLR_GEN_PROBLEM_HH

// Local Variables:
// mode: c++
// End:
