#ifndef __HLR_GEN_PROBLEM_HH
#define __HLR_GEN_PROBLEM_HH

#include <memory>

#include "hlr/apps/log_kernel.hh"
#include "hlr/apps/matern_cov.hh"
#include "hlr/apps/laplace.hh"

#include "hlr/utils/term.hh"

namespace hlr
{

using namespace cmdline;

template < typename problem_t >
std::unique_ptr< problem_t >
gen_problem ()
{
    assert( false );
}

void
print_problem_desc ( const std::string &  name )
{
    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    " << name
              << ( gridfile == "" ? hpro::to_string( ", n = %d", n ) : ", grid = " + gridfile )
              << ", ntile = " << ntile
              << ( eps > 0 ? hpro::to_string( ", ε = %.2e", eps ) : hpro::to_string( ", k = %d", k ) )
              << std::endl;
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
gen_problem ()
{
    print_problem_desc( "Matern Covariance" );

    if ( gridfile != "" )
        return std::make_unique< hlr::apps::matern_cov >( gridfile );
    else
        return std::make_unique< hlr::apps::matern_cov >( n );
}

template <>
std::unique_ptr< hlr::apps::laplace_slp >
gen_problem ()
{
    print_problem_desc( "Laplace SLP" );

    assert( gridfile != "" );
    
    return std::make_unique< hlr::apps::laplace_slp >( gridfile );
}

}// namespace hlr

#endif // __HLR_GEN_PROBLEM_HH

// Local Variables:
// mode: c++
// End:
