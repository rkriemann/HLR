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

std::unique_ptr< hlr::apps::log_kernel >
gen_log_kernel ()
{
    print_problem_desc( "LogKernel" );

    return std::make_unique< hlr::apps::log_kernel >( n );
}

std::unique_ptr< hlr::apps::matern_cov >
gen_matern_cov ()
{
    print_problem_desc( "Matern Covariance" );

    if ( gridfile != "" )
        return std::make_unique< hlr::apps::matern_cov >( gridfile );
    else
        return std::make_unique< hlr::apps::matern_cov >( n );
}

std::unique_ptr< hlr::apps::laplace_slp >
gen_laplace_slp ()
{
    print_problem_desc( "Laplace SLP" );

    assert( gridfile != "" );
    
    return std::make_unique< hlr::apps::laplace_slp >( gridfile );
}

std::unique_ptr< hlr::apps::application< hpro::real > >
gen_problem ()
{
    if      ( hlr::appl == "logkernel"  ) return gen_log_kernel();
    else if ( hlr::appl == "materncov"  ) return gen_matern_cov();
    else if ( hlr::appl == "laplaceslp" ) return gen_laplace_slp();
    else
        HLR_ERROR( "unknown application (" + hlr::appl + ")" );
}

}// namespace hlr

#endif // __HLR_GEN_PROBLEM_HH

// Local Variables:
// mode: c++
// End:
