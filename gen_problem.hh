#ifndef __HLR_GEN_PROBLEM_HH
#define __HLR_GEN_PROBLEM_HH

#include <memory>

#include "apps/logkernel.hh"
#include "apps/matern.hh"
#include "apps/Laplace.hh"

#include "utils/termcolor.hpp"

namespace term = termcolor;

template < typename problem_t >
std::unique_ptr< problem_t >
gen_problem ()
{
    assert( false );
}

void
print_problem_desc ( const std::string &  name )
{
    std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
              << "    " << name
              << ", n = " << n
              << ", ntile = " << ntile
              << ( eps > 0 ? HLIB::to_string( ", ε = %.2e", eps ) : HLIB::to_string( ", k = %d", k ) )
              << std::endl;
}

template <>
std::unique_ptr< HLR::Apps::LogKernel >
gen_problem< HLR::Apps::LogKernel > ()
{
    print_problem_desc( "LogKernel" );
    return std::make_unique< HLR::Apps::LogKernel >( n );
}

template <>
std::unique_ptr< HLR::Apps::MaternCov >
gen_problem ()
{
    print_problem_desc( "Matern Covariance" );
    return std::make_unique< HLR::Apps::MaternCov >( n );
}

template <>
std::unique_ptr< HLR::Apps::LaplaceSLP >
gen_problem ()
{
    print_problem_desc( "Laplace SLP" );
    return std::make_unique< HLR::Apps::LaplaceSLP >( n, grid );
}

#endif // __HLR_GEN_PROBLEM_HH

// Local Variables:
// mode: c++
// End:
