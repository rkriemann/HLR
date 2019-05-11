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

template <>
std::unique_ptr< HLR::Apps::LogKernel >
gen_problem< HLR::Apps::LogKernel > ()
{
    std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
              << "    LogKernel, n = " << n << ", ntile = " << ntile << ", k = " << k << std::endl;
        
    return std::make_unique< HLR::Apps::LogKernel >( n );
}

template <>
std::unique_ptr< HLR::Apps::MaternCov >
gen_problem ()
{
    std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
              << "    Matern Covariance, n = " << n << ", ntile = " << ntile << ", k = " << k << std::endl;
        
    return std::make_unique< HLR::Apps::MaternCov >( n );
}

template <>
std::unique_ptr< HLR::Apps::LaplaceSLP >
gen_problem ()
{
    std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
              << "    Laplace SLP, grid = " << grid << ", n = " << n << ", ntile = " << ntile << ", k = " << k << std::endl;
        
    return std::make_unique< HLR::Apps::LaplaceSLP >( n, grid );
}

#endif // __HLR_GEN_PROBLEM_HH

// Local Variables:
// mode: c++
// End:
