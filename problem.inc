#ifndef __HLR_PROBLEM_INC
#define __HLR_PROBLEM_INC

#include "problem.hh"
#include "logkernel.hh"
#include "matern.hh"

#include "termcolor.hpp"

namespace term = termcolor;

std::unique_ptr< ProblemBase >
gen_problem ()
{
    std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "Problem Setup" << term::reset << std::endl
              << "    LogKernel, TLR, n = " << n << ", ntile = " << ntile << ", k = " << k << std::endl;
        
    if      ( appl == "logkernel" ) return std::make_unique< LogKernel::Problem >();
    else if ( appl == "matern"    ) return std::make_unique< Matern::Problem >();
    else
        throw "unknown application";
}

#endif // __HLR_PROBLEM_INC

// Local Variables:
// mode: c++
// End:
