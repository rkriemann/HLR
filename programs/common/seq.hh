#include <hpro/config.h>

#if USE_TBB == 1
#include <tbb/global_control.h>
#endif

template < typename problem_t >
void
framework_main ()
{
    #if USE_TBB == 1
    // limit HLIBpro parallelism
    ::tbb::global_control  tbb_control( ::tbb::global_control::max_allowed_parallelism, 1 );
    #endif
    
    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
