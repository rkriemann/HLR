// #include <tbb/global_control.h>

template < typename problem_t >
void
framework_main ()
{
    // auto                   param = ::tbb::global_control::max_allowed_parallelism;
    // ::tbb::global_control  tbb_control( param, ( hlr::cmdline::nthreads > 0 ? hlr::cmdline::nthreads : ::tbb::global_control::active_value( param ) ) );
    
    program_main< problem_t >();
}

HLR_DEFAULT_MAIN
