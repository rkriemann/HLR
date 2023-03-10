
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

//
// HPX specific main functions
//
int
hpx_main ( int argc, char ** argv )
{
    hlr_main( argc, argv );
    
    return ::hpx::finalize();
}

int
main ( int argc, char ** argv )
{
    return ::hpx::init( argc, argv );
}
