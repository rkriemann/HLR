#include <hlr/utils/mach.hh>

//
// forward for framework specific main function
//
template < typename problem_t >
void framework_main ();

//
// default main function
//
#define HLR_DEFAULT_MAIN                        \
    int                                         \
    main ( int argc, char ** argv )             \
    {                                           \
        return hlr_main( argc, argv );          \
    } 

//
// actual HLR main function 
//
int
hlr_main ( int argc, char ** argv )
{
    try
    {
        hpro::INIT();

        hlr::cmdline::parse( argc, argv );
    
        std::cout << hlr::term::bullet << hlr::term::bold << hpro::Mach::hostname() << hlr::term::reset << std::endl
                  << "    processor : " << hlr::mach::cpu() << std::endl
                  << "    cores     : " << hlr::mach::cpuset() << std::endl;
        
        hpro::CFG::set_verbosity( hlr::verbosity );

        // if ( hlr::nthreads != 0 )
        //     hpro::CFG::set_nthreads( hlr::nthreads );

        if      ( hlr::appl == "logkernel"    ) framework_main< hlr::apps::log_kernel >();
        else if ( hlr::appl == "materncov"    ) framework_main< hlr::apps::matern_cov >();
        else if ( hlr::appl == "laplaceslp"   ) framework_main< hlr::apps::laplace_slp >();
        else if ( hlr::appl == "helmholtzslp" ) framework_main< hlr::apps::helmholtz_slp >();
        else if ( hlr::appl == "exp"          ) framework_main< hlr::apps::exp >();
        else
            HLR_ERROR( "unknown application (" + hlr::appl + ")" );

        hpro::DONE();
    }// try
    catch ( std::exception &  e ) { std::cout << e.what() << std::endl; }
    catch ( char const *      e ) { std::cout << hlr::term::red( hlr::term::bold( e ) ) << std::endl; }
    
    return 0;
}
