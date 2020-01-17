#include <hlr/utils/mach.hh>

//
// forward for framework specific main function
//
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

        framework_main();

        hpro::DONE();
    }// try
    catch ( char const *   e ) { std::cout << e << std::endl; }
    catch ( hpro::Error &  e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}
