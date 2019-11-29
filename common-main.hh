//
// main function specific to arithmetic
//

template < typename problem_t >
void
mymain ( int, char ** );

inline
int
hlrmain ( int argc, char ** argv )
{
    hlr::cmdline::parse( argc, argv );
    
    try
    {
        hpro::INIT();

        std::cout << hlr::term::bullet << hlr::term::bold << hpro::Mach::hostname() << hlr::term::reset << std::endl
                  << "    CPU cores : " << hpro::Mach::cpuset() << std::endl;
        
        hpro::CFG::set_verbosity( hlr::verbosity );

        if ( hlr::nthreads != 0 )
            hpro::CFG::set_nthreads( hlr::nthreads );

        if      ( hlr::appl == "logkernel"  ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( hlr::appl == "materncov"  ) mymain< hlr::apps::matern_cov >( argc, argv );
        else if ( hlr::appl == "laplaceslp" ) mymain< hlr::apps::laplace_slp >( argc, argv );
        else
            throw "unknown application";

        hpro::DONE();
    }// try
    catch ( char const *   e ) { std::cout << e << std::endl; }
    catch ( hpro::Error &  e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}