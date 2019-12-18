
#include "hlr/mpi/mpi.hh"
#include "hlr/cluster/distr.hh"
#include "hlr/cluster/tlr.hh"
#include "common.hh"
#include "hlr/utils/RedirectOutput.hh"
#include "hlr/utils/compare.hh"

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    using value_t = typename problem_t::value_t;
    
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();

    auto  tic     = timer::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tlr::cluster( coord.get(), ntile );
    auto  bct     = cluster::tlr::blockcluster( ct.get(), ct.get() );

    // assign blocks to nodes
    if      ( distr == "cyclic2d"    ) cluster::distribution::cyclic_2d( nprocs, bct->root() );
    else if ( distr == "shiftcycrow" ) cluster::distribution::shifted_cyclic_1d( nprocs, bct->root() );
    
    if (( pid == 0 ) && hpro::verbose( 3 ))
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "bct_distr" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( coeff.get() );
    auto  A      = mpi::matrix::build( bct->root(), *pcoeff, *lrapx, fixed_rank( k ) );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "A_%03d", pid ) );
    }// if

    // TLR::MPI::RANK = k;
    
    {
        std::cout << term::bullet << term::bold << "LU ( TLR MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = timer::now();
        
        ARITH::lu< hpro::real >( C.get(), fixed_rank( k ) );
        
        toc = timer::since( tic );
        
        std::cout << "    done in  " << format_time( toc ) << std::endl;

        // compare with otherwise computed result
        compare_ref_file( C.get(), "LU.hm", 1e-8 );
        
        // TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
        // std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
    }
}

int
main ( int argc, char ** argv )
{
    // init MPI before anything else
    mpi::environment   env{ argc, argv };
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    try
    {
        hpro::INIT();

        hlr::cmdline::parse( argc, argv );
    
        // redirect output for all except proc 0
        std::unique_ptr< RedirectOutput >  redir_out = ( ! noredir && (pid != 0)
                                                         ? std::make_unique< RedirectOutput >( to_string( "tlr-mpi_%03d.out", pid ) )
                                                         : nullptr );

        // adjust HLIB network data
        NET::set_nprocs( nprocs );
        NET::set_pid( pid );
    
        std::cout << term::bullet << term::bold << hpro::Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << hpro::Mach::cpuset() << std::endl;
        
        hpro::CFG::set_verbosity( verbosity );

        if ( hlr::nthreads != 0 )
            hpro::CFG::set_nthreads( hlr::nthreads );

        if      ( hlr::appl == "logkernel" ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( hlr::appl == "matern"    ) mymain< hlr::apps::matern_cov >( argc, argv );
        else
            throw "unknown application";

        hpro::DONE();
    }// try
    catch ( char const *   e ) { std::cout << e << std::endl; }
    catch ( hpro::Error &  e ) { std::cout << e.to_string() << std::endl; }

    return 0;
}

// Local Variables:
// mode: c++
// End:
