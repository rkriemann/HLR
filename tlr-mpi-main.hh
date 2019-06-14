
#include "hlr/mpi/mpi.hh"
#include "hlr/cluster/distr.hh"
#include "hlr/cluster/tlr.hh"
#include "cmdline.hh"
#include "gen_problem.hh"
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

    auto  tic     = Time::Wall::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tlr::cluster( coord.get(), ntile );
    auto  bct     = cluster::tlr::blockcluster( ct.get(), ct.get() );

    // assign blocks to nodes
    if      ( distr == "cyclic2d"    ) cluster::distribution::cyclic_2d( nprocs, bct->root() );
    else if ( distr == "shiftcycrow" ) cluster::distribution::shifted_cyclic_1d( nprocs, bct->root() );
    
    if (( pid == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "bct_distr" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( coeff.get() );
    auto  A      = mpi::matrix::build( bct->root(), *pcoeff, *lrapx, fixed_rank( k ) );
    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "A_%03d", pid ) );
    }// if

    // TLR::MPI::RANK = k;
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        ARITH::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        std::cout << "    done in " << toc << std::endl;

        // compare with otherwise computed result
        compare_ref_file( C.get(), "LU.hm" );
        
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
    
    parse_cmdline( argc, argv );
    
    // redirect output for all except proc 0
    std::unique_ptr< RedirectOutput >  redir_out = ( ! noredir && (pid != 0)
                                                     ? std::make_unique< RedirectOutput >( to_string( "tlr-mpi_%03d.out", pid ) )
                                                     : nullptr );

    try
    {
        INIT();

        // adjust HLIB network data
        NET::set_nprocs( nprocs );
        NET::set_pid( pid );
    
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << Mach::cpuset() << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        if      ( appl == "logkernel" ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( appl == "matern"    ) mymain< hlr::apps::matern_cov >( argc, argv );
        else
            throw "unknown application";

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }

    return 0;
}

// Local Variables:
// mode: c++
// End:
