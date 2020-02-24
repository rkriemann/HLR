
#include "hlr/gaspi/gaspi.hh"
#include "hlr/gaspi/matrix.hh"
#include "hlr/gaspi/arith.hh"
#include "hlr/cluster/distr.hh"
#include "hlr/cluster/tlr.hh"
#include "common.hh"
#include "hlr/utils/RedirectOutput.hh"
#include "hlr/utils/compare.hh"

#define GPI_CHECK_RESULT( Func, Args )                                  \
    {                                                                   \
        hlr::log( 5, std::string( __ASSERT_FUNCTION ) + " : " + #Func ); \
        auto _check_result = Func Args;                                 \
        assert( _check_result == GASPI_SUCCESS );                       \
    }


using namespace hlr;

//
// main function
//
template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    using value_t = typename problem_t::value_t;
    
    gaspi::process  proc;
    const auto      pid    = proc.rank();
    const auto      nprocs = proc.size();

    auto  tic     = timer::now();
    auto  acc     = gen_accuracy();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = cluster::tlr::cluster( *coord, ntile );
    auto  bct     = cluster::tlr::blockcluster( *ct, *ct );

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
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = gaspi::matrix::build( bct->root(), *pcoeff, *lrapx, acc );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), hpro::to_string( "A_%03d", pid ) );
    }// if

    {
        std::cout << term::bullet << term::bold << "LU ( TLR MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = timer::now();
        
        gaspi::tlr::lu< hpro::real >( C.get(), acc );
        
        toc = timer::since( tic );
        
        std::cout << "    done in " << toc << std::endl;

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
    gaspi::environment  env;
    gaspi::process      proc;
    const auto          pid    = proc.rank();
    const auto          nprocs = proc.size();
    
    cmdline::parse( argc, argv );
    
    // redirect output for all except proc 0
    std::unique_ptr< RedirectOutput >  redir_out = ( ! noredir && (pid != 0)
                                                     ? std::make_unique< RedirectOutput >( hpro::to_string( "tlr-gaspi_%03d.out", pid ) )
                                                     : nullptr );


    if ( false )
    {
        hpro::CFG::set_verbosity( verbosity );
        
        std::cout << '[' << pid << ']' << "  setting up queues" << std::endl;
        
        int                    n = 10;
        std::vector< double >  v( n );
        std::vector< double >  u( n );

        for ( int  i = 0; i < n; ++i )
            u[i] = v[i] = pid;
        
        std::cout << '[' << pid << ']' << "defining segments" << std::endl;
        
        auto  id_v = [] ( const gaspi_rank_t  p ) -> gaspi_segment_id_t { return 2*p; };
        auto  id_u = [] ( const gaspi_rank_t  p ) -> gaspi_segment_id_t { return 2*p+1; };

        std::cout << '[' << pid << ']' << " before ";
        for ( uint i = 0; i < n; ++i )
            std::cout << v[i] << ", ";
        std::cout << std::endl;


        if ( true )
        {
            if (( pid == 0 ) || ( pid == 1 ))
            {
                gaspi::queue    queue;
                gaspi::group    grp( { 0, 1 } );
                gaspi::segment  seg_v( id_v(pid), & v[0], n, grp );
                gaspi::segment  seg_u( id_u(pid), & u[0], n, grp );
                const auto      source = ( pid == 0 ? 1 : 0 );

                queue.read( seg_v, source, id_u(source) );
                queue.wait();
            }// if
        }// if
        else
        {
            std::vector< gaspi::queue >  queues( nprocs );  // one queue to each remote processor

            gaspi::group    world;
            gaspi::segment  seg_v( id_v(pid), & v[0], n, world );
            gaspi::segment  seg_u( id_u(pid), & u[0], n, world );

            const auto                dest  = ( pid + 1 ) % nprocs;
            gaspi::notification_id_t  first = dest;
            
            queues[dest].write_notify( seg_u, dest, id_v(dest), id_v(dest) );
            
            gaspi::notify_wait( seg_v, id_v(pid) );
            
            queues[dest].wait();

            for ( int  p = 0; p < nprocs; ++p )
            {
                if ( p != pid )
                    queues[p].wait();
            }// for

            seg_v.release();
            seg_u.release();
        }// else

        
        std::cout << '[' << pid << ']' << " after  ";
        for ( uint i = 0; i < n; ++i )
            std::cout << v[i] << ", ";
        std::cout << std::endl;
        
        return 0;
    }

    
    try
    {
        hpro::INIT();

        // adjust HLIB network data
        hpro::NET::set_nprocs( nprocs );
        hpro::NET::set_pid( pid );
    
        std::cout << term::bullet << term::bold << hpro::Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << hpro::Mach::cpuset() << std::endl;
        
        hpro::CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            hpro::CFG::set_nthreads( nthreads );

        if      ( appl == "logkernel" ) mymain< hlr::apps::log_kernel >( argc, argv );
        else if ( appl == "matern"    ) mymain< hlr::apps::matern_cov >( argc, argv );
        else
            HLR_ERROR( "unknown application" );

        hpro::DONE();
    }// try
    catch ( char const *   e ) { std::cout << e << std::endl; }
    catch ( hpro::Error &  e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}

// Local Variables:
// mode: c++
// End:
