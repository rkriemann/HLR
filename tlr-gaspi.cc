
#include "gaspi/gaspi.hh"
#include "gaspi/matrix.hh"
#include "gaspi/arith.hh"
#include "mpi/distr.hh"
#include "cluster/tlr.hh"
#include "cmdline.hh"
#include "gen_problem.hh"
#include "utils/RedirectOutput.hh"
#include "utils/compare.hh"

#define GPI_CHECK_RESULT( Func, Args )                                  \
    {                                                                   \
        HLR::log( 5, std::string( __ASSERT_FUNCTION ) + " : " + #Func ); \
        auto _check_result = Func Args;                                 \
        assert( _check_result == GASPI_SUCCESS );                       \
    }


using namespace HLR;

//
// main function
//
template < typename problem_t >
void
mymain ( int argc, char ** argv )
{
    using value_t = typename problem_t::value_t;
    
    GASPI::process  proc;
    const auto      pid    = proc.rank();
    const auto      nprocs = proc.size();

    auto  tic     = Time::Wall::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = TLR::cluster( coord.get(), ntile );
    auto  bct     = TLR::blockcluster( ct.get(), ct.get() );

    // assign blocks to nodes
    if      ( distr == "cyclic2d"    ) distribution::cyclic_2d( nprocs, bct->root() );
    else if ( distr == "shiftcycrow" ) distribution::shifted_cyclic_1d( nprocs, bct->root() );
    
    if (( pid == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "bct_distr" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = Matrix::GASPI::build( bct->root(), *pcoeff, *lrapx, fixed_rank( k ) );
    auto  toc    = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "A_%03d", pid ) );
    }// if

    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        GASPI::TLR::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
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
    GASPI::environment  env;
    GASPI::process      proc;
    const auto          pid    = proc.rank();
    const auto          nprocs = proc.size();
    
    parse_cmdline( argc, argv );
    
    // redirect output for all except proc 0
    std::unique_ptr< RedirectOutput >  redir_out = ( ! noredir && (pid != 0)
                                                     ? std::make_unique< RedirectOutput >( to_string( "tlr-gaspi_%03d.out", pid ) )
                                                     : nullptr );


    if ( false )
    {
        CFG::set_verbosity( verbosity );
        
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
                GASPI::queue    queue;
                GASPI::group    grp( { 0, 1 } );
                GASPI::segment  seg_v( id_v(pid), & v[0], n, grp );
                GASPI::segment  seg_u( id_u(pid), & u[0], n, grp );
                const auto      source = ( pid == 0 ? 1 : 0 );

                queue.read( seg_v, source, id_u(source) );
                queue.wait();
            }// if
        }// if
        else
        {
            std::vector< GASPI::queue >  queues( nprocs );  // one queue to each remote processor

            GASPI::group    world;
            GASPI::segment  seg_v( id_v(pid), & v[0], n, world );
            GASPI::segment  seg_u( id_u(pid), & u[0], n, world );

            const auto                dest  = ( pid + 1 ) % nprocs;
            GASPI::notification_id_t  first = dest;
            
            queues[dest].write_notify( seg_u, dest, id_v(dest), id_v(dest) );
            
            GASPI::notify_wait( seg_v, id_v(pid) );
            
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
        INIT();

        // adjust HLIB network data
        NET::set_nprocs( nprocs );
        NET::set_pid( pid );
    
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << Mach::hostname() << term::reset << std::endl
                  << "    CPU cores : " << Mach::cpuset() << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        if      ( appl == "logkernel" ) mymain< HLR::Apps::LogKernel >( argc, argv );
        else if ( appl == "matern"    ) mymain< HLR::Apps::MaternCov >( argc, argv );
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
