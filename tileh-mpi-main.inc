#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include "mpi/mpi.hh"
#include "mpi/distr.hh"
#include "cluster/tileh.hh"
#include "cmdline.hh"
#include "gen_problem.hh"
#include "utils/RedirectOutput.hh"

using namespace HLR;

void
compare_ref ( TMatrix *  A )
{
    mpi::communicator  world;
    const auto         pid    = world.rank();

    if ( fs::exists( "LU.hm" ) )
    {
        auto  D  = read_matrix( "LU.hm" );
        auto  BA = ptrcast( A,       TBlockMatrix );
        auto  BD = ptrcast( D.get(), TBlockMatrix );

        if (( BA->nrows() == BD->nrows() ) &&
            ( BA->ncols() == BD->ncols() ) &&
            ( BA->nblock_rows() == BD->nblock_rows() ) &&
            ( BA->nblock_cols() == BD->nblock_cols() ))
        {
            bool  correct = true;
                
            D->set_procs( ps_single( pid ), recursive );
        
            for ( uint i = 0; i < BA->nblock_rows(); ++i )
            {
                for ( uint j = 0; j < BA->nblock_cols(); ++j )
                {
                    if ( ! is_ghost( BA->block( i, j ) ) )
                    {
                        const auto  f = diff_norm_F( BD->block( i, j ), BA->block( i, j ) );

                        if ( f > 1e-10 )
                        {
                            DBG::printf( "%2d,%2d : %.6e", i, j, diff_norm_F( BD->block( i, j ), BA->block( i, j ) ) );
                            correct = false;
                        }// if
                    }// if
                }// for
            }// for

            if ( correct )
                std::cout << "    no error" << std::endl;
        }// if
    }// if
}

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
    auto  ct      = TileH::cluster( coord.get(), ntile, std::max< uint >( 3, std::log2( nprocs )+2 ) );
    auto  bct     = TileH::blockcluster( ct.get(), ct.get() );

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
    auto  lrapx  = std::make_unique< TACAPlus< value_t > >( coeff.get() );
    auto  A      = Matrix::MPI::build( bct->root(), *pcoeff, *lrapx, fixed_rank( k ) );
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
        compare_ref( C.get() );
        
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
    
    // redirect output for all except proc 0
    std::unique_ptr< RedirectOutput >  redir_out = ( pid != 0
                                                     ? make_unique< RedirectOutput >( to_string( "tlrmpi_%03d.out", pid ) )
                                                     : nullptr );

    parse_cmdline( argc, argv );
    
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
