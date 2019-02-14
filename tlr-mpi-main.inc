#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include "distr.hh"

//
// main function
//
void
mymain ( int argc, char ** argv )
{
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();

    auto  tic        = Time::Wall::now();
    auto  problem    = gen_problem();
    auto  coord      = problem->build_coord( n );
    auto [ ct, bct ] = TLR::cluster( coord.get(), ntile );

    // assign blocks to nodes
    distribution::cyclic_2d( nprocs, bct->root() );
    
    if (( pid == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "tlrmpi_bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "tlrmpi_bct_distr" );
    }// if
    
    auto  A = problem->build_matrix( bct.get(), fixed_rank( k ) );
    
    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "tlrmpi_A_%03d", pid ) );
    }// if

    TLR::MPI::RANK = k;
    
    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( TLR MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TLR::MPI::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        std::cout << "    done in " << toc << std::endl;

        // compare with otherwise computed result
        if ( fs::exists( "LU.hm" ) )
        {
            auto  D  = read_matrix( "LU.hm" );
            auto  BC = ptrcast( C.get(), TBlockMatrix );
            auto  BD = ptrcast( D.get(), TBlockMatrix );

            if (( BC->nrows() == BD->nrows() ) &&
                ( BC->ncols() == BD->ncols() ) &&
                ( BC->nblock_rows() == BD->nblock_rows() ) &&
                ( BC->nblock_cols() == BD->nblock_cols() ))
            {
                bool  correct = true;
                
                D->set_procs( ps_single( pid ), recursive );
        
                for ( uint i = 0; i < BC->nblock_rows(); ++i )
                {
                    for ( uint j = 0; j < BC->nblock_cols(); ++j )
                    {
                        if ( ! is_ghost( BC->block( i, j ) ) )
                        {
                            const auto  f = diff_norm_F( BD->block( i, j ), BC->block( i, j ) );

                            if ( f > 1e-10 )
                            {
                                DBG::printf( "%2d,%2d : %.6e", i, j, diff_norm_F( BD->block( i, j ), BC->block( i, j ) ) );
                                correct = false;
                            }// if
                        }// if
                    }// for
                }// for

                if ( correct )
                    std::cout << "    no error" << std::endl;
            }// if
        }// if
        
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
    unique_ptr< RedirectOutput >  redir_out = ( pid != 0
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

        mymain( argc, argv );

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }

    return 0;
}

// Local Variables:
// mode: c++
// End:
