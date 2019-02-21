//
// Project     : HLib
// File        : th-mpi-bcast.cc
// Description : Tile-H arithmetic with MPI broadcast
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>
#include <fstream>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

using namespace std;

#include "mpi.hh"

#include "cmdline.inc"
#include "problem.inc"
#include "tlr-mpi.inc"
#include "tensor.hh"
#include "RedirectOutput.hh"
#include "distr.hh"
#include "tools.hh"
#include "tileh.hh"

namespace TileH
{

namespace MPI
{

double  time_mpi = 0;

void
broadcast ( mpi::communicator &  comm,
            TByteStream &        bs,
            const int            root_proc )
{
    // first exchange size information
    size_t  size = bs.size();

    comm.broadcast( size, root_proc );

    log( 5, to_string( "bs size = %d", size ) );
    
    if ( bs.size() != size )
        bs.set_size( size );

    // then the actual data
    comm.broadcast( bs.data(), size, root_proc );
}

size_t  max_add_mem = 0;

template < typename value_t >
void
lu ( TBlockMatrix *     A,
     const TTruncAcc &  acc )
{
    mpi::communicator  world;
    const auto         pid    = world.rank();
    const auto         nprocs = world.size();
    
    log( 4, to_string( "lu( %d )", A->id() ) );

    auto  nbr = A->nblock_rows();
    auto  nbc = A->nblock_cols();

    //
    // set up communicators for rows/cols
    //

    vector< mpi::communicator >          row_comms( nbr ), col_comms( nbc );  // communicators for rows/columns
    vector< list< int > >                row_procs( nbr ), col_procs( nbc );  // set of processors for rows/columns
    vector< unordered_map< int, int > >  row_maps( nbr ),  col_maps( nbc );   // mapping of global ranks to row/column ranks

    TLR::MPI::build_row_comms( A, row_comms, row_procs, row_maps );
    TLR::MPI::build_col_comms( A, col_comms, col_procs, col_maps );

    //
    // LU factorization
    //

    for ( uint  i = 0; i < nbr; ++i )
    {
        // counts additional memory per step due to non-local data
        size_t  add_mem = 0;
        
        log( 4, "────────────────────────────────────────────────" );
        log( 4, to_string( "step %d", i ) );
        
        auto  A_ii = A->block( i, i );
        auto  p_ii = A_ii->procs().master();

        assert( A_ii->procs().size() == 1 );
        
        if ( pid == p_ii )
        {
            log( 4, to_string( "lu( %d )", A_ii->id() ) );
            HLIB::LU::factorise_rec( A_ii, acc );
        }// if

        if ( contains( col_procs[i], pid ) || contains( row_procs[i], pid ) )
        {
            //
            // broadcast diagonal block
            //

            auto  tic = Time::Wall::now();
            
            unique_ptr< TMatrix >  T_ii;        // temporary storage with auto-delete
            TMatrix *              H_ii = A_ii; // handle for A_ii/T_ii
            TByteStream            bs;

            if ( pid == p_ii )
            {
                log( 4, to_string( "serialization of %d ", A_ii->id() ) );
                
                bs.set_size( A_ii->bs_size() );
                A_ii->write( bs );
            }// if

            // broadcast serialized data
            if ( contains( col_procs[i], pid ) )
            {
                log( 4, to_string( "broadcast %d from %d to ", A_ii->id(), p_ii ) + to_string( col_procs[i] ) );
                broadcast( col_comms[i], bs, col_maps[i][p_ii] );
            }// if

            if (( col_procs[i] != row_procs[i] ) && contains( row_procs[i], pid ))
            {
                log( 4, to_string( "broadcast %d from %d to ", A_ii->id(), p_ii ) + to_string( row_procs[i] ) );
                broadcast( row_comms[i], bs, row_maps[i][p_ii] );
            }// if
            
            // and reconstruct matrix
            if ( pid != p_ii )
            {
                log( 4, to_string( "construction of %d ", A_ii->id() ) );
                
                TBSHMBuilder  bs_hbuild;

                T_ii = bs_hbuild.build( bs );
                T_ii->set_procs( ps_single( pid ) );
                H_ii = T_ii.get();

                add_mem += H_ii->bs_size();
            }// if
            
            auto  toc = Time::Wall::since( tic );

            time_mpi += toc.seconds();
            
            //
            // off-diagonal solves
            //
            
            for ( uint  j = i+1; j < nbr; ++j )
            {
                // L is unit diagonal !!! Only solve with U
                auto        A_ji = A->block( j, i );
                const auto  p_ji = A_ji->procs().master();

                assert( A_ji->procs().size() == 1 );
                
                if ( pid == p_ji )
                {
                    log( 4, to_string( "solve_U( %d, %d )", H_ii->id(), A->block( j, i )->id() ) );
                    solve_upper_right( A_ji, H_ii, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
                }// if
            }// for

            for ( uint  l = i+1; l < nbc; ++l )
            {
                // L is unit diagonal !!! Only solve with U
                auto        A_il = A->block( i, l );
                const auto  p_il = A_il->procs().master();
                
                assert( A_il->procs().size() == 1 );
                
                if ( pid == p_il )
                {
                    log( 4, to_string( "solve_L( %d, %d )", H_ii->id(), A->block( i, l )->id() ) );
                    solve_lower_left( apply_normal, H_ii, nullptr, A_il, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
                }// if
            }// for
        }
        
        //
        // broadcast blocks in row/column for update phase
        //

        vector< unique_ptr< TMatrix > >  row_i_mat( nbr );      // for autodeletion
        vector< unique_ptr< TMatrix > >  col_i_mat( nbc );
        vector< TMatrix * >              row_i( nbr, nullptr ); // matrix handles
        vector< TMatrix * >              col_i( nbc, nullptr );
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = A->block( j, i );
            const auto  p_ji = A_ji->procs().master();
            
            // broadcast A_ji to all processors in row j
            if ( contains( row_procs[j], pid ) )
            {
                TByteStream  bs;
                
                if ( pid == p_ji )
                {
                    log( 4, to_string( "serialisation of %d ", A_ji->id() ) );
                    
                    bs.set_size( A_ji->bs_size() );
                    A_ji->write( bs );
                    row_i[j] = A_ji;
                }// if
                
                log( 4, to_string( "broadcast %d from %d to ", A_ji->id(), p_ji ) + to_string( row_procs[j] ) );
                
                broadcast( row_comms[j], bs, row_maps[j][p_ji] );

                if ( pid != p_ji )
                {
                    log( 4, to_string( "construction of %d ", A_ji->id() ) );
                    
                    TBSHMBuilder  bs_hbuild;

                    row_i_mat[j] = bs_hbuild.build( bs );
                    row_i_mat[j]->set_procs( ps_single( pid ) );
                    row_i[j]     = row_i_mat[j].get();
                    add_mem     += row_i[j]->byte_size();
                }// if
            }// if
        }
        
        for ( uint  l = i+1; l < nbc; ++l )
        {
            const auto  A_il = A->block( i, l );
            const auto  p_il = A_il->procs().master();
            
            // broadcast A_il to all processors in column l
            if ( contains( col_procs[l], pid ) )
            {
                TByteStream  bs;
                
                if ( pid == p_il )
                {
                    log( 4, to_string( "serialisation of %d ", A_il->id() ) );
                    
                    bs.set_size( A_il->bs_size() );
                    A_il->write( bs );
                    col_i[l] = A_il;
                }// if
                
                log( 4, to_string( "broadcast %d from %d to ", A_il->id(), p_il ) + to_string( col_procs[l] ) );
                
                broadcast( col_comms[l], bs, col_maps[l][p_il] );

                if ( pid != p_il )
                {
                    log( 4, to_string( "construction of %d ", A_il->id() ) );
                    
                    TBSHMBuilder  bs_hbuild;

                    col_i_mat[l] = bs_hbuild.build( bs );
                    col_i_mat[l]->set_procs( ps_single( pid ) );
                    col_i[l]     = col_i_mat[l].get();
                    add_mem     += col_i[l]->byte_size();
                }// if
            }// if
        }// for

        max_add_mem = std::max( max_add_mem, add_mem );
        
        //
        // update of trailing sub-matrix
        //
        
        for ( uint  j = i+1; j < nbr; ++j )
        {
            const auto  A_ji = row_i[j];
            
            for ( uint  l = i+1; l < nbc; ++l )
            {
                const auto  A_il = col_i[l];
                
                //
                // update local matrix block
                //
                
                auto        A_jl = A->block( j, l );
                const auto  p_jl = A_jl->procs().master();
                
                if ( pid == p_jl )
                {
                    log( 4, to_string( "update of %d with %d × %d", A_jl->id(), A_ji->id(), A_il->id() ) );
                    
                    multiply( -1.0, A_ji, A_il, 1.0, A_jl, acc );
                }// if
            }// for
        }// for
    }// for

    std::cout << "  time in MPI : " << format( "%.2fs" ) % time_mpi << std::endl;
    std::cout << "  add memory  : " << Mem::to_string( max_add_mem ) << std::endl;
}

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    assert( is_blocked( A ) );

    lu< value_t >( ptrcast( A, TBlockMatrix ), acc );
}

}// namespace MPI

}// namespace TileH

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
    auto [ ct, bct ] = TileH::cluster( coord.get(), ntile, nprocs );

    // assign blocks to nodes
    if      ( distr == "cyclic2d"    ) distribution::cyclic_2d( nprocs, bct->root() );
    else if ( distr == "shiftcycrow" ) distribution::shifted_cyclic_1d( nprocs, bct->root() );
    
    if (( pid == 0 ) && verbose( 3 ))
    {
        TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).procs( false ).print( bct->root(), "thmpi_bct" );
        bc_vis.id( false ).procs( true ).print( bct->root(), "thmpi_bct_distr" );
    }// if
    
    auto  A = problem->build_matrix( bct.get(), fixed_rank( k ) );
    
    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    
    if ( verbose( 3 ) )
    {
        TPSMatrixVis  mvis;
    
        mvis.svd( false ).id( true ).print( A.get(), to_string( "thmpi_A_%03d", pid ) );
    }// if

    {
        std::cout << term::yellow << term::bold << "∙ " << term::reset << term::bold << "LU ( Tile-H MPI )" << term::reset << std::endl;
        
        auto  C = A->copy();
        
        tic = Time::Wall::now();
        
        TileH::MPI::lu< HLIB::real >( C.get(), fixed_rank( k ) );
        
        toc = Time::Wall::since( tic );
        
        std::cout << "    done in " << toc << std::endl;

    //     // compare with otherwise computed result
    //     if ( fs::exists( "LU.hm" ) )
    //     {
    //         auto  D  = read_matrix( "LU.hm" );
    //         auto  BC = ptrcast( C.get(), TBlockMatrix );
    //         auto  BD = ptrcast( D.get(), TBlockMatrix );

    //         if (( BC->nrows() == BD->nrows() ) &&
    //             ( BC->ncols() == BD->ncols() ) &&
    //             ( BC->nblock_rows() == BD->nblock_rows() ) &&
    //             ( BC->nblock_cols() == BD->nblock_cols() ))
    //         {
    //             bool  correct = true;
                
    //             D->set_procs( ps_single( pid ), recursive );
        
    //             for ( uint i = 0; i < BC->nblock_rows(); ++i )
    //             {
    //                 for ( uint j = 0; j < BC->nblock_cols(); ++j )
    //                 {
    //                     if ( ! is_ghost( BC->block( i, j ) ) )
    //                     {
    //                         const auto  f = diff_norm_F( BD->block( i, j ), BC->block( i, j ) );

    //                         if ( f > 1e-10 )
    //                         {
    //                             DBG::printf( "%2d,%2d : %.6e", i, j, diff_norm_F( BD->block( i, j ), BC->block( i, j ) ) );
    //                             correct = false;
    //                         }// if
    //                     }// if
    //                 }// for
    //             }// for

    //             if ( correct )
    //                 std::cout << "    no error" << std::endl;
    //         }// if
    //     }// if
        
    //     // TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
        
    //     // std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
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
                                                ? make_unique< RedirectOutput >( to_string( "thmpi_%03d.out", pid ) )
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
