//
// Project     : HLib
// File        : matern.cc
// Description : use matern kernel to fill matrix
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include <iostream>
#include <string>

using namespace std;

#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

using boost::format;
using namespace boost::program_options;

#include <hlib.hh>
#include "parallel/TDistrBC.hh"

using namespace HLIB;

using real_t = HLIB::real;

#include "logkernel.hh"
#include "matern.hh"

#include "tlr.hh"
#include "hodlr.hh"

//
// print block-by-block comparison of A and B
//
void
compare_blocks ( TMatrix *  A,
                 TMatrix *  B )
{
    if ( is_blocked( A ) && is_blocked( B ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );
        auto  BB = ptrcast( B, TBlockMatrix );

        for ( uint  i = 0; i < BA->block_rows(); ++i )
            for ( uint  j = 0; j < BA->block_cols(); ++j )
                compare_blocks( BA->block( i, j ), BB->block( i, j ) );
    }// if
    else if ( A->type() == B->type() )
    {
        cout << A->typestr() << "( " << A->id() << " ) : " <<  diff_norm_F( A, B ) << endl;
    }// if
    else
        HERROR( ERR_CONSISTENCY, "compare_blocks", "different block structure" );
}

//
// main function
//
int
main ( int argc, char ** argv )
{
    // init MPI before anything else
    boost::mpi::environment   env{ argc, argv };
    boost::mpi::communicator  world;
    const auto                my_proc = world.rank();
    const auto                nprocs  = world.size();
    
    size_t  n         = 512;
    size_t  ntile     = 64;
    size_t  k         = 5;
    string  appl      = "logkernel";
    string  arith     = "std";
    uint    nthreads  = 0;
    uint    verbosity = 1;
    bool    nostd     = false;
    
    //
    // define command line options
    //

    options_description  opts( "usage: hlrtest [options]\n  where options include" );
    variables_map        vm;

    // standard options
    opts.add_options()
        ( "arith",       value<string>(), ": type of arithmetic (std,tlr-seq,tlr-tbb,hodlr-seq,hodlr-tbb)" )
        ( "help,h",                       ": print this help text" )
        ( "ntile",       value<int>(),    ": set tile size" )
        ( "nprob,n",     value<int>(),    ": set problem size" )
        ( "nodag",                        ": do not use DAG in arithmetic" )
        ( "nostd",                        ": do not use standard arithmetic" )
        ( "app",         value<string>(), ": application type (logkernel,matern)" )
        ( "rank,k",      value<uint>(),   ": set H-algebra rank k" )
        ( "threads,t",   value<int>(),    ": number of parallel threads" )
        ( "verbosity,v", value<int>(),    ": verbosity level" )
        ;

    //
    // parse command line options
    //

    try
    {
        store( command_line_parser( argc, argv ).options( opts ).run(), vm );
        notify( vm );
    }// try
    catch ( required_option &  e )
    {
        if ( my_proc == 0 )
            cout << e.get_option_name() << " requires an argument, try \"-h\"" << endl;
        exit( 1 );
    }// catch
    catch ( unknown_option &  e )
    {
        if ( my_proc == 0 )
            cout << e.what() << ", try \"-h\"" << endl;
        exit( 1 );
    }// catch

    //
    // eval command line options
    //

    if ( vm.count( "help") )
    {
        if ( my_proc == 0 )
            cout << opts << endl;
        exit( 1 );
    }// if

    if ( vm.count( "threads"   ) ) nthreads  = vm["threads"].as<int>();
    if ( vm.count( "verbosity" ) ) verbosity = vm["verbosity"].as<int>();
    if ( vm.count( "nodag"     ) ) CFG::Arith::use_dag   = false;
    if ( vm.count( "nostd"     ) ) nostd     = true;
    if ( vm.count( "nprob"     ) ) n         = vm["nprob"].as<int>();
    if ( vm.count( "ntile"     ) ) ntile     = vm["ntile"].as<int>();
    if ( vm.count( "rank"      ) ) k         = vm["rank"].as<uint>();
    if ( vm.count( "app"       ) ) appl      = vm["app"].as<string>();
    if ( vm.count( "arith"     ) ) arith     = vm["arith"].as<string>();

    try
    {
        if ( ! ( arith == "std"       ||
                 arith == "hodlr-seq" || arith == "hodlr-tbb" ||
                 arith == "tlr-seq"   || arith == "tlr-tbb"   || "tlr-mpi" ))
            throw "unknown arithmetic";
         
        //
        // init HLIBpro
        //
        
        INIT();

        if ( my_proc == 0 )
            std::cout << "━━ " << Mach::hostname() << std::endl
                      << "    CPU cores : " << Mach::cpuset()
                      << std::endl;
        
        CFG::set_verbosity( verbosity );

        if ( nthreads != 0 )
            CFG::set_nthreads( nthreads );

        std::cout << "━━ Problem Setup ( " << appl << " )" << std::endl;
        
        unique_ptr< ProblemBase >  problem;
        
        if      ( appl == "logkernel" ) problem = make_unique< LogKernel::Problem >();
        else if ( appl == "matern"    ) problem = make_unique< Matern::Problem >();
        
        auto  tic = Time::Wall::now();

        auto coord = problem->build_coord( n );
        
        unique_ptr< TClusterTree >       ct;
        unique_ptr< TBlockClusterTree >  bct;
        
        if      ( arith.substr( 0, 6 ) == "hodlr-" ) std::tie( ct, bct ) = HODLR::cluster( coord.get(), ntile );
        else if ( arith.substr( 0, 4 ) == "tlr-"   ) std::tie( ct, bct ) = TLR::cluster( coord.get(), ntile );

        if ( arith.find( "-mpi" ) != string::npos )
        {
            TBlockCyclicDistrBC  distr;
            
            distr.distribute( nprocs, bct->root(), nullptr );
        }// if
        
        if (( my_proc == 0 ) && verbose( 3 ))
        {
            TPSBlockClusterVis   bc_vis;
            
            bc_vis.id( true ).procs( true ).print( bct->root(), "bct" );
        }// if
        
        auto  A = problem->build_matrix( bct.get(), fixed_rank( k ) );
        
        auto  toc = Time::Wall::since( tic );
        
        if ( my_proc == 0 )
        {
            std::cout << "    done in " << format( "%.2fs" ) % toc.seconds() << std::endl;
            std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
        }// if

        TPSMatrixVis  mvis;
        
        if (( my_proc == 0 ) && verbose( 3 ))
            mvis.svd( false ).id( true ).print( A.get(), "hlrtest_A" );

        if ( arith == "std" )
        {
            if ( my_proc == 0 )
                std::cout << "━━ LU facorisation ( Std )" << std::endl;
            
            auto  C = A->copy();
            
            tic = Time::Wall::now();
            
            lu( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );
            
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            if ( my_proc == 0 )
            {
                std::cout << "    done in " << toc << std::endl;
                std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
            }// if
        }// if

        if ( arith == "tlr-seq" )
        {
            if ( my_proc == 0 )
                std::cout << "━━ LU facorisation ( TLR Seq )" << std::endl;
            
            auto  C = A->copy();
            
            tic = Time::Wall::now();
            
            TLR::SEQ::lu< HLIB::real >( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );
            
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            if ( my_proc == 0 )
            {
                std::cout << "    done in " << toc << std::endl;
                std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
            }// if
        }// if

        if ( arith == "tlr-tbb" )
        {
            if ( my_proc == 0 )
                std::cout << "━━ LU facorisation ( TLR TBB )" << std::endl;
            
            auto  C = A->copy();
            
            tic = Time::Wall::now();
            
            TLR::TBB::lu< HLIB::real >( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );
            
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            if ( my_proc == 0 )
            {
                std::cout << "    done in " << toc << std::endl;
                std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
            }// if
        }// if

        if ( arith == "tlr-mpi" )
        {
            if ( my_proc == 0 )
                std::cout << "━━ LU facorisation ( TLR MPI )" << std::endl;
            
            auto  C = A->copy();
            
            tic = Time::Wall::now();
            
            TLR::MPI::lu< HLIB::real >( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );
            
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            if ( my_proc == 0 )
            {
                std::cout << "    done in " << toc << std::endl;
                // std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
            }// if
        }// if

        if ( arith == "hodlr-seq" )
        {
            if ( my_proc == 0 )
                std::cout << "━━ LU facorisation ( HODLR Seq )" << std::endl;
            
            auto  C = A->copy();
            
            tic = Time::Wall::now();
            
            HODLR::SEQ::lu< HLIB::real >( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );
            
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            if ( my_proc == 0 )
            {
                std::cout << "    done in " << toc << std::endl;
                std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
            }// if
        }// if

        if ( arith == "hodlr-tbb" )
        {
            if ( my_proc == 0 )
                std::cout << "━━ LU facorisation ( HODLR TBB )" << std::endl;
            
            auto  C = A->copy();
            
            tic = Time::Wall::now();
            
            HODLR::TBB::lu< HLIB::real >( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );
            
            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );
            
            if ( my_proc == 0 )
            {
                std::cout << "    done in " << toc << std::endl;
                std::cout << "    inversion error  = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
            }// if
        }// if

        DONE();
    }// try
    catch ( char const *  e ) { std::cout << e << std::endl; }
    catch ( Error &       e ) { std::cout << e.to_string() << std::endl; }
    
    return 0;
}
