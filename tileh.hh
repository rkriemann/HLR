//
// Project     : HLR
// File        : tileh-seq.cc
// Description : sequential Tile-H arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrixSum.hh>
#include <hpro/matrix/TMatrixProduct.hh>

#include "common.hh"
#include "common-main.hh"
#include "hlr/cluster/tileh.hh"
#include "hlr/cluster/mblr.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/norm.hh"
#include "hlr/bem/aca.hh"

using namespace hlr;

uint64_t
get_flops ( const std::string &  method )
{
    #if HLIB_COUNT_FLOPS == 1

    return blas::FLOPS;

    #else

    if ( ntile == 128 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 455151893464;   // 515345354964;
            if ( gridfile == "sphere-6" ) return 2749530544148;  // 3622694502712;
            if ( gridfile == "sphere-7" ) return 12122134505132; // 21122045509696;
            if ( gridfile == "sphere-8" ) return 118075035109436;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 124087920212;  // 122140965488;
            if ( gridfile == "sphere-6" ) return 881254402164;  // 832636379560;
            if ( gridfile == "sphere-7" ) return 5442869949704; // 5113133279628;
            if ( gridfile == "sphere-8" ) return 30466486574184;
        }// if
    }// if
    else if ( ntile == 64 )
    {
        if ( method == "mm" )
        {
            if ( gridfile == "sphere-5" ) return 362295459228;  // 362301558484;
            if ( gridfile == "sphere-6" ) return 2254979752712; // 2364851019180;
            if ( gridfile == "sphere-7" ) return 9888495763740; // 10305554560228;
            if ( gridfile == "sphere-8" ) return 119869484219652;
        }// if
        else if ( method == "lu" )
        {
            if ( gridfile == "sphere-5" ) return 111349327848; // 111663294708;
            if ( gridfile == "sphere-6" ) return 912967909892; // 936010549040;
            if ( gridfile == "sphere-7" ) return 6025437614656; // 6205509061236;
            if ( gridfile == "sphere-8" ) return 33396933144996;
        }// if
    }// if

    #endif

    return 0;
}

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;

    auto  tic     = timer::now();
    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();

    HLR_ASSERT( std::log2( coord->ncoord() ) - std::log2( ntile ) >= nlvl );

    auto  ct      = cluster::tileh::cluster( coord.get(), ntile, nlvl );
    auto  bct     = cluster::tileh::blockcluster( ct.get(), ct.get() );

    std::cout << "    tiling = " << ct->root()->nsons() << " × " << ct->root()->nsons() << std::endl;
    
    if ( verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis   bc_vis;
        
        bc_vis.id( true ).print( bct->root(), "bct" );
    }// if

    auto  acc    = gen_accuracy();
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< bem::aca_lrapx< hpro::TPermCoeffFn< value_t > > >( *pcoeff );
    // auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    auto  A      = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );
    auto  toc    = timer::since( tic );
    
    std::cout << "    done in " << format_time( toc ) << std::endl;
    std::cout << "    mem   = " << format_mem( A->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    {
        auto  D = hpro::to_dense( A.get() );
        auto  fout = fopen( "A.bin", "wb" );

        fwrite( hpro::blas_mat< value_t >( D.get() ).data(), sizeof(value_t), A->nrows()*A->ncols(), fout );

        std::cout << hpro::blas_mat< value_t >( D.get() )( 256, 123 ) << std::endl;
        
        // hpro::DBG::write( A.get(), "A.mat", "A" );
    }
    
    //////////////////////////////////////////////////////////////////////
    //
    // matrix multiplication
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "Matrix Multiplication ( Tile-H " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;
    
    auto  mul_res = hpro::matrix_product( A.get(), A.get() );
   
    if ( false ) 
    {
        std::vector< double >  runtime, flops;
        
        std::cout << "  " << term::bullet << " DAG" << std::endl;
        
        auto  C0 = impl::matrix::copy( *A );

        tic = timer::now();
        
        auto  dag = std::move( dag::gen_dag_update( A.get(), A.get(), C0.get(), nseq, impl::dag::refine ) );
            
        toc = timer::since( tic );
            
        std::cout << "    DAG in   " << format_time( toc ) << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
        
        if ( verbose( 3 ) )
            dag.print_dot( "mul.dot" );
        
        for ( int i = 0; i < nbench; ++i )
        {
            blas::FLOPS = 0;
            blas::reset_statistics();
            
            tic = timer::now();
            
            impl::dag::run( dag, acc );
            
            toc = timer::since( tic );
            std::cout << "    mult in  " << format_time( toc ) << std::endl;
            
            flops.push_back( get_flops( "mm" ) );
            runtime.push_back( toc.seconds() );
        }// for

        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;
            
        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        auto  diff = hpro::matrix_sum( 1.0, mul_res.get(), -1.0, C0.get() );

        std::cout << "    error  = " << format_error( hlr::seq::norm::norm_2( *diff ) ) << std::endl;
    }

    {
        std::cout << "  " << term::bullet << " HLR" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        auto  C1 = impl::matrix::copy( *A );

        for ( int i = 0; i < nbench; ++i )
        {
            blas::FLOPS = 0;
            blas::reset_statistics();

            tic = timer::now();
        
            impl::multiply< value_t >( value_t(1), hpro::apply_normal, *A, hpro::apply_normal, *A, *C1, acc );

            toc = timer::since( tic );
            std::cout << "    mult in  " << format_time( toc ) << std::endl;

            flops.push_back( get_flops( "mm" ) );
            runtime.push_back( toc.seconds() );
        }// for
        
        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        auto  diff = hpro::matrix_sum( 1.0, mul_res.get(), -1.0, C1.get() );

        std::cout << "    error  = " << format_error( hlr::seq::norm::norm_2( *diff ) ) << std::endl;
    }

    if ( false && (( impl_name == "seq" ) || ( impl_name == "tbb" ))) // otherwise sequential !!!
    {
        std::cout << "  " << term::bullet << " Hpro" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        auto  C2 = impl::matrix::copy( *A );

        for ( int i = 0; i < nbench; ++i )
        {
            blas::FLOPS = 0;
            blas::reset_statistics();

            tic = timer::now();

            hpro::multiply< value_t >( value_t(1), hpro::apply_normal, A.get(), hpro::apply_normal, A.get(), value_t(1), C2.get(), acc );

            toc = timer::since( tic );
            std::cout << "    mult in  " << format_time( toc ) << std::endl;

            flops.push_back( get_flops( "mm" ) );
            runtime.push_back( toc.seconds() );
        }// for

        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        auto  diff = hpro::matrix_sum( 1.0, mul_res.get(), -1.0, C2.get() );

        std::cout << "    error  = " << format_error( hlr::seq::norm::norm_2( *diff ) ) << std::endl;
    }
    
    // auto  diff = hpro::matrix_sum( 1.0, C1.get(), -1.0, C2.get() );

    // std::cout << "    error  = " << format_error( hlr::seq::norm::norm_2( *diff ) ) << std::endl;
    
    //////////////////////////////////////////////////////////////////////
    //
    // LU factorization
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "LU ( Tile-H DAG " << impl_name
              << ", " << acc.to_string()
              << " )" << term::reset << std::endl;
    
    auto  C = impl::matrix::copy( *A );
        
    {
        std::cout << "  " << term::bullet << " full DAG" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        impl::matrix::copy_to( *A, *C );
        
        // no sparsification
        hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        
        tic = timer::now();
        
        // auto  dag = std::move( dag::gen_dag_lu_oop_auto( *C, nseq, impl::dag::refine ) );
        auto  dag = std::move( dag::gen_dag_lu_ip( *C, nseq, impl::dag::refine ) );
            
        toc = timer::since( tic );
            
        std::cout << "    DAG in   " << format_time( toc ) << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
        
        if ( verbose( 3 ) )
            dag.print_dot( "lu.dot" );
        
        for ( int i = 0; i < nbench; ++i )
        {
            blas::FLOPS = 0;
            blas::reset_statistics();

            tic = timer::now();
            
            impl::dag::run( dag, acc );

            toc = timer::since( tic );
            std::cout << "  LU in      " << format_time( toc ) << std::endl;
            
            flops.push_back( get_flops( "lu" ) );
            runtime.push_back( toc.seconds() );

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *C );
        }// for

        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }
   
    if ( false ) 
    {
        std::cout << "  " << term::bullet << " Tile-H DAG" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        impl::matrix::copy_to( *A, *C );
        
        // no sparsification
        hlr::dag::sparsify_mode = hlr::dag::sparsify_none;
        
        tic = timer::now();
        
        // auto  dag = std::move( dag::gen_dag_lu_oop_auto( *C, nseq, impl::dag::refine ) );
        auto  dag = std::move( dag::gen_dag_lu_tileh( *C, nseq, impl::dag::refine, impl::dag::run ) );
            
        toc = timer::since( tic );
            
        std::cout << "    DAG in   " << format_time( toc ) << std::endl;
        std::cout << "    #nodes = " << dag.nnodes() << std::endl;
        std::cout << "    #edges = " << dag.nedges() << std::endl;
        std::cout << "    mem    = " << format_mem( dag.mem_size() ) << std::endl;
        
        if ( verbose( 3 ) )
            dag.print_dot( "lu.dot" );
        
        for ( int i = 0; i < nbench; ++i )
        {
            blas::FLOPS = 0;
            blas::reset_statistics();

            tic = timer::now();
            
            impl::dag::run( dag, acc );

            toc = timer::since( tic );
            std::cout << "  LU in      " << format_time( toc ) << std::endl;
            
            flops.push_back( get_flops( "lu" ) );
            runtime.push_back( toc.seconds() );

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *C );
        }// for

        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }
    
    if ( false ) 
    {
        std::cout << "  " << term::bullet << " recursive+DAG" << std::endl;
        
        std::vector< double >  runtime, flops;
        
        impl::matrix::copy_to( *A, *C );
        
        for ( int i = 0; i < nbench; ++i )
        {
            blas::FLOPS = 0;
            blas::reset_statistics();

            tic = timer::now();
        
            impl::tileh::lu< HLIB::real >( C.get(), acc );
        
            toc = timer::since( tic );
            std::cout << "  LU in      " << format_time( toc ) << std::endl;
            
            flops.push_back( get_flops( "lu" ) );
            runtime.push_back( toc.seconds() );

            if ( i < (nbench-1) )
                impl::matrix::copy_to( *A, *C );
        }// for
        
        std::cout     << "    flops  = " << format_flops( min( flops ), min( runtime ) ) << std::endl;

        if ( nbench > 1 )
            std::cout << "  runtime = "
                      << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                      << std::endl;

        hpro::TLUInvMatrix  A_inv( C.get(), hpro::block_wise, hpro::store_inverse );
        
        std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        std::cout << "    error  = " << format_error( hpro::inv_approx_2( A.get(), & A_inv ) ) << std::endl;
    }
}
