//
// Project     : HLR
// File        : rec-lu.hh
// Description : recursive LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>

#include <hpro/matrix/TMatBuilder.hh>

#include <hlr/approx/svd.hh>
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/utils/io.hh>
#include <hlr/bem/aca.hh>

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = typename problem_t::value_t;
    
    auto  tic = timer::now();
    auto  acc = gen_accuracy();
    auto  A   = std::unique_ptr< Hpro::TMatrix< value_t > >();

    if ( matrixfile == "" && sparsefile == "" )
    {
        auto  problem = gen_problem< problem_t >();
        auto  coord   = problem->coordinates();
        auto  ct      = gen_ct( *coord );
        auto  bct     = gen_bct( *ct, *ct );
    
        if ( verbose( 3 ) )
        {
            io::eps::print( * ct->root(), "ct" );
            io::eps::print( * bct->root(), "bct" );
        }// if
    
        auto  coeff  = problem->coeff_func();
        auto  pcoeff = Hpro::TPermCoeffFn< value_t >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
        auto  lrapx  = bem::aca_lrapx< Hpro::TPermCoeffFn< value_t > >( pcoeff );

        A = impl::matrix::build( bct->root(), pcoeff, lrapx, acc, nseq );
    }// if
    else if ( matrixfile != "" )
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    matrix = " << matrixfile
                  << std::endl;

        A = Hpro::read_matrix< value_t >( matrixfile );

        // for spreading memory usage
        if ( docopy )
            A = impl::matrix::realloc( A.release() );
    }// if
    else if ( sparsefile != "" )
    {
        std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
                  << "    sparse matrix = " << sparsefile
                  << std::endl;

        auto  M = Hpro::read_matrix< value_t >( sparsefile );
        auto  S = ptrcast( M.get(), Hpro::TSparseMatrix< value_t > );

        // convert to H
        auto  part_strat    = Hpro::TMongooseAlgPartStrat();
        auto  ct_builder    = Hpro::TAlgCTBuilder( & part_strat, ntile );
        auto  nd_ct_builder = Hpro::TAlgNDCTBuilder( & ct_builder, ntile );
        auto  cl            = nd_ct_builder.build( S );
        auto  adm_cond      = Hpro::TWeakAlgAdmCond( S, cl->perm_i2e() );
        auto  bct_builder   = Hpro::TBCBuilder();
        auto  bcl           = bct_builder.build( cl.get(), cl.get(), & adm_cond );
        auto  apx           = approx::SVD< value_t >();
        // auto  h_builder     = Hpro::TSparseMatBuilder< value_t >( S, cl->perm_i2e(), cl->perm_e2i() );

        S->permute( *cl->perm_e2i(), *cl->perm_e2i() );
        
        if ( verbose( 3 ) )
        {
            io::eps::print( * cl->root(), "ct" );
            io::eps::print( * bcl->root(), "bct" );
        }// if

        A = impl::matrix::build_nd( *bcl->root(), *S, acc, apx, nseq );
    }// else

    auto  toc = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( Hpro::verbose( 3 ) )
        io::eps::print( *A, "A" );

    std::cout << term::bullet << term::bold << "LU (rec, " << impl_name << ")" << term::reset << ", " << acc.to_string() 
              << ", nseq = " << nseq << std::endl;

    //////////////////////////////////////////////////////////////////////
    //
    // factorization
    //
    //////////////////////////////////////////////////////////////////////

    auto  apx     = approx::SVD< value_t >();
    auto  C       = std::shared_ptr( impl::matrix::copy( *A ) );
    auto  runtime = std::vector< double >();

    runtime.clear();
        
    for ( int  i = 0; i < nbench; ++i )
    {
        tic = timer::now();

        if ( sparsefile != "" )
            impl::lu_nd< value_t >( *C, acc, apx );
        else
            impl::lu< value_t >( *C, acc, apx );
        
        toc = timer::since( tic );

        std::cout << "  LU in      " << format_time( toc ) << std::endl;

        runtime.push_back( toc.seconds() );

        if ( i < (nbench-1) )
            impl::matrix::copy_to( *A, *C );
    }// for
        
    if ( nbench > 1 )
        std::cout << "  runtime  = "
                  << format( "%.3e s / %.3e s / %.3e s" ) % min( runtime ) % median( runtime ) % max( runtime )
                  << std::endl;
        
    std::cout << "    mem    = " << format_mem( C->byte_size() ) << std::endl;
        
    auto  A_inv = matrix::luinv_eval( *C );
        
    std::cout << "    error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
}
