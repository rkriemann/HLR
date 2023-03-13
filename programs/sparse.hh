//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>

#include <hpro/io/TMatrixIO.hh>

#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/utils/io.hh>

using namespace hlr;

//
// main function
//
template < typename problem_t >
void
program_main ()
{
    using value_t = double;

    std::cout << term::bullet << term::bold << "Problem Setup" << term::reset << std::endl
              << "    sparse matrix = " << sparsefile
              << std::endl;

    auto  M = Hpro::read_matrix< value_t >( sparsefile );
    auto  S = ptrcast( M.get(), Hpro::TSparseMatrix< value_t > );

    std::cout << "    dims   = " << S->nrows() << " × " << S->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( S->byte_size() ) << std::endl;
    
    const auto  normS = hlr::norm::spectral( impl::arithmetic, *S, 1e-4 );
    
    std::cout << "    |S|    = " << format_norm( normS ) << std::endl;
    
    auto  tic = timer::now();
    auto  toc = timer::since( tic );

    //
    // convert to H
    //

    std::cout << term::bullet << term::bold << "Converting to H" << term::reset << std::endl;
    
    std::cout << "  " << term::bullet << "clustering" << std::endl;
    
    tic = timer::now();
    
    auto  part_strat    = Hpro::TMongooseAlgPartStrat();
    auto  ct_builder    = Hpro::TAlgCTBuilder( & part_strat, ntile );
    auto  nd_ct_builder = Hpro::TAlgNDCTBuilder( & ct_builder, ntile );
    auto  cl            = nd_ct_builder.build( S );
    auto  adm_cond      = Hpro::TWeakAlgAdmCond( S, cl->perm_i2e() );
    auto  bct_builder   = Hpro::TBCBuilder( 0, Hpro::cluster_level_any );
    auto  bcl           = bct_builder.build( cl.get(), cl.get(), & adm_cond );

    toc = timer::since( tic );
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    
    if ( verbose( 3 ) )
        io::eps::print( *bcl->root(), "bcl" );
    
    std::cout << "  " << term::bullet << "reordering sparse matrix" << std::endl;
    
    tic = timer::now();
    
    S->permute( *cl->perm_e2i(), *cl->perm_e2i() );

    toc = timer::since( tic );
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    
    if ( verbose( 3 ) )
        io::eps::print( *S, "S", "noid,pattern" );

    std::cout << "  " << term::bullet << "building matrix" << std::endl;
    
    tic = timer::now();
    
    auto  apx = approx::SVD< value_t >();
    auto  acc = gen_accuracy();
    auto  A   = impl::matrix::build_nd( *bcl->root(), *S, acc, apx, nseq );
    
    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
        io::eps::print( *A, "A", "pattern,noid" );

    {
        auto  diff  = matrix::sum( 1, *S, -1, *A );
        auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
        std::cout << "    error  = " << format_error( error, error / normS ) << std::endl;
    }

    //
    // factorize
    //
    
    std::cout << term::bullet << term::bold << "LU factorization" << term::reset << std::endl;

    auto  B = impl::matrix::copy( *A );

    tic = timer::now();

    impl::lu( *B, acc, apx );
    
    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( B->byte_size() ) << std::endl;

    auto  A_inv = matrix::luinv_eval( *B );
        
    std::cout << "    error  = " << format_error( norm::inv_error_2( *A, A_inv ) ) << std::endl;
    
    if ( verbose( 3 ) )
        io::eps::print( *B, "LU", "pattern,noid,nonempty" );
    
    //
    // convert to H using sparse instead of dense blocks
    //

    if constexpr ( false )
    {
        std::cout << term::bullet << term::bold << "Converting to H (sparse)" << term::reset << std::endl;

        auto  A2 = impl::matrix::build_sparse( *bcl->root(), *S, acc, apx, nseq );

        if ( verbose( 3 ) )
            io::eps::print( *A2, "A2", "pattern,noid" );
    
        std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;

        {
            auto  diff  = matrix::sum( 1, *S, -1, *A2 );
            auto  error = hlr::norm::spectral( impl::arithmetic, *diff, 1e-4 );
        
            std::cout << "    error  = " << format_error( error, error / normS ) << std::endl;
        }

        impl::lu< value_t >( *A2, acc, apx );
    }// if
}
