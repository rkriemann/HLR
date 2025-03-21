//
// Project     : HLR
// File        : sparse.hh
// Description : example for sparse matrices
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <common.hh>
#include <common-main.hh>

#include <hpro/io/TMatrixIO.hh>

#include <hlr/arith/multiply.hh>
#include <hlr/matrix/sparse_matrix.hh>
#include <hlr/matrix/luinv_eval.hh>
#include <hlr/utils/io.hh>
#include <hlr/arith/blas_eigen.hh>

#include <hlr/seq/ipt.hh>

using namespace hlr;

template < typename value_t >
std::unique_ptr< Hpro::TSparseMatrix< value_t > >
read_csv ( const std::string &  filename )
{
    auto  in   = std::ifstream( filename );
    auto  line = std::string();

    // skip first line
    std::getline( in, line );

    // number of rows
    std::getline( in, line );

    size_t  nrows = atoi( line.c_str() );
    
    //
    // read number of rows/cols
    //
    
    auto    fpos  = in.tellg();
    size_t  ncols = 0;
    size_t  nnz   = 0;
    auto    rows = std::vector< uint >( nrows+1, 0 );
    auto    parts = std::vector< std::string >();
    
    while ( in.good() )
    {
        std::getline( in, line );
        Hpro::split( line, ",", parts );

        if ( line.size() == 0 )
            continue;
        
        if ( parts.size() != 3 )
            HLR_ERROR( "expected r,c,v" );

        auto  row = atoi( parts[0].c_str() )-1;
        auto  col = atoi( parts[1].c_str() )-1;

        rows[row]++;
        ncols = std::max< int >( ncols, col+1 );
        nnz++;
    }// while

    std::cout << nrows << " × " << ncols << std::endl;
    
    //
    // read data as CSV matrix
    //

    auto    S   = std::make_unique< Hpro::TSparseMatrix< value_t > >( Hpro::is( 0, nrows-1 ), Hpro::is( 0, ncols-1 ) );
    size_t  ofs = 0;

    S->init( nnz );

    for ( uint i = 0; i < nrows; i++ )
    {
        S->rowptr(i) = ofs;
        ofs += rows[i];
        rows[i] = 0; // reset for later usage
    }// for

    S->rowptr(nrows) = nnz;
    
    in.clear();
    in.seekg( fpos );
    
    while ( in.good() )
    {
        std::getline( in, line );
        Hpro::split( line, ",", parts );

        if ( line.size() == 0 )
            continue;
        
        if ( parts.size() != 3 )
            HLR_ERROR( "expected r,c,v" );

        auto  row = atoi( parts[0].c_str() )-1;
        auto  col = atoi( parts[1].c_str() )-1;
        auto  val = atof( parts[2].c_str() );

        ofs = S->rowptr(row) + rows[row];

        S->colind(ofs) = col;
        S->coeff(ofs)  = value_t(val);
        
        rows[row]++;
    }// while

    return S;
}

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
    // auto  M = read_csv< value_t >( sparsefile );
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
    
    // auto  part_strat    = Hpro::TMongooseAlgPartStrat();
    auto  part_strat    = Hpro::TMETISAlgPartStrat();
    auto  ct_builder    = Hpro::TAlgCTBuilder( & part_strat, ntile );
    // auto  nd_ct_builder = Hpro::TAlgNDCTBuilder( & ct_builder, ntile );
    // auto  cl            = nd_ct_builder.build( S );
    auto  cl            = ct_builder.build( S );

    
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
    // compute eigenvalues via IPT
    //

    {
        //
        // first as dense matrix
        //

        auto  M    = matrix::convert_to_dense( *A );
        auto  stat = blas::eigen_stat();
        auto  M2   = blas::copy( M->mat() );

        io::matlab::write( M->mat(), "M" );
        
        tic = timer::now();
            
        auto [ E, V ] = blas::eigen_ipt( M2, 1000, cmdline::eps, "frobenius", cmdline::verbosity, & stat );

        toc = timer::since( tic );
            
        std::cout << "IPT in    " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                
        io::matlab::write( V, "V1" );
        io::matlab::write( E, "E1" );
        
        if ( stat.converged )
            std::cout << "    error = " << format_error( blas::everror( M->mat(), E, V ) ) << std::endl;
    }
    
    {
        using approx_t = approx::SVD< value_t >;
        
        tic = timer::now();
        
        auto  apx      = approx_t();
        auto  [ V, E ] = impl::ipt( *A, cmdline::eps, acc, apx );
        
        io::eps::print( *V, "V", "noid,sv" );
            
        toc = timer::since( tic );

        std::cout << "H-IPT in    " << format_time( toc ) << std::endl;
                
        {
            auto  DM = matrix::convert_to_dense( *M );
            auto  DV = matrix::convert_to_dense( *V );

            io::matlab::write( DV->mat(), "V2" );
            io::matlab::write( E, "E2" );

            std::cout << "    error = " << format_error( blas::everror( DM->mat(), E, DV->mat() ) ) << std::endl;
        }

        return;
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
