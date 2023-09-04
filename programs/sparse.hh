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
#include <hlr/arith/blas_eigen.hh>

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

        if ( parts.size() != 3 )
            HLR_ERROR( "expected r,c,v" );

        auto  row = atoi( parts[0].c_str() );
        auto  col = atoi( parts[1].c_str() );

        rows[row-1]++;
        ncols = std::max< int >( ncols, col );
        nnz++;
    }// while

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
    
    in.seekg( fpos );
    
    while ( in.good() )
    {
        std::getline( in, line );
        Hpro::split( line, ",", parts );

        if ( parts.size() != 3 )
            HLR_ERROR( "expected r,c,v" );

        auto  row = atoi( parts[0].c_str() );
        auto  col = atoi( parts[1].c_str() );
        auto  val = atof( parts[2].c_str() );

        --row; --col;

        ofs = S->rowptr(row) + rows[row];

        S->colind(ofs) = col;
        S->coeff(ofs)  = value_t(val);
        
        rows[row]++;
    }// while

    return S;
}

//
// see IPT iteration below
//
template < typename value_t,
           typename approx_t >
std::unique_ptr< Hpro::TMatrix< value_t > >
build_G ( const Hpro::TMatrix< value_t > &  M,
          const blas::vector< value_t > &   d,
          const accuracy &                  acc,
          const approx_t &                  apx )
{
    if ( is_blocked( M ) )
    {
        auto  BM = cptrcast( &M, Hpro::TBlockMatrix< value_t > );
        auto  N  = std::make_unique< Hpro::TBlockMatrix< value_t > >();
        auto  B  = ptrcast( N.get(), Hpro::TBlockMatrix< value_t > );

        B->copy_struct_from( BM );
        
        for ( uint  i = 0; i < B->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < B->nblock_cols(); ++j )
            {
                if ( BM->block( i, j ) != nullptr )
                {
                    auto  B_ij = build_G( * BM->block( i, j ), d, acc, apx );
                    
                    B_ij->set_parent( B );
                    B->set_block( i, j, B_ij.release() );
                }// if
            }// for
        }// for

        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_dense( M ) )
    {
        auto  N  = M.copy();
        auto  D  = ptrcast( N.get(), matrix::dense_matrix< value_t > );
        auto  DD = D->mat();
            
        HLR_ASSERT( ! D->is_compressed() );
            
        for ( uint  j = 0; j < DD.ncols(); ++j )
        {
            auto  d_j = d( M.col_ofs() + j );
            
            for ( uint  i = 0; i < DD.nrows(); ++i )
            {
                auto  d_i = d( M.row_ofs() + i );
                
                DD(i,j) = value_t(1) / ( d_i - d_j );
            }// for
        }// for
        
        N->set_id( M.id() );
        
        return N;
    }// if
    else if ( matrix::is_lowrank( M ) )
    {
        HLR_ASSERT( M.row_is() != M.col_is() );
        
        auto  N = M.copy_struct();
        auto  R = ptrcast( N.get(), matrix::lrmatrix< value_t > );

        // TODO: make efficient!
        auto  D = blas::matrix< value_t >( M.nrows(), M.ncols() );
        
        for ( uint  j = 0; j < D.ncols(); ++j )
        {
            auto  d_j = d( M.col_ofs() + j );
            
            for ( uint  i = 0; i < D.nrows(); ++i )
            {
                auto  d_i = d( M.row_ofs() + i );
                
                D(i,j) = value_t(1) / ( d_i - d_j );
            }// for
        }// for

        auto  [ U, V ] = apx( D, acc );

        R->set_lrmat( std::move( U ), std::move( V ) );

        return  N;
    }// if
    else
        HLR_ERROR( "todo" );
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

    // auto  M = Hpro::read_matrix< value_t >( sparsefile );
    // auto  S = ptrcast( M.get(), Hpro::TSparseMatrix< value_t > );
    auto  S = read_csv< value_t >( sparsefile );

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
    auto  cl            = ct_builder.build( S.get() );
    auto  adm_cond      = Hpro::TWeakAlgAdmCond( S.get(), cl->perm_i2e() );
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
        
        tic = timer::now();
            
        auto [ E, V ] = blas::eigen_ipt( M2, 1e-14, 1000, "frobenius", cmdline::verbosity, & stat );

        toc = timer::since( tic );
            
        std::cout << "IPT in    " << format_time( toc ) << " (" << stat.nsweeps << " sweeps)" << std::endl;
                
        if ( stat.converged )
            std::cout << "    error = " << format_error( blas::everror( M->mat(), E, V ) ) << std::endl;
    }
    
    {
        //
        // initial setup:
        //
        //   D = diag(M)
        //
        //   Δ = M - D
        //
        //       ⎧ 1 / ( d_ii - d_jj ) , i ≠ j
        //   G = ⎨ 
        //       ⎩ 0                   , i = j
        //
        //   Z = I
        //

        auto  apx = approx::SVD< value_t >();
            
        auto  M = std::move( A );
        auto  d = impl::matrix::diagonal( *M );
        auto  Δ = impl::matrix::copy( *M );
        auto  G = build_G( *M, d, acc, apx );
        auto  Z = impl::matrix::identity( *M );

        blas::scale( value_t(-1), d );
        hlr::add_diag( *Δ, d );
        
        // do
        // {
        //     //
        //     // iteration step:
        //     //
        //     //   F(Z) := I + G ⊗ ( Z·diag(Δ·Z) - Δ·Z )
        //     //         = I + G ⊗ ( Z·diag(T) - T )   with T = Δ·Z
        //     //
            
        //     // T = Δ·V
        //     prod( value_t(1), Delta, V, value_t(0), T );
        
        //     // T = Δ·V - V·diag(Δ·V) = T - V·diag(T) 
        //     // computed as T(i,:) = T(i,:) - T(i,i) · V(i,:)
        //     for ( size_t  i = 0; i < nrows; ++i )
        //         diag_T(i) = T(i,i);
        
        //     for ( size_t  i = 0; i < nrows; ++i )
        //     {
        //         auto  V_i = V.column(i);
        //         auto  T_i = T.column(i);

        //         add( -diag_T(i), V_i, T_i );
        //     }// for

        //     // I - Θ ∗ (Δ·V - V·diag(Δ·V)) = I - Θ ∗ T
        //     // compute I - Θ⊗M with Θ_ij = 1 / ( m_ii - m_jj )
        //     hmul_theta( T );

        //     //
        //     // compute error ||V-T||_F
        //     //

        //     real_t  error = 0;

        //     if (( error_type == "frobenius" ) || ( error_type == "fro" ))
        //     {
        //         add( value_t(-1), T, V );
        //         error = norm_F( V );
        //     }// if
        //     else if (( error_type == "maximum" ) || ( error_type == "max" ))
        //     {
        //         add( value_t(-1), T, V );
        //         error = norm_max( V );
        //     }// if
        //     else if (( error_type == "residual" ) || ( error_type == "res" ))
        //     {
        //         // // extract eigenvalues as diag( M + Δ·V ) and eigenvectors as V
        //         // // (T holds new V)
        //         // std::vector< value_t >  E( n );

        //         // for ( int  i = 0; i < n; ++i )
        //         //     E[i] = diag_M[ i ] + dot( n, Delta + i, n, T.data() + i*n, 1 );

        //         // // copy diagonal back to M
        //         // copy( n, diag_M.data(), 1, M.data(), n+1 );
        //         // gemm( 'N', 'N', n, n, n, value_t(1), M.data(), n, T.data(), n, value_t(0), V.data(), n );
        //         // for ( int  i = 0; i < n; ++i )
        //         // {
        //         //     axpy( n, -E[i], T.data() + i*n, 1, V.data() + i*n, 1 );
        //         //     M[ i*n+i ] = value_t(0); // reset diagonal for Delta
        //         // }// for
            
        //         // error = normF( n, n, V ) / ( M_norm * norm1( n, n, T ) );
        //     }// if
        //     else
        //         HLR_ERROR( "unknown error type" );

        //     //
        //     // test stop criterion
        //     //

        //     copy( T, V );

        //     if ( verbosity >= 1 )
        //     {
        //         std::cout << "    sweep " << sweep << " : error = " << error;

        //         if ( sweep > 0 )
        //             std::cout << ", reduction = " << error / old_error;
            
        //         std::cout << std::endl;
        //     }// if

        //     if (( sweep > 0 ) && ( error / old_error > real_t(10) ))
        //         return { vector< value_t >(), matrix< value_t >() };
        
        //     old_error = error;
        
        //     ++sweep;

        //     if ( ! is_null( stat ) )
        //         stat->nsweeps = sweep;
        
        //     if ( error < tolerance )
        //     {
        //         if ( ! is_null( stat ) )
        //             stat->converged = true;
            
        //         break;
        //     }// if

        //     if ( ! std::isnormal( error ) )
        //         break;
        
        // } while ( sweep < max_sweeps );

        // reset afterwards
        A = std::move( M );
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
