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

#include <hlr/arith/multiply.hh>
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
            auto  pj  = M.col_ofs() + j;
            auto  d_j = d( pj );
            
            for ( uint  i = 0; i < DD.nrows(); ++i )
            {
                auto  pi  = M.row_ofs() + i;
                auto  d_i = d( pi );

                if ( pi == pj )
                    DD(i,j) = value_t(0);
                else
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

        tic = timer::now();
        
        using approx_t = approx::SVD< value_t >;
        
        auto  apx = approx_t();
            
        auto  M = impl::matrix::copy( *A );
        auto  d = impl::matrix::diagonal( *M );
        auto  Δ = impl::matrix::copy( *M );
        auto  G = build_G( *M, d, acc, apx );
        auto  Z = impl::matrix::identity( *M );
        auto  T = impl::matrix::copy_struct( *M );

        blas::scale( value_t(-1), d );
        hlr::add_diag( *Δ, d );
        
        //
        // iteration
        //

        using  real_t = real_type_t< value_t >;

        uint  sweep      = 0;
        uint  max_sweeps = M->nrows();
        auto  tolerance  = cmdline::eps;
        auto  old_error  = real_t(1);
        
        do
        {
            //
            // iteration step:
            //
            //   F(Z) := I + G ⊗ ( Z·diag(Δ·Z) - Δ·Z )
            //         = I + G ⊗ ( Z·diag(T) - T )   with T = Δ·Z
            //
            
            // T = Δ·Z
            impl::multiply( value_t(1),
                            apply_normal, *Δ,
                            apply_normal, *Z,
                            *T, acc, apx );

            // {
            //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
            //     auto  DT = impl::matrix::convert_to_dense( *T );

            //     io::matlab::write( DT->mat(), Hpro::to_string( "Ra%d", sweep ) );
            // }
            // {
            //     auto  T1 = matrix::convert_to_dense( *T );

            //     io::matlab::write( T1->mat(), "T" );
            // }

            auto  dT = impl::matrix::diagonal( *T );

            //
            // T := Z·diag(T) - T
            //
            
            // • ZT := Z·diag(T)
            auto  ZT = impl::matrix::copy( *Z );
            
            hlr::multiply_diag( *ZT, dT );
            
            // {
            //     auto  T1 = matrix::convert_to_dense( *ZT );

            //     io::matlab::write( T1->mat(), "ZT" );
            // }

            // • T := - T + ZT
            T->scale( value_t(-1) );
            impl::add( value_t(1), *ZT, *T, acc, apx );

            // {
            //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
            //     auto  DT = impl::matrix::convert_to_dense( *T );

            //     io::matlab::write( DT->mat(), Hpro::to_string( "Rb%d", sweep ) );
            // }

            // {
            //     auto  T1 = matrix::convert_to_dense( *T );

            //     io::matlab::write( T1->mat(), "T" );
            // }

            //
            // T := I + G ⊗ T
            //
            
            // • T := G ⊗ T
            impl::multiply_hadamard( value_t(1), *T, *G, acc, apx );
                                     
            // {
            //     auto  T1 = matrix::convert_to_dense( *T );

            //     io::matlab::write( T1->mat(), "T" );
            // }

            // • T := I - T
            hlr::add_identity( *T, value_t(1) );

            // {
            //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
            //     auto  DT = impl::matrix::convert_to_dense( *T );

            //     io::matlab::write( DT->mat(), Hpro::to_string( "Rc%d", sweep ) );
            // }

            // {
            //     auto  T1 = matrix::convert_to_dense( *T );

            //     io::matlab::write( T1->mat(), "T" );
            // }

            //
            // compute error ||Z-T||_F
            //

            impl::add( value_t(-1), *T, *Z, acc, apx );

            auto  error = impl::norm::frobenius( *Z );

            //
            // test stop criterion
            //

            impl::matrix::copy_to( *T, *Z );

            // {
            //     // auto  DV = io::matlab::read( Hpro::to_string( "V%d", sweep ) );
            //     auto  DZ = impl::matrix::convert_to_dense( *Z );

            //     io::matlab::write( DZ->mat(), Hpro::to_string( "Z%d", sweep ) );
            // }

            if ( verbosity >= 1 )
            {
                std::cout << "    sweep " << sweep << " : error = " << error;

                if ( sweep > 0 )
                    std::cout << ", reduction = " << error / old_error;
            
                std::cout << std::endl;
            }// if

            if (( sweep > 0 ) && ( error / old_error > real_t(10) ))
                break;
        
            old_error = error;
        
            ++sweep;

            if ( error < tolerance )
                break;

            if ( ! std::isnormal( error ) )
                break;
        
        } while ( sweep < max_sweeps );

        io::eps::print( *Z, "V", "noid,sv" );
            
        auto  T2 = impl::matrix::copy( *M );
            
        impl::multiply( value_t(1),
                        apply_normal, *Δ,
                        apply_normal, *Z,
                        *T2, acc, apx );

        auto  E = impl::matrix::diagonal( *T2 );

        toc = timer::since( tic );

        std::cout << "H-IPT in    " << format_time( toc ) << " (" << sweep << " sweeps)" << std::endl;
                
        {
            auto  DM = matrix::convert_to_dense( *M );
            auto  V  = matrix::convert_to_dense( *Z );

            io::matlab::write( V->mat(), "V2" );
            io::matlab::write( E, "E2" );

            std::cout << "    error = " << format_error( blas::everror( DM->mat(), E, V->mat() ) ) << std::endl;
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
