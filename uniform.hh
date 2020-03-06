//
// Project     : HLR
// File        : uniform.hh
// Description : program for testing uniform matrix arithmetic
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/cluster/TClusterBasisBuilder.hh>
#include <hpro/io/TClusterBasisVis.hh>

#include <hlr/matrix/cluster_basis.hh>

#include "common.hh"
#include "common-main.hh"

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
    auto  A   = std::unique_ptr< hpro::TMatrix >();

    auto  problem = gen_problem< problem_t >();
    auto  coord   = problem->coordinates();
    auto  ct      = gen_ct( *coord );
    auto  bct     = gen_bct( *ct, *ct );
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSBlockClusterVis  bc_vis;
        
        print_ps( ct->root(), "ct" );
        bc_vis.id( false ).print( bct->root(), "bct" );
    }// if
    
    auto  coeff  = problem->coeff_func();
    auto  pcoeff = std::make_unique< hpro::TPermCoeffFn< value_t > >( coeff.get(), ct->perm_i2e(), ct->perm_i2e() );
    auto  lrapx  = std::make_unique< hpro::TACAPlus< value_t > >( pcoeff.get() );
    
    A = impl::matrix::build( bct->root(), *pcoeff, *lrapx, acc, nseq );

    auto  toc    = timer::since( tic );
    
    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    dims   = " << A->nrows() << " × " << A->ncols() << std::endl;
    std::cout << "    mem    = " << format_mem( A->byte_size() ) << std::endl;
    
    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).id( true ).print( A.get(), "A" );
    }// if

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////

    {
        std::cout << term::bullet << term::bold << "H² conversion" << term::reset << std::endl;

        std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
        tic = timer::now();
    
        auto  [ rowcb, colcb ] = matrix::construct_from_H< value_t >( *ct->root(), *ct->root(), *A, acc );

        toc = timer::since( tic );

        std::cout << "    done in  " << format_time( toc ) << std::endl;
        std::cout << "    mem    = " << format_mem( rowcb->byte_size() ) << " / " << format_mem( colcb->byte_size() ) << std::endl;
    }

    //////////////////////////////////////////////////////////////////////
    //
    // conversion to H²
    //
    //////////////////////////////////////////////////////////////////////
    
    std::cout << term::bullet << term::bold << "H² conversion" << term::reset << std::endl;

    std::cout << "  " << term::bullet << term::bold << "build cluster bases" << term::reset << std::endl;
    
    hpro::THClusterBasisBuilder< value_t >  bbuilder;

    tic = timer::now();
    
    auto  [ rowcb, colcb ] = bbuilder.build( ct->root(), ct->root(), A.get(), acc );

    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( rowcb->byte_size() ) << " / " << format_mem( colcb->byte_size() ) << std::endl;

    if ( verbose( 3 ) )
    {
        hpro::TPSClusterBasisVis< value_t >  cbvis;

        cbvis.print( rowcb.get(), "rowcb.eps" );
    }// if

    std::cout << "  " << term::bullet << term::bold << "convert matrix" << term::reset << std::endl;

    tic = timer::now();
    
    auto  A2 = to_h2( A.get(), rowcb.get(), colcb.get() );
    
    toc = timer::since( tic );

    std::cout << "    done in  " << format_time( toc ) << std::endl;
    std::cout << "    mem    = " << format_mem( A2->byte_size() ) << std::endl;
    std::cout << "    error  = " << format_error( hpro::diff_norm_2( A.get(), A2.get() ) ) << std::endl;

    if ( hpro::verbose( 3 ) )
    {
        hpro::TPSMatrixVis  mvis;
        
        mvis.svd( false ).print( A2.get(), "A2" );
    }// if

}
