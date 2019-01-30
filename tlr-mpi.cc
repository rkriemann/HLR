//
// Project     : HLib
// File        : tlr-tbb.cc
// Description : TLR arithmetic with TBB+MPI
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <memory>

using namespace std;

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

#include <hlib.hh>

using namespace HLIB;

namespace B = HLIB::BLAS;

#include "tlr.inc"

///////////////////////////////////////////////////////////////////////////////
//
// recursive approach
//

namespace TLR
{

namespace MPI
{

template < typename value_t >
void
lu ( TMatrix *          A,
     const TTruncAcc &  acc )
{
    mpi::communicator  world;
    const auto         my_proc = world.rank();
    
    if ( HLIB::verbose( 4 ) )
        DBG::printf( "lu( %d )", A->id() );

    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  nbr = BA->nblock_rows();
        auto  nbc = BA->nblock_cols();

        //
        // set up communicators for rows/cols
        //

        list< mpi::communicator >    comms;
        vector< mpi::communicator >  row_comms;
        vector< list< int > >        row_procs( nbr ), col_procs( nbc );

        std::cout << "rows" << std::endl;
        for ( uint  i = 0; i < nbr; ++i )
        {
            list< int >  procs;
            
            for ( uint  j = 0; j < nbc; ++j )
                procs.push_back( BA->block( i, j )->procs().master() );

            procs.sort();
            procs.unique();

            std::cout << i << " : "; for ( auto  p : procs ) std::cout << p << ", "; std::cout << std::endl;

            for ( uint  l = 0; l < i; ++l )
            {
                if ( procs == row_procs[l] )
                {
                    std::cout << i << " : " << l << std::endl;
                    break;
                }// if
            }// for
            
            row_procs[i] = std::move( procs );
        }// for

        std::cout << "columns" << std::endl;
        for ( uint  j = 0; j < nbc; ++j )
        {
            list< int >  procs;
            
            for ( uint  i = 0; i < nbr; ++i )
                procs.push_back( BA->block( i, j )->procs().master() );

            procs.sort();
            procs.unique();

            std::cout << j << " : "; for ( auto  p : procs ) std::cout << p << ", "; std::cout << std::endl;

            for ( uint  l = 0; l < j; ++l )
            {
                if ( procs == col_procs[l] )
                {
                    std::cout << j << " : " << l << std::endl;
                    break;
                }// if
            }// for
            
            col_procs[j] = std::move( procs );
        }// for




        
        for ( uint  i = 0; i < nbr; ++i )
        {
            auto  A_ii = BA->block( i, i );
            auto  p_ii = A_ii->procs().master();

            if ( my_proc == p_ii )
                TLR::MPI::lu< value_t >( A_ii, acc );

            //
            // broadcast diagonal block
            //

            unique_ptr< TDenseMatrix >  T_ii;           // temporary storage with auto-delete
            TDenseMatrix *              D_ii = nullptr; // dense handle for A_ii/T_ii

            if ( my_proc != p_ii )
            {
                T_ii = make_unique< TDenseMatrix >( A_ii->row_is(), A_ii->col_is(), A_ii->is_complex() );
                D_ii = T_ii.get();
            }// if
            else
                D_ii = ptrcast( A_ii, TDenseMatrix );
                
            mpi::broadcast( world, blas_mat< value_t >( D_ii ).data(), D_ii->nrows() * D_ii->ncols(), p_ii );

            //
            // off-diagonal solve
            //
            
            tbb::parallel_for( i+1, nbc,
                               [D_ii,BA,i,my_proc] ( uint  j )
                               {
                                   auto        A_ji = BA->block( j, i );
                                   const auto  p_ji = A_ji->procs().master();
                                   
                                   // L is unit diagonal !!!
                                   // trsml(  A_ii, BA->block( i, j ) ); // A01->blas_mat_A() );
                                   if ( my_proc == p_ji )
                                       trsmuh< value_t >( D_ii, BA->block( j, i ) ); // A10->blas_mat_B() );
                               } );

            //
            // update of trailing sub-matrix
            //
            
            // tbb::parallel_for( tbb::blocked_range2d< uint >( i+1, nbr,
            //                                                  i+1, nbc ),
            //                    [BA,i,&acc] ( const tbb::blocked_range2d< uint > & r )
            //                    {
            //                        for ( auto  j = r.rows().begin(); j != r.rows().end(); ++j )
            //                        {
            //                            for ( uint  l = r.cols().begin(); l != r.cols().end(); ++l )
            //                            {
            //                                update< value_t >( BA->block( j, i ), BA->block( i, l ), BA->block( j, l ), acc );
            //                            }// for
            //                        }// for
            //                    } );
        }// for
    }// if
    else
    {
        auto  DA = ptrcast( A, TDenseMatrix );
        
        B::invert( blas_mat< value_t >( DA ) );
    }// else
}

template
void
lu< HLIB::real > ( TMatrix *          A,
                   const TTruncAcc &  acc );

}// namespace MPI

}// namespace TLR
