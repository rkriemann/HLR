//
// Project     : HLR
// Module      : lu.cc
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <list>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <hpro/matrix/structure.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/utils/tensor.hh"
#include "hlr/dag/lu.hh"
#include "hlr/seq/matrix.hh"

namespace hlr { namespace dag {

namespace
{

using Hpro::id_t;

// identifiers for memory blocks
constexpr id_t  ID_A = 'A';
constexpr id_t  ID_L = 'L';
constexpr id_t  ID_U = 'U';

template < typename value_t >
struct lu_node : public node
{
    Hpro::TMatrix< value_t > *  A;
    
    lu_node ( Hpro::TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

template < typename value_t >
struct trsmu_node : public node
{
    const Hpro::TMatrix< value_t > *  U;
    Hpro::TMatrix< value_t > *        A;
    
    trsmu_node ( const Hpro::TMatrix< value_t > *  aU,
                 Hpro::TMatrix< value_t > *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "L%d = trsmu( U%d, A%d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

template < typename value_t >
struct trsml_node : public node
{
    const Hpro::TMatrix< value_t > *  L;
    Hpro::TMatrix< value_t > *        A;

    trsml_node ( const Hpro::TMatrix< value_t > *  aL,
                 Hpro::TMatrix< value_t > *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "U%d = trsml( L%d, A%d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};
    
template < typename value_t >
struct update_node : public node
{
    const Hpro::TMatrix< value_t > *  A;
    const Hpro::TMatrix< value_t > *  B;
    Hpro::TMatrix< value_t > *        C;

    update_node ( const Hpro::TMatrix< value_t > *  aA,
                  const Hpro::TMatrix< value_t > *  aB,
                  Hpro::TMatrix< value_t > *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "A%d = mul( L%d, U%d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const Hpro::TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, A->block_is() }, { ID_U, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A,    C->block_is() } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
lu_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BL  = BA;
        auto        BU  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();

        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            //
            // factorise diagonal block
            //
            
            auto  A_ii  = BA->block( i, i );
            auto  L_ii  = A_ii;
            auto  U_ii  = A_ii;

            assert( ! is_null_any( A_ii, L_ii, U_ii ) );

            finished( i, i ) = g.alloc_node< lu_node< value_t > >( A_ii );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( BA->block( j, i ) ) )
                {
                    finished( j, i ) = g.alloc_node< trsmu_node< value_t > >( U_ii, BA->block( j, i ) );
                    finished( j, i )->after( finished( i, i ) );
                }// if

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    finished( i, j ) = g.alloc_node< trsml_node< value_t > >( L_ii, BA->block( i, j ) );
                    finished( i, j )->after( finished( i, i ) );
                }// if
        }// for
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            for ( uint j = i+1; j < nbr; j++ )
            {
                for ( uint l = i+1; l < nbc; l++ )
                {
                    if ( ! is_null_any( BL->block( j, i ), BU->block( i, l ), BA->block( j, l ) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( BL->block( j, i ),
                                                                               BU->block( i, l ),
                                                                               BA->block( j, l ) );

                        update->after( finished( j, i ) );
                        update->after( finished( i, l ) );
                        finished( j, l )->after( update );
                    }// if
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
lu_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // Hpro::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_inverse, false ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsmu_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
trsmu_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, U ) && ! is_small_any( min_size, A, U ) )
    {
        auto        BU  = cptrcast( U, Hpro::TBlockMatrix< value_t > );
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();

        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            assert( ! is_null( U_jj ) );

            for ( uint i = 0; i < nbr; ++i )
                if ( ! is_null( BA->block(i,j) ) )
                    finished( i, j ) = g.alloc_node< trsmu_node< value_t > >(  U_jj, BA->block( i, j ) );
        }// for
        
        for ( uint j = 0; j < nbc; ++j )
        {
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( BX->block( i, j ),
                                                                               BU->block( j, k ),
                                                                               BA->block( i, k ) );

                        update->after( finished( i, j ) );
                        finished( i, k )->after( update );
                    }// if
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
trsmu_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
trsml_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, L ) && ! is_small_any( min_size, A, L ) )
    {
        auto        BL  = cptrcast( L, Hpro::TBlockMatrix< value_t > );
        auto        BA  = ptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto        BX  = BA;
        const auto  nbr = BA->nblock_rows();
        const auto  nbc = BA->nblock_cols();

        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            assert( ! is_null( L_ii ) );

            for ( uint j = 0; j < nbc; ++j )
                if ( ! is_null( BA->block( i, j ) ) )
                    finished( i, j ) = g.alloc_node< trsml_node< value_t > >(  L_ii, BA->block( i, j ) );
        }// for
        
        for ( uint i = 0; i < nbr; ++i )
        {
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( BL->block( k, i ),
                                                                               BX->block( i, j ),
                                                                               BA->block( k, j ) );

                        update->after( finished( i, j ) );
                        finished( k, j )->after( update );
                    }// if
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
trsml_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
update_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, Hpro::TBlockMatrix< value_t > );
        auto  BB = cptrcast( B, Hpro::TBlockMatrix< value_t > );
        auto  BC = ptrcast(  C, Hpro::TBlockMatrix< value_t > );

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node< value_t > >( BA->block( i, k ),
                                                                BB->block( k, j ),
                                                                BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
update_node< value_t >::run_ ( const Hpro::TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public function to generate DAG for LU
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
graph
gen_dag_lu_oop ( Hpro::TMatrix< value_t > &  A,
                 const size_t                min_size,
                 refine_func_t               refine )
{
    return refine( new lu_node( & A ), min_size, use_single_end_node );
}

#define INST_ALL( type ) \
    template graph gen_dag_lu_oop< type > ( Hpro::TMatrix< type > &, \
                                            const size_t           , \
                                            refine_func_t          );
    
INST_ALL( float )
INST_ALL( double )
INST_ALL( std::complex< float > )
INST_ALL( std::complex< double > )

}}// namespace hlr::dag

