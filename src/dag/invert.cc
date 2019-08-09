//
// Project     : HLib
// File        : invert.cc
// Description : DAGs for matrix inversion
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <cassert>

#include <matrix/structure.hh>
#include <algebra/mat_mul.hh>
#include <algebra/mat_inv.hh>

#include "hlr/seq/arith.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/invert.hh"

namespace hlr { namespace dag {

using namespace HLIB;

namespace
{

using id_t = HLIB::id_t;

struct id_matrix_t
{
    const id_t  id;
    TMatrix *   mat;

    id_matrix_t ( const id_t  aid,
                  TMatrix *   amat )
            : id(  aid )
            , mat( amat )
    {
        assert( ! is_null( mat ) );
    }

    id_matrix_t ( const id_t       aid,
                  const TMatrix *  amat )
            : id(  aid )
            , mat( const_cast< TMatrix * >( amat ) )
    {
        assert( ! is_null( mat ) );
    }
};

// memory block identifiers
const id_t  ID_A('A');
const id_t  ID_B('A');
const id_t  ID_C('A');
const id_t  ID_L('L');
const id_t  ID_U('U');
const id_t  ID_X('X');  // final result
const id_t  ID_T('T');  // intermediate result

struct inv_ll_node : public node
{
    TMatrix *          L;
    const diag_type_t  diag;
    
    inv_ll_node ( TMatrix *          aL,
                  const diag_type_t  adiag )
            : L( aL )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "X%d = inv_ll( L%d )", L->id(), L->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_X, L->block_is() } }; }
};

struct mul_ll_right_node : public node
{
    // const real         alpha; == 1
    TMatrix *          A;
    const TMatrix *    L;
    const diag_type_t  diag;
    
    mul_ll_right_node ( TMatrix *          aA,
                        const TMatrix *    aL,
                        const diag_type_t  adiag )
            : A( aA )
            , L( aL )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "T%d = mul_ll_r( X%d, L%d )", A->id(), A->id(), L->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_X, L->block_is() }, { ID_L, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_T, A->block_is() } }; }
};

struct mul_ll_left_node : public node
{
    // const real         alpha; = -1
    const TMatrix *    L;
    TMatrix *          A;
    const diag_type_t  diag;
    
    mul_ll_left_node ( const TMatrix *    aL,
                       TMatrix *          aA,
                       const diag_type_t  adiag )
            : L( aL )
            , A( aA )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "X%d = mul_ll_l( X%d, T%d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_X, L->block_is() }, { ID_T, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_X, A->block_is() } }; }
};

struct inv_ur_node : public node
{
    TMatrix *          U;
    const diag_type_t  diag;
    
    inv_ur_node ( TMatrix *          aU,
                  const diag_type_t  adiag )
            : U( aU )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "inv_ur( %d )", U->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, U->block_is() } }; }
};

struct mul_ur_left_node : public node
{
    // const real         alpha; == -1
    const TMatrix *    U;
    TMatrix *          A;
    const diag_type_t  diag;
    
    mul_ur_left_node ( const TMatrix *    aU,
                       TMatrix *          aA,
                       const diag_type_t  adiag )
            : U( aU )
            , A( aA )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "%d = mul_ur_l( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
};

struct mul_ur_right_node : public node
{
    // const real         alpha; = 1
    TMatrix *          A;
    const TMatrix *    U;
    const diag_type_t  diag;
    
    mul_ur_right_node ( TMatrix *          aA,
                        const TMatrix *    aU,
                        const diag_type_t  adiag )
            : A( aA )
            , U( aU )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "%d = mul_ur_r( %d, %d )", A->id(), A->id(), U->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
};

struct update_node : public node
{
    const real         alpha;
    const id_matrix_t  A;
    const id_matrix_t  B;
    const id_matrix_t  C;

    update_node ( const real         aalpha,
                  const id_matrix_t  aA,
                  const id_matrix_t  aB,
                  const id_matrix_t  aC )
            : alpha( aalpha )
            , A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%c%d = upd( %c%d, %c%d )", char(C.id), C.mat->id(), char(A.id), A.mat->id(), char(B.id), B.mat->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { A.id, A.mat->block_is() }, { B.id, B.mat->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { C.id, C.mat->block_is() } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// inv_ll_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
inv_ll_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked( L ) && ! is_small( min_size, L ) )
    {
        auto        BL  = ptrcast( L, TBlockMatrix );
        const uint  nbr = BL->nblock_rows();
        const uint  nbc = BL->nblock_cols();

        for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            assert( ! is_null( BL->block( i, i ) ) );

            hlr::dag::alloc_node< inv_ll_node >( g, BL->block( i, i ), diag );
        }// for

        //
        // preconstruct mul_ll_right nodes
        //
        
        for ( uint i = 0; i < nbr; i++ )
        {
            for ( uint j = 0; j < i; j++ )
            {
                if ( is_null( BL->block( i, j ) ) )
                    continue;

                hlr::dag::alloc_node< mul_ll_right_node >( g,
                                                           BL->block(i,j), // unmodified
                                                           BL->block(j,j), // finished
                                                           diag );

                hlr::dag::alloc_node< mul_ll_left_node >( g,
                                                          BL->block(i,i), // finished
                                                          BL->block(i,j), // unmodified
                                                          diag );

                for ( uint l = j+1; l < i; l++ )
                {
                    if ( ! is_null_any( BL->block(i,l), BL->block(l,j) ) )
                    {
                        hlr::dag::alloc_node< update_node >( g,
                                                             real(1),
                                                             id_matrix_t{ ID_L, BL->block(i,l) }, // unmodified
                                                             id_matrix_t{ ID_X, BL->block(l,j) }, // finished
                                                             id_matrix_t{ ID_T, BL->block(i,j) } ); // TODO: check ids
                    }// if
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
inv_ll_node::run_ ( const TTruncAcc &  acc )
{
    const inv_options_t  opts{ diag };
    
    invert_ll_rec( L, acc, opts );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ll_right_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
mul_ll_right_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( L, A ) && ! is_small_any( min_size, L, A ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix );
        auto        BA  = ptrcast(  A, TBlockMatrix );
        const uint  nbr = BA->nblock_rows();
        const uint  nbc = BA->nblock_cols();

        for ( uint j = 0; j < nbc; j++ )
        {
            for ( uint i = 0; i < nbr; i++ )
            {
                assert( ! is_null_any( BL->block(j,j), BA->block(i,j) ) );

                hlr::dag::alloc_node< mul_ll_right_node >( g, BA->block(i,j), BL->block(j,j), diag );

                for ( uint l = j+1; l < nbc; l++ )
                {
                    if ( is_null_any( BA->block(i,l), BL->block(l,j) ) )
                        continue;

                    hlr::dag::alloc_node< update_node >( g,
                                                         1.0,
                                                         id_matrix_t{ ID_L, BA->block(i,l) }, // unmodified block needed
                                                         id_matrix_t{ ID_X, BL->block(l,j) },
                                                         id_matrix_t{ ID_T, BA->block(i,j) } );
                }// for
            }// for
        }// for
    }// if

    return  g;
}

void
mul_ll_right_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ll_right( 1.0, A, L, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ll_left_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
mul_ll_left_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( L, A ) && ! is_small_any( min_size, L, A ) )
    {
        auto       BL  = cptrcast( L, TBlockMatrix );
        auto       BA  = ptrcast(  A, TBlockMatrix );
        const int  nbr = BA->nblock_rows();
        const int  nbc = BA->nblock_cols();

        for ( int i = nbr-1; i >= 0; i-- )
        {
            for ( int j = 0; j < nbc; j++ )
            {
                assert( ! is_null_any( BL->block(i,i), BA->block(i,j) ) );
                    
                hlr::dag::alloc_node< mul_ll_left_node >( g, BL->block(i,i), BA->block(i,j), diag );

                for ( int l = 0; l < i; l++ )
                {
                    if ( is_null_any( BL->block(i,l), BA->block(l,j) ) )
                        continue;
                    
                    hlr::dag::alloc_node< update_node >( g,
                                                         -1.0,
                                                         id_matrix_t{ ID_X, BL->block(i,l) },
                                                         id_matrix_t{ ID_L, BA->block(l,j) }, // unmodified block needed
                                                         id_matrix_t{ ID_T, BA->block(i,j) } );
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
mul_ll_left_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ll_left( -1.0, L, A, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// inv_ur_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
inv_ur_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked( U ) && ! is_small( min_size, U ) )
    {
        auto       BU  = ptrcast( U, TBlockMatrix );
        const int  nbr = BU->nblock_rows();
        const int  nbc = BU->nblock_cols();
        
        for ( int  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            assert( ! is_null( BU->block( i, i ) ) );

            hlr::dag::alloc_node< inv_ur_node >( g, BU->block( i, i ), diag );
        }// for
        
        for ( int j = nbc-1; j >= 0; j-- )
        {
            for ( int i = j-1; i >= 0; i-- )
            {
                if ( is_null( BU->block( i, j ) ) )
                    continue;

                hlr::dag::alloc_node< mul_ur_right_node >( g,
                                                           BU->block(i,j), // unmodified
                                                           BU->block(j,j), // finished
                                                           diag );

                for ( int l = i+1; l < j; l++ )
                {
                    if ( ! is_null_any( BU->block(i,l), BU->block(l,j) ) )
                        hlr::dag::alloc_node< update_node >( g, real(1),
                                                             id_matrix_t{ ID_U, BU->block(i,l) }, // unmodified
                                                             id_matrix_t{ ID_X, BU->block(l,j) }, // finished
                                                             id_matrix_t{ ID_X, BU->block(i,j) } ); // TODO: check ids
                }// for

                hlr::dag::alloc_node< mul_ur_left_node >( g,
                                                          BU->block(i,i), // finished
                                                          BU->block(i,j), // unmodified
                                                          diag );
            }// for
        }// for
    }// if

    return g;
}

void
inv_ur_node::run_ ( const TTruncAcc &  acc )
{
    const inv_options_t  opts{ diag };
    
    invert_ur_rec( U, acc, opts );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ur_left_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
mul_ur_left_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( U, A ) && ! is_small_any( min_size, U, A ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        auto        BA  = ptrcast(  A, TBlockMatrix );
        const uint  nbr = BA->nblock_rows();
        const uint  nbc = BA->nblock_cols();

        for ( uint i = 0; i < nbr; i++ )
        {
            for ( uint j = 0; j < nbc; j++ )
            {
                assert( ! is_null_any( BU->block(i,i), BA->block(i,j) ) );
                    
                hlr::dag::alloc_node< mul_ur_left_node >( g, BU->block(i,i), BA->block(i,j), diag );

                for ( uint l = i+1; l < nbr; l++ )
                {
                    if ( is_null_any( BU->block(i,l), BA->block(l,j) ) )
                        continue;

                    hlr::dag::alloc_node< update_node >( g,
                                                         -1.0,
                                                         id_matrix_t{ ID_U, BU->block(i,l) },
                                                         id_matrix_t{ ID_U, BA->block(l,j) }, // unmodified block needed
                                                         id_matrix_t{ ID_U, BA->block(i,j) } ); // TODO: check ids
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
mul_ur_left_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ur_left( -1.0, U, A, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ur_right_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
mul_ur_right_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( U, A ) && ! is_small_any( min_size, U, A ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        auto        BA  = ptrcast(  A, TBlockMatrix );
        const uint  nbr = BA->nblock_rows();
        const uint  nbc = BA->nblock_cols();

        for ( uint j = 0; j < nbc; j++ )
        {
            for ( uint i = 0; i < nbr; i++ )
            {
                assert( ! is_null_any( BU->block(j,j), BA->block(i,j) ) );

                hlr::dag::alloc_node< mul_ur_right_node >( g, BA->block(i,j), BU->block(j,j), diag );

                for ( uint l = 0; l < j; l++ )
                {
                    if ( is_null_any( BA->block(i,l), BU->block(l,j) ) )
                        continue;

                    hlr::dag::alloc_node< update_node >( g,
                                                         1.0,
                                                         id_matrix_t{ ID_U, BA->block(i,l) }, // unmodified block needed
                                                         id_matrix_t{ ID_U, BU->block(l,j) },
                                                         id_matrix_t{ ID_U, BA->block(i,j) } ); // TODO: check ids
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
mul_ur_right_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ur_right( 1.0, A, U, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
update_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( A.mat, B.mat, C.mat ) && ! is_small_any( min_size, A.mat, B.mat, C.mat ) )
    {
        auto  BA = cptrcast( A.mat, TBlockMatrix );
        auto  BB = cptrcast( B.mat, TBlockMatrix );
        auto  BC = ptrcast(  C.mat, TBlockMatrix );

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        hlr::dag::alloc_node< update_node >( g,
                                                             alpha,
                                                             id_matrix_t{ A.id, BA->block( i, k ) },
                                                             id_matrix_t{ B.id, BB->block( k, j ) },
                                                             id_matrix_t{ C.id, BC->block( i, j ) } );
                }// for
            }// for
        }// for
    }// if

    return g;
}

void
update_node::run_ ( const TTruncAcc &  acc )
{
    multiply( alpha,
              apply_normal, A.mat,
              apply_normal, B.mat,
              1.0, C.mat, acc );
}

}// namespace anonymous

///////////////////////////////////////////////////////////////////////////////////////
//
// public functions
//
///////////////////////////////////////////////////////////////////////////////////////

//
// compute DAG for lower triangular inversion of L
// - if <diag> == unit_diag, diagonal blocks are not modified
//
dag::graph
gen_dag_invert_ll ( HLIB::TMatrix *    L,
                    const diag_type_t  diag,
                    refine_func_t      refine )
{
    return refine( new inv_ll_node( L, diag ), HLIB::CFG::Arith::max_seq_size );
}

//
// compute DAG for upper triangular inversion of U
// - if <diag> == unit_diag, diagonal blocks are not modified
//
dag::graph
gen_dag_invert_ur ( HLIB::TMatrix *    U,
                    const diag_type_t  diag,
                    refine_func_t      refine )
{
    return refine( new inv_ur_node( U, diag ), HLIB::CFG::Arith::max_seq_size );
}

}}// namespace hlr::dag
