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
#include "hlr/utils/tensor.hh"
#include "hlr/dag/lu.hh"
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
constexpr id_t  ID_A('A');
constexpr id_t  ID_B('A');
constexpr id_t  ID_C('A');
constexpr id_t  ID_L('A');
constexpr id_t  ID_U('A');

struct inv_ll_node : public node
{
    TMatrix *             L;
    const diag_type_t     diag;
    const storage_type_t  storage;
    
    inv_ll_node ( TMatrix *             aL,
                  const diag_type_t     adiag,
                  const storage_type_t  astorage )
            : L( aL )
            , diag( adiag )
            , storage( astorage )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "inv_ll( %d )", L->id(), L->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, L->block_is() } }; }
};

struct mul_ll_right_node : public node
{
    const real         alpha;
    TMatrix *          A;
    const TMatrix *    L;
    const diag_type_t  diag;
    
    mul_ll_right_node ( const real         aalpha,
                        TMatrix *          aA,
                        const TMatrix *    aL,
                        const diag_type_t  adiag )
            : alpha( aalpha )
            , A( aA )
            , L( aL )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "%d = mul_ll_r( %d, %d )",
                                                                      A->id(),
                                                                      A->id(),
                                                                      L->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_L, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, L->block_is() }, { ID_L, A->block_is() } }; }
};

struct mul_ll_left_node : public node
{
    const real         alpha;
    const TMatrix *    L;
    TMatrix *          A;
    const diag_type_t  diag;
    
    mul_ll_left_node ( const real         aalpha,
                       const TMatrix *    aL,
                       TMatrix *          aA,
                       const diag_type_t  adiag )
            : alpha( aalpha )
            , L( aL )
            , A( aA )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "%d = mul_ll_l( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_L, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, L->block_is() }, { ID_L, A->block_is() } }; }
};

struct inv_ur_node : public node
{
    TMatrix *             U;
    const diag_type_t     diag;
    const storage_type_t  storage;
    
    inv_ur_node ( TMatrix *             aU,
                  const diag_type_t     adiag,
                  const storage_type_t  astorage )
            : U( aU )
            , diag( adiag )
            , storage( astorage )
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
    const real         alpha;
    const TMatrix *    U;
    TMatrix *          A;
    const diag_type_t  diag;
    
    mul_ur_left_node ( const real         aalpha,
                       const TMatrix *    aU,
                       TMatrix *          aA,
                       const diag_type_t  adiag )
            : alpha( aalpha )
            , U( aU )
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
    const real         alpha;
    TMatrix *          A;
    const TMatrix *    U;
    const diag_type_t  diag;
    
    mul_ur_right_node ( const real         aalpha,
                        TMatrix *          aA,
                        const TMatrix *    aU,
                        const diag_type_t  adiag )
            : alpha( aalpha )
            , A( aA )
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
    const real       alpha;
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;

    update_node ( const real       aalpha,
                  const TMatrix *  aA,
                  const TMatrix *  aB,
                  TMatrix *        aC )
            : alpha( aalpha )
            , A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "%d = upd( %d, %d )", C->id(), A->id(), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() }, { ID_B, B->block_is() }, { ID_C, C->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() }, { ID_B, B->block_is() }, { ID_C, C->block_is() } }; }
};

struct mul_ur_ll_node : public node
{
    TMatrix *  A;
    
    mul_ur_ll_node ( TMatrix *  aA )
            : A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "mul_ur_ll( %d )", A->id() ); }
    virtual std::string  color     () const { return "75507b"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
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

        //
        // inversion tasks come first
        //
        
        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            assert( ! is_null( BL->block(i,i) ) );

            finished(i,i) = g.alloc_node< inv_ll_node >( BL->block(i,i), diag, storage );
        }// for

        //
        // mul_ll_right before all others
        //
        
        tensor2< node * >  mul_ll_right( nbr, nbc );
        
        for ( uint i = 0; i < nbr; i++ )
        {
            for ( uint j = 0; j < i; j++ )
            {
                if ( is_null( BL->block( i, j ) ) )
                    continue;

                mul_ll_right(i,j) = g.alloc_node< mul_ll_right_node >( real(1),
                                                                       BL->block(i,j), // unmodified
                                                                       BL->block(j,j), // finished
                                                                       diag );
                mul_ll_right(i,j)->after( finished(j,j) );
            }// for
        }// for
        
        for ( uint i = 0; i < nbr; i++ )
        {
            for ( uint j = 0; j < i; j++ )
            {
                if ( is_null( BL->block( i, j ) ) )
                    continue;

                finished(i,j) = g.alloc_node< mul_ll_left_node >( real(-1),
                                                                  BL->block(i,i), // finished
                                                                  BL->block(i,j), // after all updates
                                                                  diag );

                finished(i,j)->after( finished(i,i) );
                finished(i,j)->after( mul_ll_right(i,j) );
                
                for ( uint l = j+1; l < i; l++ )
                {
                    if ( ! is_null_any( BL->block(i,l), BL->block(l,j) ) )
                    {
                        auto  update = g.alloc_node< update_node >( real(1),
                                                                    BL->block(i,l), // unmodified
                                                                    BL->block(l,j), // finished
                                                                    BL->block(i,j) ); // TODO: check ids

                        update->after( mul_ll_right(i,j) );
                        update->after( finished(l,j) );
                        finished(i,j)->after( update );
                        mul_ll_right(i,l)->after( update );
                    }// if
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
inv_ll_node::run_ ( const TTruncAcc &  acc )
{
    HLR_ASSERT( is_dense( L ) );
    
    const inv_options_t  opts{ diag, storage };
    
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
        auto        BC  = ptrcast(  A, TBlockMatrix );
        const uint  nbr = BC->nblock_rows();
        const uint  nbc = BC->nblock_cols();

        //
        // create mul_ll_right tasks for reference below
        //

        tensor2< node * >  mul_ll_right( nbr, nbc );
        
        for ( uint j = 0; j < nbc; j++ )
        {
            for ( uint i = 0; i < nbr; i++ )
            {
                assert( ! is_null_any( BL->block(j,j), BA->block(i,j), BC->block(i,j) ) );

                mul_ll_right(i,j) = g.alloc_node< mul_ll_right_node >( alpha, BA->block(i,j), BL->block(j,j), diag );
            }// for
        }// for

        //
        // update tasks
        //
        
        for ( uint j = 0; j < nbc; j++ )
        {
            for ( uint i = 0; i < nbr; i++ )
            {
                assert( ! is_null_any( BL->block(j,j), BA->block(i,j), BC->block(i,j) ) );

                for ( uint l = j+1; l < nbc; l++ )
                {
                    if ( is_null_any( BA->block(i,l), BL->block(l,j) ) )
                        continue;

                    auto  update = g.alloc_node< update_node >( alpha,
                                                                BA->block(i,l), // unmodified block needed
                                                                BL->block(l,j),
                                                                BC->block(i,j) );

                    mul_ll_right(i,l)->after( update ); // change only after unmodified matrix was used in update
                    update->after( mul_ll_right(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return  g;
}

void
mul_ll_right_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ll_right( alpha, A, L, acc, eval_option_t( block_wise, diag, store_normal ) );
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
        auto       BC  = ptrcast(  A, TBlockMatrix );
        const int  nbr = BC->nblock_rows();
        const int  nbc = BC->nblock_cols();

        //
        // create mul_ll_left tasks for reference below
        //

        tensor2< node * >  mul_ll_left( nbr, nbc );

        for ( int i = nbr-1; i >= 0; i-- )
        {
            for ( int j = 0; j < nbc; j++ )
            {
                assert( ! is_null_any( BL->block(i,i), BA->block(i,j), BC->block(i,j) ) );
                    
                mul_ll_left(i,j) = g.alloc_node< mul_ll_left_node >( alpha, BL->block(i,i), BA->block(i,j), diag );
            }// for
        }// for

        //
        // then the update tasks
        //
        
        for ( int i = nbr-1; i >= 0; i-- )
        {
            for ( int j = 0; j < nbc; j++ )
            {
                assert( ! is_null_any( BL->block(i,i), BA->block(i,j), BC->block(i,j) ) );

                for ( int l = 0; l < i; l++ )
                {
                    if ( is_null_any( BL->block(i,l), BA->block(l,j) ) )
                        continue;
                    
                    auto  update = g.alloc_node< update_node >( alpha,
                                                                BL->block(i,l),
                                                                BA->block(l,j), // unmodified block needed
                                                                BC->block(i,j) );
                    
                    mul_ll_left(l,j)->after( update );
                    update->after( mul_ll_left(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
mul_ll_left_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ll_left( alpha, L, A, acc, eval_option_t( block_wise, diag, store_normal ) );
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
        
        //
        // inversion tasks come first
        //

        tensor2< node * >  finished( nbr, nbc );
        
        for ( int  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            assert( ! is_null( BU->block( i, i ) ) );

            finished(i,i) = g.alloc_node< inv_ur_node >( BU->block( i, i ), diag, storage );
        }// for
        
        //
        // mul_ll_right before all others
        //
        
        tensor2< node * >  mul_ur_right( nbr, nbc );
        
        for ( int j = nbc-1; j >= 0; j-- )
        {
            for ( int i = j-1; i >= 0; i-- )
            {
                if ( is_null( BU->block( i, j ) ) )
                    continue;

                mul_ur_right(i,j) = g.alloc_node< mul_ur_right_node >( real(1),
                                                                       BU->block(i,j), // unmodified
                                                                       BU->block(j,j), // finished
                                                                       diag );
                mul_ur_right(i,j)->after( finished(j,j) );
            }// for
        }// for

        for ( int j = nbc-1; j >= 0; j-- )
        {
            for ( int i = j-1; i >= 0; i-- )
            {
                if ( is_null( BU->block( i, j ) ) )
                    continue;

                finished(i,j) = g.alloc_node< mul_ur_left_node >( real(-1),
                                                                  BU->block(i,i), // finished
                                                                  BU->block(i,j), // unmodified
                                                                  diag );

                finished(i,j)->after( mul_ur_right(i,j) );
                finished(i,j)->after( finished(i,i) );
                
                for ( int l = i+1; l < j; l++ )
                {
                    if ( ! is_null_any( BU->block(i,l), BU->block(l,j) ) )
                    {
                        auto  update = g.alloc_node< update_node >( real(1),
                                                                    BU->block(i,l), // unmodified
                                                                    BU->block(l,j), // finished
                                                                    BU->block(i,j) );

                        update->after( mul_ur_right(i,j) );
                        update->after( finished(l,j) );
                        finished(i,j)->after( update );
                        mul_ur_right(i,l)->after( update ); // must not change before applying update
                    }// if
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
inv_ur_node::run_ ( const TTruncAcc &  acc )
{
    HLR_ASSERT( is_dense( U ) );
    
    const inv_options_t  opts{ diag, storage };
    
    invert_ur_rec( U, acc, opts );
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

        //
        // create mul_ur_right tasks for reference below
        //

        tensor2< node * >  mul_ur_right( nbr, nbc );
        
        for ( uint j = 0; j < nbc; j++ )
        {
            for ( uint i = 0; i < nbr; i++ )
            {
                assert( ! is_null_any( BU->block(j,j), BA->block(i,j) ) );

                mul_ur_right(i,j) = g.alloc_node< mul_ur_right_node >( alpha, BA->block(i,j), BU->block(j,j), diag );
            }// for
        }// for

        //
        // then the update tasks
        //
        
        for ( uint j = 0; j < nbc; j++ )
        {
            for ( uint i = 0; i < nbr; i++ )
            {
                assert( ! is_null_any( BU->block(j,j), BA->block(i,j) ) );

                for ( uint l = 0; l < j; l++ )
                {
                    if ( is_null_any( BA->block(i,l), BU->block(l,j) ) )
                        continue;

                    auto  update = g.alloc_node< update_node >( alpha,
                                                                BA->block(i,l), // unmodified block needed
                                                                BU->block(l,j),
                                                                BA->block(i,j) );

                    mul_ur_right(i,l)->after( update );
                    update->after( mul_ur_right(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
mul_ur_right_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ur_right( alpha, A, U, acc, eval_option_t( block_wise, diag, store_normal ) );
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

        //
        // create mul_ur_left tasks for reference below
        //

        tensor2< node * >  mul_ur_left( nbr, nbc );

        for ( uint i = 0; i < nbr; i++ )
        {
            for ( uint j = 0; j < nbc; j++ )
            {
                assert( ! is_null_any( BU->block(i,i), BA->block(i,j) ) );
                    
                mul_ur_left(i,j) = g.alloc_node< mul_ur_left_node >( alpha, BU->block(i,i), BA->block(i,j), diag );
            }// for
        }// for

        //
        // then the update tasks
        //
        
        for ( uint i = 0; i < nbr; i++ )
        {
            for ( uint j = 0; j < nbc; j++ )
            {
                assert( ! is_null_any( BU->block(i,i), BA->block(i,j) ) );

                for ( uint l = i+1; l < nbr; l++ )
                {
                    if ( is_null_any( BU->block(i,l), BA->block(l,j) ) )
                        continue;

                    auto  update = g.alloc_node< update_node >( alpha,
                                                                BU->block(i,l),
                                                                BA->block(l,j), // unmodified block needed
                                                                BA->block(i,j) );
                    
                    mul_ur_left(l,j)->after( update );
                    update->after( mul_ur_left(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
mul_ur_left_node::run_ ( const TTruncAcc &  acc )
{
    multiply_ur_left( alpha, U, A, acc, eval_option_t( block_wise, diag, store_normal ) );
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
    
    if ( is_blocked_all( A, B, C ) && ! is_small_any( min_size, A, B, C ) )
    {
        auto  BA = cptrcast( A, TBlockMatrix );
        auto  BB = cptrcast( B, TBlockMatrix );
        auto  BC = ptrcast(  C, TBlockMatrix );

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node >( alpha,
                                                     BA->block( i, k ),
                                                     BB->block( k, j ),
                                                     BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
update_node::run_ ( const TTruncAcc &  acc )
{
    multiply( alpha,
              apply_normal, A,
              apply_normal, B,
              real(1), C,
              acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ur_ll_node
//
///////////////////////////////////////////////////////////////////////////////////////

local_graph
mul_ur_ll_node::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( A ) && ! is_small( min_size, A ) )
    {
        auto        BA  = ptrcast( A, TBlockMatrix );
        const uint  nbr = BA->nblock_rows();
        const uint  nbc = BA->nblock_cols();
        
        //
        // first, tasks for the in-place update 
        //
        
        tensor2< node * >  mul_tri( nbr, nbc );

        for ( uint  i = 0; i < nbr; ++i )
        {
            for ( uint  j = 0; j < nbc; ++j )
            {
                assert( ! is_null( BA->block( i, j ) ) );

                if      ( i == j ) mul_tri(i,j) = g.alloc_node< mul_ur_ll_node    >( BA->block(i,j) );
                else if ( i <  j ) mul_tri(i,j) = g.alloc_node< mul_ll_right_node >( real(1), BA->block(i,j), BA->block(j,j), unit_diag );
                else               mul_tri(i,j) = g.alloc_node< mul_ur_left_node  >( real(1), BA->block(i,i), BA->block(i,j), general_diag );
            }// for
        }// for

        //
        // then all update tasks
        //
        
        for ( uint  i = 0; i < nbr; ++i )
        {
            for ( uint  j = 0; j < nbc; ++j )
            {
                assert( ! is_null( BA->block( i, j ) ) );

                // for multiplication above, diagonal block must not be modified
                if      ( i < j ) mul_tri(j,j)->after( mul_tri(i,j) );
                else if ( i > j ) mul_tri(i,i)->after( mul_tri(i,j) );
                
                for ( uint  k = std::max(i,j)+1; k < nbc; ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BA->block( k, j ) ) )
                    {
                        auto  update = g.alloc_node< update_node >( real(1),
                                                                    BA->block( i, k ),
                                                                    BA->block( k, j ),
                                                                    BA->block( i, j ) );

                        update->after( mul_tri(i,j) );
                        mul_tri(i,k)->after( update );
                        mul_tri(k,j)->after( update );
                    }// if
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

void
mul_ur_ll_node::run_ ( const TTruncAcc & )
{
     // block-wise operation with lower left part being identity, so result is upper right part (or A)
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
    return refine( new inv_ll_node( L, diag, store_normal ), HLIB::CFG::Arith::max_seq_size );
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
    return refine( new inv_ur_node( U, diag, store_normal ), HLIB::CFG::Arith::max_seq_size );
}

//
// compute DAG for inversion of A
//
dag::graph
gen_dag_invert ( HLIB::TMatrix *  A,
                 refine_func_t    refine )
{
    auto  dag_lu     = gen_dag_lu_rec( A, refine );
    
    auto  dag_ll     = refine( new inv_ll_node( A, unit_diag,    store_inverse ), HLIB::CFG::Arith::max_seq_size );
    auto  dag_ur     = refine( new inv_ur_node( A, general_diag, store_inverse ), HLIB::CFG::Arith::max_seq_size );
    auto  dag_inv    = merge( dag_ll, dag_ur );

    // dag_inv.print_dot( "inv.dot" );
    
    auto  dag_lu_inv = concat( dag_lu, dag_inv );
    
    // dag_lu_inv.print_dot( "lu_inv.dot" );
    
    auto  dag_mul    = refine( new mul_ur_ll_node( A ), HLIB::CFG::Arith::max_seq_size );

    // dag_mul.print_dot( "mul.dot" );
    
    auto  dag_all    = concat( dag_lu_inv, dag_mul );

    return std::move( dag_all );
}

}}// namespace hlr::dag
