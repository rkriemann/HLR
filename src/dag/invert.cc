//
// Project     : HLR
// Module      : invert.cc
// Description : DAGs for matrix inversion
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2024. All Rights Reserved.
//

#include <cassert>

#include <hpro/matrix/structure.hh>

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

using id_t = Hpro::id_t;

template < typename value_t >
struct id_matrix_t
{
    const id_t  id;
    TMatrix< value_t > *   mat;

    id_matrix_t ( const id_t            aid,
                  TMatrix< value_t > *  amat )
            : id(  aid )
            , mat( amat )
    {
        assert( ! is_null( mat ) );
    }

    id_matrix_t ( const id_t                  aid,
                  const TMatrix< value_t > *  amat )
            : id(  aid )
            , mat( const_cast< TMatrix< value_t > * >( amat ) )
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

template < typename value_t,
           storage_type_t  store_mode >
struct lu_node : public node
{
    TMatrix< value_t > *  A;
    
    lu_node ( TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() }, { ID_U, A->block_is() } }; }
};

template < typename value_t,
           storage_type_t  store_mode >
struct trsmu_node : public node
{
    const TMatrix< value_t > *  U;
    TMatrix< value_t > *        A;
    
    trsmu_node ( const TMatrix< value_t > *  aU,
                 TMatrix< value_t > *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "L%d = trsmu( U%d, A%d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, A->block_is() } }; }
};

template < typename value_t,
           storage_type_t  store_mode >
struct trsml_node : public node
{
    const TMatrix< value_t > *  L;
    TMatrix< value_t > *        A;

    trsml_node ( const TMatrix< value_t > *  aL,
                 TMatrix< value_t > *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "U%d = trsml( L%d, A%d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, A->block_is() } }; }
};

template < typename value_t >
struct waz_node : public node
{
    TMatrix< value_t > *  A;
    
    waz_node ( TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "waz( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct inv_node : public node
{
    TMatrix< value_t > *  A;
    
    inv_node ( TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "inv( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct inv_ll_node : public node
{
    TMatrix< value_t > *             L;
    const diag_type_t     diag;
    const storage_type_t  storage;
    
    inv_ll_node ( TMatrix< value_t > *             aL,
                  const diag_type_t     adiag,
                  const storage_type_t  astorage )
            : L( aL )
            , diag( adiag )
            , storage( astorage )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "inv_ll( %d )", L->id(), L->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, L->block_is() } }; }
};

template < typename value_t >
struct mul_ll_right_node : public node
{
    const value_t         alpha;
    TMatrix< value_t > *          A;
    const TMatrix< value_t > *    L;
    const diag_type_t  diag;
    
    mul_ll_right_node ( const value_t         aalpha,
                        TMatrix< value_t > *          aA,
                        const TMatrix< value_t > *    aL,
                        const diag_type_t  adiag )
            : alpha( aalpha )
            , A( aA )
            , L( aL )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = mul_ll_r( %d, %d )",
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

template < typename value_t >
struct mul_ll_left_node : public node
{
    const value_t         alpha;
    const TMatrix< value_t > *    L;
    TMatrix< value_t > *          A;
    const diag_type_t  diag;
    
    mul_ll_left_node ( const value_t         aalpha,
                       const TMatrix< value_t > *    aL,
                       TMatrix< value_t > *          aA,
                       const diag_type_t  adiag )
            : alpha( aalpha )
            , L( aL )
            , A( aA )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = mul_ll_l( %d, %d )", A->id(), L->id(), A->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_L, L->block_is() }, { ID_L, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_L, L->block_is() }, { ID_L, A->block_is() } }; }
};

template < typename value_t >
struct inv_ur_node : public node
{
    TMatrix< value_t > *             U;
    const diag_type_t     diag;
    const storage_type_t  storage;
    
    inv_ur_node ( TMatrix< value_t > *             aU,
                  const diag_type_t     adiag,
                  const storage_type_t  astorage )
            : U( aU )
            , diag( adiag )
            , storage( astorage )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "inv_ur( %d )", U->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, U->block_is() } }; }
};

template < typename value_t >
struct mul_ur_left_node : public node
{
    const value_t         alpha;
    const TMatrix< value_t > *    U;
    TMatrix< value_t > *          A;
    const diag_type_t  diag;
    
    mul_ur_left_node ( const value_t         aalpha,
                       const TMatrix< value_t > *    aU,
                       TMatrix< value_t > *          aA,
                       const diag_type_t  adiag )
            : alpha( aalpha )
            , U( aU )
            , A( aA )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = mul_ur_l( %d, %d )", A->id(), U->id(), A->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct mul_ur_right_node : public node
{
    const value_t         alpha;
    TMatrix< value_t > *          A;
    const TMatrix< value_t > *    U;
    const diag_type_t  diag;
    
    mul_ur_right_node ( const value_t         aalpha,
                        TMatrix< value_t > *          aA,
                        const TMatrix< value_t > *    aU,
                        const diag_type_t  adiag )
            : alpha( aalpha )
            , A( aA )
            , U( aU )
            , diag( adiag )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "%d = mul_ur_r( %d, %d )", A->id(), A->id(), U->id() ); }
    virtual std::string  color     () const { return "4e9a06"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_U, U->block_is() }, { ID_A, A->block_is() } }; }
};

template < typename value_t >
struct update_node : public node
{
    const value_t       alpha;
    const id_t       id_A;
    const TMatrix< value_t > *  A;
    const id_t       id_B;
    const TMatrix< value_t > *  B;
    const id_t       id_C;
    TMatrix< value_t > *        C;

    update_node ( const value_t       aalpha,
                  const id_t       aid_A,
                  const TMatrix< value_t > *  aA,
                  const id_t       aid_B,
                  const TMatrix< value_t > *  aB,
                  const id_t       aid_C,
                  TMatrix< value_t > *        aC )
            : alpha( aalpha )
            , id_A( aid_A )
            , A( aA )
            , id_B( aid_B )
            , B( aB )
            , id_C( aid_C )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return Hpro::to_string( "%c%d = upd( %c%d, %c%d )",
                                                                      char(id_C), C->id(), char(id_A), A->id(), char(id_B), B->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_B, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_C, C->block_is() } }; }
};

template < typename value_t >
struct mul_ur_ll_node : public node
{
    TMatrix< value_t > *  A;
    
    mul_ur_ll_node ( TMatrix< value_t > *  aA )
            : A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return Hpro::to_string( "mul_ur_ll( %d )", A->id() ); }
    virtual std::string  color     () const { return "75507b"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t  min_size );
    virtual const block_list_t  in_blocks_   () const { return { { ID_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { ID_A, A->block_is() } }; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           storage_type_t  store_mode >
local_graph
lu_node< value_t, store_mode >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked( A ) && ! is_small( min_size, A ) )
    {
        auto        BA  = ptrcast( A, TBlockMatrix< value_t > );
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

            finished( i, i ) = g.alloc_node< lu_node< value_t, store_mode > >( A_ii );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( BA->block( j, i ) ) )
                {
                    finished( j, i ) = g.alloc_node< trsmu_node< value_t, store_mode > >( U_ii, BA->block( j, i ) );
                    finished( j, i )->after( finished( i, i ) );
                }// if

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    finished( i, j ) = g.alloc_node< trsml_node< value_t, store_mode > >( L_ii, BA->block( i, j ) );
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
                        auto  update = g.alloc_node< update_node< value_t > >( value_t(-1),
                                                                    ID_L, BL->block( j, i ),
                                                                    ID_U, BU->block( i, l ),
                                                                    ID_A, BA->block( j, l ) );

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

template < typename value_t,
           storage_type_t  store_mode >
void
lu_node< value_t, store_mode >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // Hpro::LU::factorise_rec( A, acc, fac_options_t( block_wise, store_mode, false ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsmu_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           storage_type_t  store_mode >
local_graph
trsmu_node< value_t, store_mode >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, U ) && ! is_small_any( min_size, A, U ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix< value_t > );
        auto        BA  = ptrcast( A, TBlockMatrix< value_t > );
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
                    finished( i, j ) = g.alloc_node< trsmu_node< value_t, store_mode > >(  U_jj, BA->block( i, j ) );
        }// for
        
        for ( uint j = 0; j < nbc; ++j )
        {
            for ( uint  k = j+1; k < nbc; ++k )
                for ( uint  i = 0; i < nbr; ++i )
                    if ( ! is_null_any( BA->block(i,k), BA->block(i,j), BU->block(j,k) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( value_t(-1),
                                                                    ID_L, BX->block( i, j ),
                                                                    ID_U, BU->block( j, k ),
                                                                    ID_A, BA->block( i, k ) );

                        update->after( finished( i, j ) );
                        finished( i, k )->after( update );
                    }// if
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           storage_type_t  store_mode >
void
trsmu_node< value_t, store_mode >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_mode ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t,
           storage_type_t  store_mode >
local_graph
trsml_node< value_t, store_mode >::refine_ ( const size_t  min_size )
{
    local_graph  g;

    if ( is_blocked_all( A, L ) && ! is_small_any( min_size, A, L ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix< value_t > );
        auto        BA  = ptrcast( A, TBlockMatrix< value_t > );
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
                    finished( i, j ) = g.alloc_node< trsml_node< value_t, store_mode > >(  L_ii, BA->block( i, j ) );
        }// for
        
        for ( uint i = 0; i < nbr; ++i )
        {
            for ( uint  k = i+1; k < nbr; ++k )
                for ( uint  j = 0; j < nbc; ++j )
                    if ( ! is_null_any( BA->block(k,j), BA->block(i,j), BL->block(k,i) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( value_t(-1),
                                                                    ID_L, BL->block( k, i ),
                                                                    ID_U, BX->block( i, j ),
                                                                    ID_A, BA->block( k, j ) );

                        update->after( finished( i, j ) );
                        finished( k, j )->after( update );
                    }// if
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t,
           storage_type_t  store_mode >
void
trsml_node< value_t, store_mode >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_mode ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// waz_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
waz_node< value_t >::refine_ ( const size_t )
{
    local_graph  g;

    auto  lu     = g.alloc_node< lu_node< value_t, store_inverse > >( A );
    auto  inv_ll = g.alloc_node< inv_ll_node< value_t > >( A, unit_diag,    store_inverse );
    auto  inv_ur = g.alloc_node< inv_ur_node< value_t > >( A, general_diag, store_inverse );

    inv_ll->after( lu );
    inv_ur->after( lu );

    g.finalize();
    
    return g;
}

template < typename value_t >
void
waz_node< value_t >::run_ ( const TTruncAcc & )
{
    HLR_ASSERT( false );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// inv_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
inv_node< value_t >::refine_ ( const size_t )
{
    local_graph  g;

    auto  lu     = g.alloc_node< lu_node< value_t, store_normal > >( A );
    auto  inv_ll = g.alloc_node< inv_ll_node< value_t > >( A, unit_diag,    store_normal );
    auto  inv_ur = g.alloc_node< inv_ur_node< value_t > >( A, general_diag, store_normal );
    auto  mul    = g.alloc_node< mul_ur_ll_node< value_t > >( A );

    inv_ll->after( lu );
    inv_ur->after( lu );

    mul->after( inv_ll );
    mul->after( inv_ur );

    g.finalize();
    
    return g;
}

template < typename value_t >
void
inv_node< value_t >::run_ ( const TTruncAcc & )
{
    HLR_ASSERT( false );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// inv_ll_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
inv_ll_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked( L ) && ! is_small( min_size, L ) )
    {
        auto        BL  = ptrcast( L, TBlockMatrix< value_t > );
        const uint  nbr = BL->nblock_rows();
        const uint  nbc = BL->nblock_cols();

        //
        // inversion tasks come first
        //
        
        tensor2< node * >  finished( nbr, nbc );
        
        for ( uint  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            assert( ! is_null( BL->block(i,i) ) );

            finished(i,i) = g.alloc_node< inv_ll_node< value_t > >( BL->block(i,i), diag, storage );
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

                mul_ll_right(i,j) = g.alloc_node< mul_ll_right_node< value_t > >( value_t(1),
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

                finished(i,j) = g.alloc_node< mul_ll_left_node< value_t > >( value_t(-1),
                                                                  BL->block(i,i), // finished
                                                                  BL->block(i,j), // after all updates
                                                                  diag );

                finished(i,j)->after( finished(i,i) );
                finished(i,j)->after( mul_ll_right(i,j) );
                
                for ( uint l = j+1; l < i; l++ )
                {
                    if ( ! is_null_any( BL->block(i,l), BL->block(l,j) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( value_t(1),
                                                                    ID_A, BL->block(i,l), // unmodified
                                                                    ID_A, BL->block(l,j), // finished
                                                                    ID_A, BL->block(i,j) ); // TODO: check ids

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

template < typename value_t >
void
inv_ll_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ASSERT( is_dense( L ) );
    
    HLR_ERROR( "todo" );

    // const inv_options_t  opts{ diag, storage };
    
    // invert_ll_rec( L, acc, opts );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ll_right_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
mul_ll_right_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( L, A ) && ! is_small_any( min_size, L, A ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix< value_t > );
        auto        BA  = ptrcast(  A, TBlockMatrix< value_t > );
        auto        BC  = ptrcast(  A, TBlockMatrix< value_t > );
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

                mul_ll_right(i,j) = g.alloc_node< mul_ll_right_node< value_t > >( alpha, BA->block(i,j), BL->block(j,j), diag );
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

                    auto  update = g.alloc_node< update_node< value_t > >( alpha,
                                                                ID_A, BA->block(i,l), // unmodified block needed
                                                                ID_A, BL->block(l,j),
                                                                ID_A, BC->block(i,j) );

                    mul_ll_right(i,l)->after( update ); // change only after unmodified matrix was used in update
                    update->after( mul_ll_right(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return  g;
}

template < typename value_t >
void
mul_ll_right_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // multiply_ll_right( alpha, A, L, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ll_left_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
mul_ll_left_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( L, A ) && ! is_small_any( min_size, L, A ) )
    {
        auto       BL  = cptrcast( L, TBlockMatrix< value_t > );
        auto       BA  = ptrcast(  A, TBlockMatrix< value_t > );
        auto       BC  = ptrcast(  A, TBlockMatrix< value_t > );
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
                    
                mul_ll_left(i,j) = g.alloc_node< mul_ll_left_node< value_t > >( alpha, BL->block(i,i), BA->block(i,j), diag );
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
                    
                    auto  update = g.alloc_node< update_node< value_t > >( alpha,
                                                                ID_A, BL->block(i,l),
                                                                ID_A, BA->block(l,j), // unmodified block needed
                                                                ID_A, BC->block(i,j) );
                    
                    mul_ll_left(l,j)->after( update );
                    update->after( mul_ll_left(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
mul_ll_left_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // multiply_ll_left( alpha, L, A, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// inv_ur_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
inv_ur_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked( U ) && ! is_small( min_size, U ) )
    {
        auto       BU  = ptrcast( U, TBlockMatrix< value_t > );
        const int  nbr = BU->nblock_rows();
        const int  nbc = BU->nblock_cols();
        
        //
        // inversion tasks come first
        //

        tensor2< node * >  finished( nbr, nbc );
        
        for ( int  i = 0; i < std::min( nbr, nbc ); ++i )
        {
            assert( ! is_null( BU->block( i, i ) ) );

            finished(i,i) = g.alloc_node< inv_ur_node< value_t > >( BU->block( i, i ), diag, storage );
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

                mul_ur_right(i,j) = g.alloc_node< mul_ur_right_node< value_t > >( value_t(1),
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

                finished(i,j) = g.alloc_node< mul_ur_left_node< value_t > >( value_t(-1),
                                                                  BU->block(i,i), // finished
                                                                  BU->block(i,j), // unmodified
                                                                  diag );

                finished(i,j)->after( mul_ur_right(i,j) );
                finished(i,j)->after( finished(i,i) );
                
                for ( int l = i+1; l < j; l++ )
                {
                    if ( ! is_null_any( BU->block(i,l), BU->block(l,j) ) )
                    {
                        auto  update = g.alloc_node< update_node< value_t > >( value_t(1),
                                                                    ID_A, BU->block(i,l), // unmodified
                                                                    ID_A, BU->block(l,j), // finished
                                                                    ID_A, BU->block(i,j) );

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

template < typename value_t >
void
inv_ur_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ASSERT( is_dense( U ) );

    HLR_ERROR( "todo" );
    
    // const inv_options_t  opts{ diag, storage };
    
    // invert_ur_rec( U, acc, opts );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ur_right_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
mul_ur_right_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( U, A ) && ! is_small_any( min_size, U, A ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix< value_t > );
        auto        BA  = ptrcast(  A, TBlockMatrix< value_t > );
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

                mul_ur_right(i,j) = g.alloc_node< mul_ur_right_node< value_t > >( alpha, BA->block(i,j), BU->block(j,j), diag );
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

                    auto  update = g.alloc_node< update_node< value_t > >( alpha,
                                                                ID_A, BA->block(i,l), // unmodified block needed
                                                                ID_A, BU->block(l,j),
                                                                ID_A, BA->block(i,j) );

                    mul_ur_right(i,l)->after( update );
                    update->after( mul_ur_right(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
mul_ur_right_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // multiply_ur_right( alpha, A, U, acc, eval_option_t( block_wise, diag, store_normal ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ur_left_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
mul_ur_left_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( U, A ) && ! is_small_any( min_size, U, A ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix< value_t > );
        auto        BA  = ptrcast(  A, TBlockMatrix< value_t > );
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
                    
                mul_ur_left(i,j) = g.alloc_node< mul_ur_left_node< value_t > >( alpha, BU->block(i,i), BA->block(i,j), diag );
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

                    auto  update = g.alloc_node< update_node< value_t > >( alpha,
                                                                ID_A, BU->block(i,l),
                                                                ID_B, BA->block(l,j), // unmodified block needed
                                                                ID_C, BA->block(i,j) );
                    
                    mul_ur_left(l,j)->after( update );
                    update->after( mul_ur_left(i,j) ); // apply update only after initial change
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
mul_ur_left_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // multiply_ur_left( alpha, U, A, acc, eval_option_t( block_wise, diag, store_normal ) );
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
        auto  BA = cptrcast( A, TBlockMatrix< value_t > );
        auto  BB = cptrcast( B, TBlockMatrix< value_t > );
        auto  BC = ptrcast(  C, TBlockMatrix< value_t > );

        for ( uint  i = 0; i < BC->nblock_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->nblock_cols(); ++j )
            {
                if ( is_null( BC->block( i, j ) ) )
                    continue;
                
                for ( uint  k = 0; k < BA->nblock_cols(); ++k )
                {
                    if ( ! is_null_any( BA->block( i, k ), BB->block( k, j ) ) )
                        g.alloc_node< update_node< value_t > >( alpha,
                                                     ID_A, BA->block( i, k ),
                                                     ID_B, BB->block( k, j ),
                                                     ID_C, BC->block( i, j ) );
                }// for
            }// for
        }// for
    }// if

    g.finalize();
    
    return g;
}

template < typename value_t >
void
update_node< value_t >::run_ ( const TTruncAcc &  acc )
{
    HLR_ERROR( "todo" );
    // multiply( alpha,
    //           apply_normal, A,
    //           apply_normal, B,
    //           value_t(1), C,
    //           acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// mul_ur_ll_node
//
///////////////////////////////////////////////////////////////////////////////////////

template < typename value_t >
local_graph
mul_ur_ll_node< value_t >::refine_ ( const size_t  min_size )
{
    local_graph  g;
    
    if ( is_blocked_all( A ) && ! is_small( min_size, A ) )
    {
        auto        BA  = ptrcast( A, TBlockMatrix< value_t > );
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

                if      ( i == j ) mul_tri(i,j) = g.alloc_node< mul_ur_ll_node< value_t >    >( BA->block(i,j) );
                else if ( i <  j ) mul_tri(i,j) = g.alloc_node< mul_ll_right_node< value_t > >( value_t(1), BA->block(i,j), BA->block(j,j), unit_diag );
                else               mul_tri(i,j) = g.alloc_node< mul_ur_left_node< value_t >  >( value_t(1), BA->block(i,i), BA->block(i,j), general_diag );
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
                        auto  update = g.alloc_node< update_node< value_t > >( value_t(1),
                                                                    ID_A, BA->block( i, k ),
                                                                    ID_B, BA->block( k, j ),
                                                                    ID_C, BA->block( i, j ) );

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

template < typename value_t >
void
mul_ur_ll_node< value_t >::run_ ( const TTruncAcc & )
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
template < typename value_t >
dag::graph
gen_dag_invert_ll ( Hpro::TMatrix< value_t > &    L,
                    const diag_type_t  diag,
                    const size_t       min_size,
                    refine_func_t      refine )
{
    return refine( new inv_ll_node< value_t >( & L, diag, store_normal ), min_size, use_single_end_node );
}

//
// compute DAG for upper triangular inversion of U
// - if <diag> == unit_diag, diagonal blocks are not modified
//
template < typename value_t >
dag::graph
gen_dag_invert_ur ( Hpro::TMatrix< value_t > &    U,
                    const diag_type_t  diag,
                    const size_t       min_size,
                    refine_func_t      refine )
{
    return refine( new inv_ur_node< value_t >( & U, diag, store_normal ), min_size, use_single_end_node );
}

//
// compute DAG for WAZ factorization of A
//
template < typename value_t >
dag::graph
gen_dag_waz ( Hpro::TMatrix< value_t > &  A,
              const size_t     min_size,
              refine_func_t    refine )
{
    auto  dag = refine( new waz_node< value_t >( &A ), min_size, use_single_end_node );

    return dag;
}

//
// compute DAG for inversion of A
//
template < typename value_t >
dag::graph
gen_dag_invert ( Hpro::TMatrix< value_t > &  A,
                 const size_t     min_size,
                 refine_func_t    refine )
{
    auto  dag = refine( new inv_node( &A ), Hpro::CFG::Arith::max_seq_size, use_multiple_end_nodes );

    return dag;



    
    auto  dag_lu     = gen_dag_lu_oop_auto( A, min_size, refine );
    
    // if ( verbose( 3 ) )
    //     dag_lu.print_dot( "lu.dot" );
    
    auto  dag_ll     = refine( new inv_ll_node< value_t >( &A, unit_diag,    store_inverse ), min_size, use_multiple_end_nodes );

    // if ( verbose( 3 ) )
    //     dag_ll.print_dot( "inv_ll.dot" );
    
    auto  dag_ur     = refine( new inv_ur_node< value_t >( &A, general_diag, store_inverse ), min_size, use_multiple_end_nodes );
    auto  dag_inv    = merge( dag_ll, dag_ur );

    auto  dag_lu_inv = concat( dag_lu, dag_inv );
    
    auto  dag_mul    = refine( new mul_ur_ll_node< value_t >( &A ), min_size, use_single_end_node );

    // if ( verbose( 3 ) )
    //     dag_mul.print_dot( "mul.dot" );
    
    auto  dag_all    = concat( dag_lu_inv, dag_mul );

    return dag_all;
}

}}// namespace hlr::dag
