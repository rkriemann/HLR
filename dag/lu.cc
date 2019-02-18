//
// Project     : HLib
// File        : lu.cc
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <list>

#include <matrix/structure.hh>
#include <algebra/solve_tri.hh>
#include <algebra/mat_mul.hh>

#include "../tensor.hh"
#include "lu.hh"

using std::list;
using namespace HLIB;

namespace DAG
{

namespace LU
{

namespace
{

// identifiers for memory blocks
const HLIB::id_t  id_A = 'A';
const HLIB::id_t  id_U = 'U';

struct LUNode : public Node
{
    TMatrix *  A;
    
    LUNode ( TMatrix *  aA )
            : A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual void                refine_      ( list< Node * > &  subnodes );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct SolveUNode : public Node
{
    const TMatrix *     U;
    TMatrix *           A;
    
    SolveUNode ( const TMatrix *  aU,
                 TMatrix *        aA )
            : U( aU )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string () const { return HLIB::to_string( "solve_U( %d, %d )",
                                                                      U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual void                refine_      ( list< Node * > &  subnodes );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, U->block_is() }, { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};

struct SolveLNode : public Node
{
    const TMatrix *     L;
    TMatrix *           A;

    SolveLNode ( const TMatrix *  aL,
                 TMatrix *        aA )
            : L( aL )
            , A( aA )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "solve_L( %d, %d )",
                                                                      L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual void                refine_      ( list< Node * > &  subnodes );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, L->block_is() }, { id_A, A->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const { return { { id_A, A->block_is() } }; }
};
    
struct UpdateNode : public Node
{
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;

    UpdateNode ( const TMatrix *  aA,
                 const TMatrix *  aB,
                 TMatrix *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return HLIB::to_string( "update( %d, %d, %d )",
                                                                      A->id(), B->id(), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual void                refine_      ( list< Node * > &  subnodes );
    virtual const block_list_t  in_blocks_   () const { return { { id_A, A->block_is() }, { id_A, B->block_is() } }; }
    virtual const block_list_t  out_blocks_  () const
    {
        if ( CFG::Arith::use_accu ) return { { id_U, C->block_is() } };
        else                        return { { id_A, C->block_is() } };
    }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// LUNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
LUNode::refine_ ( list< Node * > &  subnodes )
{
    if ( is_blocked( A ) && ! is_small( A ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto        B   = ptrcast( A, TBlockMatrix );
        const auto  nbr = B->block_rows();
        const auto  nbc = B->block_cols();

        //
        // then create factorise/solve nodes for all blocks
        //

        tensor2< Node * >  nodes( nbr, nbc );
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            //
            // factorise diagonal block
            //
            
            auto  A_ii  = B->block( i, i );

            if ( A_ii == nullptr )
                HERROR( ERR_NULL, "(LUNode) refine", "diagonal block is nullptr" );

            auto  lu_ii = ::DAG::alloc_node< LUNode >( subnodes, A_ii );

            nodes(i,i) = lu_ii;

            for ( uint j = i+1; j < nbr; j++ )
            {
                if ( B->block( j, i ) != nullptr )
                {
                    auto solve_ji = ::DAG::alloc_node< SolveUNode >( subnodes, A_ii, B->block( j, i ) );

                    solve_ji->after( lu_ii );
                    nodes(j,i) = solve_ji;
                }// if
            }// for

            for ( uint j = i+1; j < nbc; j++ )
            {
                if ( B->block( i, j ) != nullptr )
                {
                    auto solve_ij = ::DAG::alloc_node< SolveLNode >( subnodes, A_ii, B->block( i, j ) );

                    solve_ij->after( lu_ii );
                    nodes(i,j) = solve_ij;
                }// if
            }// for
        }// for

        //
        // now create update nodes with dependencies
        //
        
        for ( uint i = 0; i < std::min( nbr, nbc ); ++i )
        {
            for ( uint j = i+1; j < nbr; j++ )
            {
                for ( uint l = i+1; l < nbc; l++ )
                {
                    if (( B->block( j, i ) != nullptr ) &&
                        ( B->block( i, l ) != nullptr ) &&
                        ( B->block( j, l ) != nullptr ))
                    {
                        auto update_jl = ::DAG::alloc_node< UpdateNode >( subnodes, B->block( j, i ), B->block( i, l ), B->block( j, l ) );
                        
                        update_jl->after( nodes(j,i) );
                        update_jl->after( nodes(i,l) );

                        if ( ! CFG::Arith::use_accu )
                            update_jl->before( nodes(j,l) );
                    }// if
                }// for
            }// for
        }// for
    }// if
}

void
LUNode::run_ ( const TTruncAcc &  acc )
{
    HLIB::LU::factorise_rec( A, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SolveLNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
SolveLNode::refine_ ( list< Node * > &  subnodes )
{
    if ( is_blocked_all( A, L ) && ! is_small_any( A, L ) )
    {
        auto        BL  = cptrcast( L, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        tensor2< Node * >  nodes( nbr, nbc );

        //
        // first create all solve nodes
        //
        
        for ( uint i = 0; i < nbr; ++i )
        {
            const auto  L_ii = BL->block( i, i );
        
            //
            // solve in current block row
            //

            if ( L_ii != nullptr )
            {
                for ( uint j = 0; j < nbc; ++j )
                {
                    if ( BA->block( i, j ) != nullptr )
                    {
                        auto  solve_ij = ::DAG::alloc_node< SolveLNode >( subnodes, L_ii, BA->block( i, j ) );

                        nodes(i,j) = solve_ij;
                    }// if
                }// for
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint i = 0; i < nbr; ++i )
        {
            for ( uint  k = i+1; k < nbr; ++k )
            {
                for ( uint  j = 0; j < nbc; ++j )
                {
                    if (( BA->block(k,j) != nullptr ) &&
                        ( BA->block(i,j) != nullptr ) &&
                        ( BL->block(k,i) != nullptr ))
                    {
                        auto  update_kj = ::DAG::alloc_node< UpdateNode >( subnodes, BL->block( k, i ), BA->block( i, j ), BA->block( k, j ) );

                        update_kj->after( nodes(i,j) );

                        if ( ! CFG::Arith::use_accu )
                            update_kj->before( nodes(k,j) );
                    }// if
                }// for
            }// for
        }// for
    }// if
}

void
SolveLNode::run_ ( const TTruncAcc &  acc )
{
    solve_lower_left( apply_normal, L, A, acc, solve_option_t( block_wise, unit_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SolveUNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
SolveUNode::refine_ ( list< Node * > &  subnodes )
{
    if ( is_blocked_all( A, U ) && ! is_small_any( A, U ) )
    {
        auto        BU  = cptrcast( U, TBlockMatrix );
        auto        BA  = ptrcast( A, TBlockMatrix );
        const auto  nbr = BA->block_rows();
        const auto  nbc = BA->block_cols();

        tensor2< Node * >  nodes( nbr, nbc );

        //
        // first create all solve nodes
        //
        
        for ( uint j = 0; j < nbc; ++j )
        {
            const auto  U_jj = BU->block( j, j );
        
            if ( U_jj != nullptr )
            {
                for ( uint i = 0; i < nbr; ++i )
                {
                    if ( BA->block(i,j) != nullptr )
                    {
                        auto solve_ij = ::DAG::alloc_node< SolveUNode >( subnodes, U_jj, BA->block( i, j ) );
                        
                        nodes(i,j) = solve_ij;
                    }// if
                }// for
            }// if
        }// for

        //
        // then create update nodes with dependencies
        //

        for ( uint j = 0; j < nbc; ++j )
        {
            for ( uint  k = j+1; k < nbc; ++k )
            {
                for ( uint  i = 0; i < nbr; ++i )
                {
                    if (( BA->block(i,k) != nullptr ) &&
                        ( BA->block(i,j) != nullptr ) &&
                        ( BU->block(j,k) != nullptr ))
                    {
                        auto  update_ik = ::DAG::alloc_node< UpdateNode >( subnodes, BA->block( i, j ), BU->block( j, k ), BA->block( i, k ) );

                        update_ik->after( nodes(i,j) );

                        if ( ! CFG::Arith::use_accu )
                            update_ik->before( nodes(i,k) );
                    }// if
                }// for
            }// for
        }// for
    }// if
}

void
SolveUNode::run_ ( const TTruncAcc &  acc )
{
    solve_upper_right( A, U, nullptr, acc, solve_option_t( block_wise, general_diag, store_inverse ) );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// UpdateNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
UpdateNode::refine_ ( list< Node * > &  subnodes )
{
    if ( is_blocked_all( A, B, C ) && ! is_small_any( A, B, C ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA = cptrcast( A, TBlockMatrix );
        auto  BB = cptrcast( B, TBlockMatrix );
        auto  BC = ptrcast(  C, TBlockMatrix );

        for ( uint  i = 0; i < BC->block_rows(); ++i )
        {
            for ( uint  j = 0; j < BC->block_cols(); ++j )
            {
                if ( BC->block( i, j ) == nullptr )
                    continue;
                
                for ( uint  k = 0; k < BA->block_cols(); ++k )
                {
                    if (( BA->block( i, k ) != nullptr ) && ( BB->block( k, j ) != nullptr ))
                    {
                        ::DAG::alloc_node< UpdateNode >( subnodes,
                                                         BA->block( i, k ),
                                                         BB->block( k, j ),
                                                         BC->block( i, j ) );
                    }// if
                }// for
            }// for
        }// for
    }// if
}

void
UpdateNode::run_ ( const TTruncAcc &  acc )
{
    multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
}

}// namespace anonymous

//
// public function to generate DAG for LU
//
Graph
gen_dag ( TMatrix *  A )
{
    return ::DAG::refine( new LUNode( A ) );
}

}// namespace LU

}// namespace DAG
