//
// Project     : HLib
// File        : lu.cc
// Description : generate DAG for Tile-H LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2020. All Rights Reserved.
//

#include <hpro/matrix/structure.hh>
#include <hpro/algebra/solve_tri.hh>
#include <hpro/algebra/mat_mul.hh>
#include <hpro/algebra/mat_fac.hh>

#include "hlr/utils/tensor.hh"
#include "hlr/utils/checks.hh"
#include "hlr/utils/tools.hh"
#include "hlr/dag/lu.hh"

namespace hlr { namespace dag {

namespace hpro = HLIB;

using namespace hpro;

namespace
{

struct lu_node : public node
{
    TMatrix *      A;
    refine_func_t  refine;
    exec_func_t    exec;
    
    lu_node ( TMatrix *      aA,
              refine_func_t  arefine,
              exec_func_t    aexec )
            : A( aA )
            , refine( arefine )
            , exec( aexec )
    { init(); }

    virtual std::string  to_string () const { return hpro::to_string( "lu( %d )", A->id() ); }
    virtual std::string  color     () const { return "ef2929"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

struct trsmu_node : public node
{
    const TMatrix *  U;
    TMatrix *        A;
    refine_func_t    refine;
    exec_func_t      exec;
    
    trsmu_node ( const TMatrix *  aU,
                 TMatrix *        aA,
                 refine_func_t    arefine,
                 exec_func_t      aexec )
            : U( aU )
            , A( aA )
            , refine( arefine )
            , exec( aexec )
    { init(); }
    
    virtual std::string  to_string () const { return hpro::to_string( "solve_U( %d, %d )", U->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

struct trsml_node : public node
{
    const TMatrix *  L;
    TMatrix *        A;
    refine_func_t    refine;
    exec_func_t      exec;

    trsml_node ( const TMatrix *  aL,
                 TMatrix *        aA,
                 refine_func_t    arefine,
                 exec_func_t      aexec )
            : L( aL )
            , A( aA )
            , refine( arefine )
            , exec( aexec )
    { init(); }

    virtual std::string  to_string () const { return hpro::to_string( "solve_L( %d, %d )", L->id(), A->id() ); }
    virtual std::string  color     () const { return "729fcf"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};
    
struct update_node : public node
{
    const TMatrix *  A;
    const TMatrix *  B;
    TMatrix *        C;

    update_node ( const TMatrix *  aA,
                  const TMatrix *  aB,
                  TMatrix *        aC )
            : A( aA )
            , B( aB )
            , C( aC )
    { init(); }

    virtual std::string  to_string () const { return hpro::to_string( "update( %d, %d, %d )", A->id(), B->id(), C->id() ); }
    virtual std::string  color     () const { return "8ae234"; }
    
private:
    virtual void                run_         ( const TTruncAcc &  acc );
    virtual local_graph         refine_      ( const size_t ) { return {}; }
    virtual const block_list_t  in_blocks_   () const { return {}; }
    virtual const block_list_t  out_blocks_  () const { return {}; }
};

///////////////////////////////////////////////////////////////////////////////////////
//
// lu_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
lu_node::run_ ( const TTruncAcc &  acc )
{
    auto  dag = std::move( hlr::dag::gen_dag_lu_oop_auto( *A, 128, refine ) );
    
    exec( dag, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsml_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
trsml_node::run_ ( const TTruncAcc &  acc )
{
    auto  dag = std::move( hlr::dag::gen_dag_solve_lower( L, A, 128, refine ) );
    
    exec( dag, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// trsmu_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
trsmu_node::run_ ( const TTruncAcc &  acc )
{
    auto  dag = std::move( hlr::dag::gen_dag_solve_upper( U, A, 128, refine ) );
    
    exec( dag, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// update_node
//
///////////////////////////////////////////////////////////////////////////////////////

void
update_node::run_ ( const TTruncAcc &  acc )
{
    multiply( real(-1), apply_normal, A, apply_normal, B, real(1), C, acc );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// Tile-H algorithm
//
///////////////////////////////////////////////////////////////////////////////////////

//
// generate nodes for level-wise LU
//
graph
dag_lu_tileh ( TMatrix *      A,
               const size_t   min_size,
               refine_func_t  refine,
               exec_func_t    exec )
{
    if ( ! is_blocked( A ) )
    {
        //
        // single LU node for A
        //

        auto  node_A = new lu_node( A, refine, exec );

        return dag::graph( { node_A }, { node_A }, { node_A } );
    }// if
    else
    {
        //
        // single level LU for sub-blocks of A
        //

        node_list_t  nodes, start, end;
        auto         BA  = ptrcast( A, TBlockMatrix );
        auto         BL  = BA;
        auto         BU  = BA;
        const auto   nbr = BA->nblock_rows();
        const auto   nbc = BA->nblock_cols();

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

            finished( i, i ) = new lu_node( A_ii, refine, exec );

            nodes.push_back( finished( i, i ) );
            
            if ( i == 0 )
                start.push_back( finished( i, i ) );
            
            if ( i == std::min( nbr, nbc )-1 )
                end.push_back( finished( i, i ) );

            for ( uint j = i+1; j < nbr; j++ )
                if ( ! is_null( BA->block( j, i ) ) )
                {
                    finished( j, i ) = new trsmu_node( U_ii, BA->block( j, i ), refine, exec );
                    finished( j, i )->after( finished( i, i ) );
                    finished( j, i )->inc_dep_cnt();
                    nodes.push_back( finished( j, i ) );
                }// if

            for ( uint j = i+1; j < nbc; j++ )
                if ( ! is_null( BA->block( i, j ) ) )
                {
                    finished( i, j ) = new trsml_node( L_ii, BA->block( i, j ), refine, exec );
                    finished( i, j )->after( finished( i, i ) );
                    finished( i, j )->inc_dep_cnt();
                    nodes.push_back( finished( i, j ) );
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
                        auto  update = new update_node( BL->block( j, i ),
                                                        BU->block( i, l ),
                                                        BA->block( j, l ) );

                        update->after( finished( j, i ) );
                        update->inc_dep_cnt();
                        update->after( finished( i, l ) );
                        update->inc_dep_cnt();
                        finished( j, l )->after( update );
                        finished( j, l )->inc_dep_cnt();
                        nodes.push_back( update );
                    }// if
                }// for
            }// for
        }// for

        for ( auto  node : nodes )
            node->finalize();

        return dag::graph( nodes, start, end );
    }// else
}

}// namespace anonymous

graph
gen_dag_lu_tileh ( TMatrix &      A,
                   const size_t   min_size,
                   refine_func_t  refine,
                   exec_func_t    exec )
{
    return dag_lu_tileh( &A, min_size, refine, exec );
}

}}// namespace hlr::dag
