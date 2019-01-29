//
// Project     : HLib
// File        : hodlr-lu.cc
// Description : example for dense H-matrix using a 1d integral equation
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2018. All Rights Reserved.
//

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <deque>

using namespace std;

#include <boost/format.hpp>

using boost::format;

#include <tbb/parallel_invoke.h>
#include <tbb/spin_mutex.h>

#include <hlib.hh>
#include "../src/include/term.hh"
#include "../src/include/vtrace.hh"

using namespace HLIB;
using namespace HLIB::Term;

namespace B = HLIB::BLAS;

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif

using mutex_t       = tbb::spin_mutex;
using scoped_lock_t = mutex_t::scoped_lock;

//
// global constants
//
size_t  k     = 10;
size_t  ntile = 64;


//
// coefficient function for log|x-y| in [0,1]
//
class TLogCoeffFn : public TCoeffFn< real_t >
{
private:
    // stepwidth
    const double  _h;

public:
    // constructor
    TLogCoeffFn ( const double  h )
            : _h(h)
    {}

    //
    // coefficient evaluation
    //
    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const
    {
        const size_t  n = rowidxs.size();
        const size_t  m = colidxs.size();

        for ( size_t  j = 0; j < m; ++j )
        {
            const idx_t  idx1 = colidxs[ j ];
            
            for ( size_t  i = 0; i < n; ++i )
            {
                const idx_t  idx0 = rowidxs[ i ];
                double       value;

                if ( idx0 == idx1 ) 
                    value = -1.5*_h*_h + _h*_h*std::log(_h);
                else
                {
                    const double dist = _h * ( std::abs( double( idx0 - idx1 ) ) - 1.0 );
                    const double t1   = dist+1.0*_h;
                    const double t2   = dist+2.0*_h;
            
                    value = ( - 1.5*_h*_h + 0.5*t2*t2*std::log(t2) - t1*t1*std::log(t1) );
            
                    if ( std::abs(dist) > 1e-8 )
                        value += 0.5*dist*dist*std::log(dist);
                }
        
                matrix[ j*n + i ] = real_t(-value);
            }// for
        }// for
    }
    using TCoeffFn< real_t >::eval;

    //
    // return format of matrix, e.g. symmetric or hermitian
    //
    virtual matform_t  matrix_format  () const { return symmetric; }
    
};

///////////////////////////////////////////////////////////////////////////////
//
// DAG approach
//

//
// description for a memory block with base address and block indexset
//
struct memblk_t
{
    const void *    base;
    TBlockIndexSet  is;

    memblk_t ()
            : base( nullptr )
            , is( TIndexSet(), TIndexSet() )
    {}

    memblk_t ( const void *            abase,
               const TBlockIndexSet &  ais )
            : base( abase )
            , is( ais )
    {}
};

// return memblk struct for given matrix
memblk_t
memblk ( const B::Matrix< real_t > &  M )
{
    return memblk_t( M.data(), bis( is( 0, M.nrows()-1 ), is( 0, M.ncols()-1 ) ) );
}

memblk_t
memblk ( const B::Matrix< real_t > *  M )
{
    return memblk_t( M->data(), bis( is( 0, M->nrows()-1 ), is( 0, M->ncols()-1 ) ) );
}

memblk_t
memblk ( const TMatrix * M )
{
    return memblk_t( M, M->block_is() );
}

using  mem_blk_list_t = vector< memblk_t >;

//
// intersection tests for memory blocks
//
bool
is_intersecting ( const mem_blk_list_t &  blks0,
                  const mem_blk_list_t &  blks1 )
{
    for ( auto &  blk0 : blks0 )
        for ( auto &  blk1 : blks1 )
            if (( blk0.base == blk1.base ) && is_intersecting( blk0.is, blk1.is ))
                return true;

    return false;
}

struct Node;

using  node_list_t = list< Node * >;

//
// base class for all nodes in DAG
//
struct Node
{
    // dependencies
    node_list_t       in;
    node_list_t       out;
    int               dep_cnt;

    mem_blk_list_t    in_blk_deps;
    mem_blk_list_t    out_blk_deps;

    vector< Node * >  sub;
    
    Node ()
            : dep_cnt( 0 )
    {}
    
    virtual ~Node () {}

    // per node initialization
    void
    init ()
    {
        in_blk_deps  = in_mem_blks_();
        out_blk_deps = out_mem_blks_();
    }
    
    // return true if node is refined
    bool  is_refined () const { return ! sub.empty(); }

    // add node <t> as dependency (in-dependency)
    void  add_dep ( Node * t )
    {
        in.push_back( t );
        t->out.push_back( this );
    }

    // return local list of block index sets for input dependencies
    const mem_blk_list_t &  in_mem_blks  () const { return in_blk_deps; }
    
    // return local list of block index sets for output dependencies
    const mem_blk_list_t &  out_mem_blks () const { return out_blk_deps; }

    // split node into subnodes and update dependencies
    // if retval is empty, no refinement was done
    void
    refine ()
    {
        // cout << "refine( " << this->to_string() << " )" << endl;

        refine_();

        // if ( ! sub.empty() )
        // {
        //     cout << "    subnodes:" << endl;
            
        //     for ( auto  node : sub )
        //         cout << "        " << node->to_string() << endl;
        // }// if
    }

    //
    // refine dependencies for sub nodes
    //
    void
    refine_deps ()
    {
        // cout << "refine_deps( " << this->to_string() << " )" << endl;
    
        if ( sub.size() == 0 )
            return;
        
        for ( auto  dep : in )
        {
            if ( dep->is_refined() )
            {
                for ( auto  dep_sub : dep->sub )
                {
                    for ( auto  node : sub )
                    {
                        if ( is_intersecting( node->in_mem_blks(), dep_sub->out_mem_blks() ) )
                            node->in.push_back( dep_sub );
                    }// for
                }// for
            }// if
            else
            {
                for ( auto  node : sub )
                {
                    if ( is_intersecting( node->in_mem_blks(), dep->out_mem_blks() ) )
                        node->in.push_back( dep );
                }// for
            }// else
        }// for
        
        for ( auto  dep : out )
        {
            if ( dep->is_refined() )
            {
                for ( auto  dep_sub : dep->sub )
                {
                    for ( auto  node : sub )
                    {
                        if ( is_intersecting( node->out_mem_blks(), dep_sub->in_mem_blks() ) )
                            node->out.push_back( dep_sub );
                    }// for
                }// for
            }// if
            else
            {
                for ( auto  node : sub )
                {
                    if ( is_intersecting( node->out_mem_blks(), dep->in_mem_blks() ) )
                        node->out.push_back( dep );
                }// for
            }// else
        }// for
    }
    
    //
    // check local dependencies for refinement
    //
    bool
    check_deps ()
    {
        // cout << "check_deps( " << this->to_string() << " )" << endl;

        bool  changed = false;

        {
            node_list_t  new_in;
            auto         dep = in.begin();

            while ( dep != in.end() )
            {
                if ( (*dep)->is_refined() )
                {
                    changed = true;
                
                    // insert dependencies for subnodes (intersection test notwendig???)
                    for ( auto  dep_sub : (*dep)->sub )
                        new_in.push_back( dep_sub );

                    // remove previous dependency
                    dep = in.erase( dep );
                }// if
                else
                    ++dep;
            }// for

            in.splice( in.end(), new_in );
        }

        {
            node_list_t  new_out;
            auto         dep = out.begin();
            
            while ( dep != out.end() )
            {
                if ( (*dep)->is_refined() )
                {
                    changed = true;
                
                    // insert dependencies for subnodes (intersection test notwendig???)
                    for ( auto  dep_sub : (*dep)->sub )
                        new_out.push_back( dep_sub );

                    // remove previous dependency
                    dep = out.erase( dep );
                }// if
                else
                    ++dep;
            }// for

            out.splice( out.end(), new_out );
        }

        return changed;
    }
    
    // return text version of node
    virtual std::string
    to_string ()
    {
        return "node";
    }

private:
    virtual const mem_blk_list_t  in_mem_blks_ () const
    {
        return mem_blk_list_t();
    }
    
    // return list of block index sets for output dependencies
    virtual const mem_blk_list_t  out_mem_blks_ () const
    {
        return mem_blk_list_t();
    }

    virtual void
    refine_ ()
    {
        HERROR( ERR_NOT_IMPL, "", "" );
    }
};

struct LUNode : public Node
{
    TMatrix *  A;

    LUNode ( TMatrix *  aA )
            : A( aA )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "lu( %d )", A->id() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { memblk( A ) }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { memblk( A ) }; }
};

struct SolveLNode : public Node
{
    const TMatrix *      L;
    const memblk_t       mb_U;
    B::Matrix< real_t >  U;
    string               id;

    SolveLNode ( const TMatrix *         aL,
                 const memblk_t          amb_U,
                 B::Matrix< real_t > &   aU,
                 const string &          aid )
            : L( aL )
            , mb_U( amb_U )
            , U( aU )
            , id( aid )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "solve_L( %d, %s )", L->id(), id.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { memblk( L ), mb_U }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_U }; }
};
    
struct SolveUNode : public Node
{
    const TMatrix *      U;
    const memblk_t       mb_V;
    B::Matrix< real_t >  V;
    string               id;

    SolveUNode ( const TMatrix *         aU,
                 const memblk_t          amb_V,
                 B::Matrix< real_t > &   aV,
                 const string &          aid )
            : U( aU )
            , mb_V( amb_V )
            , V( aV )
            , id( aid )
    { init(); }

    virtual std::string  to_string  () { return HLIB::to_string( "solve_U( %d, %s )", U->id(), id.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { memblk( U ), mb_V }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_V }; }
};

struct LRUpdateNode : public Node
{
    const memblk_t             mb_U;
    const B::Matrix< real_t >  U;
    const string               id_U;
    const memblk_t             mb_T;
    const B::Matrix< real_t >  T;
    const memblk_t             mb_V;
    const B::Matrix< real_t >  V;
    const string               id_V;
    TMatrix *                  A;

    LRUpdateNode ( const memblk_t               amb_U,
                   const B::Matrix< real_t > &  aU,
                   const string                 aid_U,
                   const memblk_t               amb_T,
                   const B::Matrix< real_t > &  aT,
                   const memblk_t               amb_V,
                   const B::Matrix< real_t > &  aV,
                   const string                 aid_V,
                   TMatrix *                    aA )
            : mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
            , mb_T( amb_T )
            , T( aT )
            , mb_V( amb_V )
            , V( aV )
            , id_V( aid_V )
            , A( aA )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "LRupdate( %d, %s, %s )",
                                                                 A->id(), id_U.c_str(), id_V.c_str() ); }
    
private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_U, mb_T, mb_V }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { memblk( A ) }; }
};

struct TSAddNode : public Node
{
    const memblk_t             mb_U;
    const B::Matrix< real_t >  U;
    const string               id_U;
    const memblk_t             mb_T;
    const B::Matrix< real_t >  T;
    const memblk_t             mb_V;
    B::Matrix< real_t >        V;
    const string               id_V;

    TSAddNode ( const memblk_t               amb_U,
                 const B::Matrix< real_t > &  aU,
                 const string                 aid_U,
                 const memblk_t               amb_T,
                 const B::Matrix< real_t > &  aT,
                 const memblk_t               amb_V,
                 B::Matrix< real_t > &        aV,
                 const string                 aid_V )
            : mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
            , mb_T( amb_T )
            , T( aT )
            , mb_V( amb_V )
            , V( aV )
            , id_V( aid_V )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "TSadd( %s, %s )", id_U.c_str(), id_V.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_U, mb_T }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_V }; }
};

struct CompTNode : public Node
{
    const memblk_t             mb_V;
    const B::Matrix< real_t >  V;
    const string               id_V;
    const memblk_t             mb_U;
    const B::Matrix< real_t >  U;
    const string               id_U;
    const memblk_t             mb_T;
    B::Matrix< real_t >        T;

    CompTNode ( const memblk_t               amb_V,
                 const B::Matrix< real_t > &  aV,
                 const string                 aid_V,
                 const memblk_t               amb_U,
                 const B::Matrix< real_t > &  aU,
                 const string                 aid_U,
                 const memblk_t               amb_T,
                 B::Matrix< real_t > &        aT )
            : mb_V( amb_V )
            , V( aV )
            , id_V( aid_V )
            , mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
            , mb_T( amb_T )
            , T( aT )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "compT( %s, %s )", id_V.c_str(), id_U.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_V, mb_U }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_T }; }
};


struct AddNode : public Node
{
    const memblk_t             mb_T0;
    const B::Matrix< real_t >  T0;
    const string               id_T0;
    const memblk_t             mb_T1;
    const B::Matrix< real_t >  T1;
    const string               id_T1;
    const memblk_t             mb_T;
    B::Matrix< real_t >        T;

    AddNode ( const memblk_t               amb_T0,
              const B::Matrix< real_t > &  aT0,
              const string                 aid_T0,
              const memblk_t               amb_T1,
              const B::Matrix< real_t > &  aT1,
              const string                 aid_T1,
              const memblk_t               amb_T,
              B::Matrix< real_t > &        aT )
            : mb_T0( amb_T0 )
            , T0( aT0 )
            , id_T0( aid_T0 )
            , mb_T1( amb_T1 )
            , T1( aT1 )
            , id_T1( aid_T1 )
            , mb_T( amb_T )
            , T( aT )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "add( %s, %s )", id_T0.c_str(), id_T1.c_str() ); }

private:
    virtual void                  refine_       () {}
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_T0, mb_T1 }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_T }; }
};


struct TruncateNode : public Node
{
    const real_t               alpha;
    const memblk_t             mb_U;
    const B::Matrix< real_t >  U;
    const string               id_U;
    const memblk_t             mb_T;
    const B::Matrix< real_t >  T;
    const memblk_t             mb_V;
    const B::Matrix< real_t >  V;
    const string               id_V;
    TRkMatrix *                R;

    TruncateNode ( const real_t                 aalpha,
                    const memblk_t               amb_U,
                    const B::Matrix< real_t > &  aU,
                    const string                 aid_U,
                    const memblk_t               amb_T,
                    const B::Matrix< real_t > &  aT,
                    const memblk_t               amb_V,
                    const B::Matrix< real_t > &  aV,
                    const string                 aid_V,
                    TRkMatrix *                  aR )
            : alpha( aalpha )
            , mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
            , mb_T( amb_T )
            , T( aT )
            , mb_V( amb_V )
            , V( aV )
            , id_V( aid_V )
            , R( aR )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "truncate( %d, %s, %s )",
                                                                 R->id(), id_U.c_str(), id_V.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_U, mb_T, mb_V }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { memblk( R->blas_rmat_A() ),
                                                              memblk( R->blas_rmat_B() ) }; }
};

struct SVDNode : public Node
{
    const memblk_t             mb_RU;
    const B::Matrix< real_t >  RU;
    const memblk_t             mb_RV;
    const B::Matrix< real_t >  RV;
    const memblk_t             mb_U;
    B::Matrix< real_t >        U;
    const memblk_t             mb_V;
    B::Matrix< real_t >        V;

    SVDNode ( const memblk_t               amb_RU,
              const B::Matrix< real_t > &  aRU,
              const memblk_t               amb_RV,
              const B::Matrix< real_t > &  aRV,
              const memblk_t               amb_U,
              B::Matrix< real_t > &        aU,
              const memblk_t               amb_V,
              B::Matrix< real_t > &        aV )
            : mb_RU( amb_RU )
            , RU( aRU )
            , mb_RV( amb_RV )
            , RV( aRV )
            , mb_U( amb_U )
            , U( aU )
            , mb_V( amb_V )
            , V( aV )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "svd( %d, %d )", RU.nrows(), RV.nrows() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_RU, mb_RV }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_U, mb_V }; }
};

struct TSQRNode : public Node
{
    const real_t               alpha;
    const memblk_t             mb_X;
    const B::Matrix< real_t >  X;
    const string               id_X;
    const memblk_t             mb_T;
    const B::Matrix< real_t >  T;
    const memblk_t             mb_U;
    const B::Matrix< real_t >  U;
    const string               id_U;
    const memblk_t             mb_Q;
    B::Matrix< real_t >        Q;
    const memblk_t             mb_R;
    B::Matrix< real_t >        R;

    TSQRNode ( const real_t                 aalpha,
               const memblk_t               amb_X,
               const B::Matrix< real_t > &  aX,
               const string                 aid_X,
               const memblk_t               amb_T,
               const B::Matrix< real_t > &  aT,
               const memblk_t               amb_U,
               const B::Matrix< real_t > &  aU,
               const string                 aid_U,
               const memblk_t               amb_Q,
               B::Matrix< real_t >          aQ,
               const memblk_t               amb_R,
               B::Matrix< real_t >          aR )
            : alpha( aalpha )
            , mb_X( amb_X )
            , X( aX )
            , id_X( aid_X )
            , mb_T( amb_T )
            , T( aT )
            , mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
            , mb_Q( amb_Q )
            , Q( aQ )
            , mb_R( amb_R )
            , R( aR )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "tsqr( %s, %s )", id_X.c_str(), id_U.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_X, mb_T, mb_U }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_Q, mb_R }; }
};

struct QRNode : public Node
{
    const real_t               alpha;
    const memblk_t             mb_X;
    const B::Matrix< real_t >  X;
    const string               id_X;
    const memblk_t             mb_T;
    const B::Matrix< real_t >  T;
    const memblk_t             mb_U;
    const B::Matrix< real_t >  U;
    const string               id_U;
    const memblk_t             mb_Q;
    B::Matrix< real_t >        Q;
    const memblk_t             mb_R;
    B::Matrix< real_t >        R;

    QRNode ( const real_t                 aalpha,
             const memblk_t               amb_X,
             const B::Matrix< real_t > &  aX,
             const string                 aid_X,
             const memblk_t               amb_T,
             const B::Matrix< real_t > &  aT,
             const memblk_t               amb_U,
             const B::Matrix< real_t > &  aU,
             const string                 aid_U,
             const memblk_t               amb_Q,
             B::Matrix< real_t >          aQ,
             const memblk_t               amb_R,
             B::Matrix< real_t >          aR )
            : alpha( aalpha )
            , mb_X( amb_X )
            , X( aX )
            , id_X( aid_X )
            , mb_T( amb_T )
            , T( aT )
            , mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
            , mb_Q( amb_Q )
            , Q( aQ )
            , mb_R( amb_R )
            , R( aR )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "qr( %s, %s )", id_X.c_str(), id_U.c_str() ); }
    
private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_X, mb_T, mb_U }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_Q, mb_R }; }
};

struct TSQRCombineNode : public Node
{
    const memblk_t             mb_R0;
    const B::Matrix< real_t >  R0;
    const string               id_R0;
    const memblk_t             mb_R1;
    const B::Matrix< real_t >  R1;
    const string               id_R1;
    const memblk_t             mb_Q;
    B::Matrix< real_t >        Q;
    const memblk_t             mb_R;
    B::Matrix< real_t >        R;

    TSQRCombineNode ( const memblk_t &           amb_R0,
                      const B::Matrix< real_t >  aR0,
                      const string               aid_R0,
                      const memblk_t &           amb_R1,
                      const B::Matrix< real_t >  aR1,
                      const string               aid_R1,
                      const memblk_t &           amb_Q,
                      B::Matrix< real_t > &      aQ,
                      const memblk_t &           amb_R,
                      B::Matrix< real_t > &      aR )
            : mb_R0( amb_R0 )
            , R0( aR0 )
            , id_R0( aid_R0 )
            , mb_R1( amb_R1 )
            , R1( aR1 )
            , id_R1( aid_R1 )
            , mb_Q( amb_Q )
            , Q( aQ )
            , mb_R( amb_R )
            , R( aR )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "tsqr_combine( %s, %s )",
                                                                 id_R0.c_str(), id_R1.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_R0, mb_R1 }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_Q, mb_R }; }
};

struct MulTNode : public Node
{
    const memblk_t             mb_X;
    const B::Matrix< real_t >  X;
    const string               id_X;
    const memblk_t             mb_T;
    const B::Matrix< real_t >  T;
    const memblk_t             mb_U;
    B::Matrix< real_t >        U;
    const string               id_U;

    MulTNode ( const memblk_t               amb_X,
               const B::Matrix< real_t > &  aX,
               const string                 aid_X,
               const memblk_t               amb_T,
               const B::Matrix< real_t > &  aT,
               const memblk_t               amb_U,
               B::Matrix< real_t > &        aU,
               const string                 aid_U )
            : mb_X( amb_X )
            , X( aX )
            , id_X( aid_X )
            , mb_T( amb_T )
            , T( aT )
            , mb_U( amb_U )
            , U( aU )
            , id_U( aid_U )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "mulT( %s, %s )", id_X.c_str(), id_U.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_X, mb_T }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_U }; }
};


struct GEMMNode : public Node
{
    const real_t               alpha;
    
    const memblk_t             mb_A;
    const matop_t              op_A;
    const B::Matrix< real_t >  A;
    const string               id_A;
    
    const memblk_t             mb_B;
    const matop_t              op_B;
    const B::Matrix< real_t >  B;
    const string               id_B;
    
    const real_t               beta;
    const memblk_t             mb_C;
    B::Matrix< real_t >        C;
    const string               id_C;

    GEMMNode ( const real_t                 aalpha,
               const memblk_t               amb_A,
               const matop_t                aop_A,
               const B::Matrix< real_t > &  aA,
               const string                 aid_A,
               const memblk_t               amb_B,
               const matop_t                aop_B,
               const B::Matrix< real_t > &  aB,
               const string                 aid_B,
               const real_t                 abeta,
               const memblk_t               amb_C,
               B::Matrix< real_t > &        aC,
               const string                 aid_C )
            : alpha( aalpha )
            , mb_A( amb_A )
            , op_A( aop_A )
            , A( aA )
            , id_A( aid_A )
            , mb_B( amb_B )
            , op_B( aop_B )
            , B( aB )
            , id_B( aid_B )
            , beta( abeta )
            , mb_C( amb_C )
            , C( aC )
            , id_C( aid_C )
    { init(); }
    
    virtual std::string  to_string  () { return HLIB::to_string( "gemm( %s, %s, %s )",
                                                                 id_A.c_str(), id_B.c_str(), id_C.c_str() ); }

private:
    virtual void                  refine_       ();
    virtual const mem_blk_list_t  in_mem_blks_  () const { return { mb_A, mb_B }; }
    virtual const mem_blk_list_t  out_mem_blks_ () const { return { mb_C }; }
};

//
// recursive functions
//

void lu          ( TMatrix *                    A );
void solve_lower ( const TMatrix *              L,
                   B::Matrix< real_t > &        U );
void solve_upper ( const TMatrix *              U,
                   B::Matrix< real_t > &        V );
void LRupdate    ( const B::Matrix< real_t > &  U,
                   const B::Matrix< real_t > &  T,
                   const B::Matrix< real_t > &  V,
                   TMatrix *                    A );
void TSadd       ( const B::Matrix< real_t > &  U,
                   const B::Matrix< real_t > &  T,
                   B::Matrix< real_t > &        V );
void compT       ( const B::Matrix< real_t > &  V,
                   const B::Matrix< real_t > &  U,
                   B::Matrix< real_t > &        T );
void truncate    ( const real_t                 alpha,
                   const B::Matrix< real_t > &  U,
                   const B::Matrix< real_t > &  T,
                   const B::Matrix< real_t > &  V,
                   TRkMatrix *                  R );
void tsqr        ( const real_t                 alpha,
                   const B::Matrix< real_t > &  U,
                   const B::Matrix< real_t > &  T,
                   const B::Matrix< real_t > &  V,
                   B::Matrix< real_t > &        Q,
                   B::Matrix< real_t > &        R );

void mulT        ( const B::Matrix< real_t > &  X,
                   const B::Matrix< real_t > &  T,
                   B::Matrix< real_t > &        U );

///////////////////////////////////////////////////////////////////////////////////////
//
// LUNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
lu ( TMatrix *  A )
{
    // cout << "lu( " << A->id() << " )" << endl;
    
    if ( is_blocked( A ) )
    {
        auto  B   = ptrcast( A, TBlockMatrix );
        auto  A00 = B->block( 0, 0 ); 
        auto  A01 = ptrcast( B->block( 0, 1 ), TRkMatrix ); 
        auto  A10 = ptrcast( B->block( 1, 0 ), TRkMatrix );
        auto  A11 = B->block( 1, 1 );
        auto  T   = B::Matrix< real_t >( k, k );

        lu( A00 );
        tbb::parallel_invoke( [A00,A01] () { solve_lower( A00, A01->blas_rmat_A() ); },
                              [A00,A10] () { solve_upper( A00, A10->blas_rmat_B() ); } );
        compT( A10->blas_rmat_B(), A01->blas_rmat_A(), T );
        LRupdate( A10->blas_rmat_A(), T, A01->blas_rmat_B(), A11 );
        lu( A11 );
    }// if
    else
    {
        invert( A, acc_exact );
    }// else
}

void
LUNode::refine_ ()
{
    if ( is_blocked( A ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  B   = ptrcast( A, TBlockMatrix );
        auto  A00 = B->block( 0, 0 ); 
        auto  A01 = ptrcast( B->block( 0, 1 ), TRkMatrix ); 
        auto  A10 = ptrcast( B->block( 1, 0 ), TRkMatrix ); 
        auto  A11 = B->block( 1, 1 );

        auto T           = new B::Matrix< real_t >( k, k );  // wohin damit?
        auto lu_00       = new LUNode( A00 );
        auto solve_01    = new SolveLNode( A00, memblk( A01->blas_rmat_A() ), A01->blas_rmat_A(),
                                           HLIB::to_string( "%dU", A01->id() ) );
        auto solve_10    = new SolveUNode( A00, memblk( A10->blas_rmat_B() ), A10->blas_rmat_B(),
                                           HLIB::to_string( "%dV", A10->id() ) );
        auto compT_10_01 = new CompTNode( memblk( A10->blas_rmat_B() ), A10->blas_rmat_B(), HLIB::to_string( "%dV", A10->id() ),
                                          memblk( A01->blas_rmat_A() ), A01->blas_rmat_A(), HLIB::to_string( "%dU", A01->id() ),
                                          memblk( T ), *T );
        auto LRupdate_11 = new LRUpdateNode( memblk( A10->blas_rmat_A() ), A10->blas_rmat_A(), HLIB::to_string( "%dU", A10->id() ),
                                             memblk( T ),                  *T,
                                             memblk( A01->blas_rmat_B() ), A01->blas_rmat_B(), HLIB::to_string( "%dV", A01->id() ),
                                             A11 );
        auto lu_11       = new LUNode( A11 );

        sub.resize( 6 );
        sub[0] = lu_00;
        sub[1] = solve_01;
        sub[2] = solve_10;
        sub[3] = compT_10_01;
        sub[4] = LRupdate_11;
        sub[5] = lu_11;

        //
        // set internal dependencies
        //

        solve_01->add_dep( lu_00 );
        solve_10->add_dep( lu_00 );
        compT_10_01->add_dep( solve_01 );
        compT_10_01->add_dep( solve_10 );
        LRupdate_11->add_dep( compT_10_01 );
        lu_11->add_dep( LRupdate_11 );
    }// if
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SolveLNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
solve_lower ( const TMatrix *        L,
              B::Matrix< real_t > &  U )
{
    // cout << "solve_lower( " << L->id() << ", " << U.nrows() << " )" << endl;
    
    if ( is_blocked( L ) )
    {
        auto  B   = cptrcast( L, TBlockMatrix );
        auto  L00 = B->block( 0, 0 ); 
        auto  L10 = cptrcast( B->block( 1, 0 ), TRkMatrix );
        auto  L11 = B->block( 1, 1 );
        auto  U0  = B::Matrix< real_t >( U, L00->col_is() - L->col_ofs(), B::Range::all );
        auto  U1  = B::Matrix< real_t >( U, L11->col_is() - L->col_ofs(), B::Range::all );
        auto  T   = B::Matrix< real_t >( k, k );

        solve_lower( L00, U0 );
        compT( L10->blas_rmat_B(), U0, T );
        TSadd( L10->blas_rmat_A(), T, U1 );
        solve_lower( L11, U1 );
    }// if
    else
    {
        //
        // L is unit diagonal, hence nothing to do
        //
        
        // auto                 D  = ptrcast( L, TDenseMatrix );
        // B::Matrix< real_t >  UC( U, copy_value );

        // B::prod( real_t(1), D->blas_rmat(), UC, real_t(0), U );
    }// else
}

void
SolveLNode::refine_ ()
{
    if ( is_blocked( L ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BL    = cptrcast( L, TBlockMatrix );
        auto  L00   = BL->block( 0, 0 ); 
        auto  L10   = cptrcast( BL->block( 1, 0 ), TRkMatrix ); 
        auto  L11   = BL->block( 1, 1 );
        auto  U0    = B::Matrix< real_t >( U, L00->row_is() - L->row_ofs(), B::Range::all );
        auto  mb_U0 = memblk_t( mb_U.base, bis( L00->row_is() - L->row_ofs() + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  U1    = B::Matrix< real_t >( U, L11->row_is() - L->row_ofs(), B::Range::all );
        auto  mb_U1 = memblk_t( mb_U.base, bis( L11->row_is() - L->row_ofs() + mb_U.is.row_is().first(), mb_U.is.col_is() ) );

        auto T        = new B::Matrix< real_t >( k, k );  // wohin damit?
        auto solve_0  = new SolveLNode( L00, mb_U0, U0, id + "0" );
        auto compT_10 = new CompTNode( memblk( L10->blas_rmat_B() ), L10->blas_rmat_B(), HLIB::to_string( "%dV", L10->id() ),
                                       mb_U0,       U0, id + "0",
                                       memblk( T ), *T );
        auto TSadd_1  = new TSAddNode( memblk( L10->blas_rmat_A() ), L10->blas_rmat_A(), HLIB::to_string( "%dU", L10->id() ),
                                       memblk( T ), *T,
                                       mb_U1,       U1, id + "1" );
        auto solve_1  = new SolveLNode( L11, mb_U1, U1, id + "1" );

        sub.resize( 4 );
        sub[0] = solve_0;
        sub[1] = compT_10;
        sub[2] = TSadd_1;
        sub[3] = solve_1;

        //
        // set internal dependencies
        //

        compT_10->add_dep( solve_0 );
        TSadd_1->add_dep( compT_10 );
        solve_1->add_dep( TSadd_1 );
    }// if
    else
    {
        // gemm??
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SolveUNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
solve_upper ( const TMatrix *        U,
              B::Matrix< real_t > &  V )
{
    // cout << "solve_upper( " << U->id() << ", " << V.nrows() << " )" << endl;

    if ( is_blocked( U ) )
    {
        auto  B   = cptrcast( U, TBlockMatrix );
        auto  U00 = B->block( 0, 0 ); 
        auto  U01 = cptrcast( B->block( 0, 1 ), TRkMatrix );
        auto  U11 = B->block( 1, 1 );
        auto  V0  = B::Matrix< real_t >( V, U00->row_is() - U->row_ofs(), B::Range::all );
        auto  V1  = B::Matrix< real_t >( V, U11->row_is() - U->row_ofs(), B::Range::all );
        auto  T   = B::Matrix< real_t >( k, k );

        solve_upper( U00, V0 );
        compT( U01->blas_rmat_A(), V0, T );
        TSadd( U01->blas_rmat_B(), T, V1 );
        solve_upper( U11, V1 );
    }// if
    else
    {
        auto                 D  = cptrcast( U, TDenseMatrix );
        B::Matrix< real_t >  VC( V, copy_value );

        B::prod( real_t(1), B::transposed( D->blas_rmat() ), VC, real_t(0), V );
    }// else
}

void
SolveUNode::refine_ ()
{
    if ( is_blocked( U ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BU    = cptrcast( U, TBlockMatrix );
        auto  U00   = BU->block( 0, 0 ); 
        auto  U01   = cptrcast( BU->block( 0, 1 ), TRkMatrix ); 
        auto  U11   = BU->block( 1, 1 );
        auto  V0    = B::Matrix< real_t >( V, U00->row_is() - U->row_ofs(), B::Range::all );
        auto  mb_V0 = memblk_t( mb_V.base, bis( U00->row_is() - U->row_ofs() + mb_V.is.row_is().first(), mb_V.is.col_is() ) );
        auto  V1    = B::Matrix< real_t >( V, U11->row_is() - U->row_ofs(), B::Range::all );
        auto  mb_V1 = memblk_t( mb_V.base, bis( U11->row_is() - U->row_ofs() + mb_V.is.row_is().first(), mb_V.is.col_is() ) );

        auto T        = new B::Matrix< real_t >( k, k );  // wohin damit?
        auto solve_0  = new SolveUNode( U00, mb_V0, V0, id + "0" );
        auto compT_01 = new CompTNode( memblk( U01->blas_rmat_A() ), U01->blas_rmat_A(), HLIB::to_string( "%dU", U01->id() ),
                                        mb_V0,       V0, id + "0",
                                        memblk( T ), *T );
        auto TSadd_1  = new TSAddNode( memblk( U01->blas_rmat_B() ), U01->blas_rmat_B(), HLIB::to_string( "%dV", U01->id() ),
                                        memblk( T ), *T,
                                        mb_V1,       V1, id + "1" );
        auto solve_1  = new SolveUNode( U11, mb_V1, V1, id + "1" );

        sub.resize( 4 );
        sub[0] = solve_0;
        sub[1] = compT_01;
        sub[2] = TSadd_1;
        sub[3] = solve_1;

        //
        // set internal dependencies
        //

        compT_01->add_dep( solve_0 );
        TSadd_1->add_dep( compT_01 );
        solve_1->add_dep( TSadd_1 );
    }// if
    else
    {
        // gemm??
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// LRUpdateNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
LRupdate ( const B::Matrix< real_t > &  U,
           const B::Matrix< real_t > &  T,
           const B::Matrix< real_t > &  V,
           TMatrix *                    A )
{
    // cout << "LRupdate( " << A->id() << " )" << endl;

    if ( is_blocked( A ) )
    {
        auto  BA  = ptrcast( A, TBlockMatrix );
        auto  A00 = BA->block( 0, 0 ); 
        auto  A01 = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10 = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11 = BA->block( 1, 1 );
        auto  U0  = B::Matrix< real_t >( U, A00->row_is() - A->row_ofs(), B::Range::all );
        auto  U1  = B::Matrix< real_t >( U, A11->row_is() - A->row_ofs(), B::Range::all );
        auto  V0  = B::Matrix< real_t >( V, A00->col_is() - A->col_ofs(), B::Range::all );
        auto  V1  = B::Matrix< real_t >( V, A11->col_is() - A->col_ofs(), B::Range::all );

        LRupdate( U0, T, V0, A00 );
        tbb::parallel_invoke( [&,A01] () { truncate( real_t(-1), U0, T, V1, A01 ); },
                              [&,A10] () { truncate( real_t(-1), U1, T, V0, A10 ); } );
        LRupdate( U1, T, V1, A11 );
    }// if
    else
    {
        auto  D = ptrcast( A, TDenseMatrix );
        auto  W = B::prod( real_t(1), U, T );

        B::prod( real_t(-1), W, B::transposed( V ), real_t(1), D->blas_rmat() );
    }// else
}

void
LRUpdateNode::refine_ ()
{
    if ( is_blocked( A ) )
    {
        //
        // generate sub nodes assuming 2x2 block structure
        //

        auto  BA    = ptrcast( A, TBlockMatrix );
        auto  A00   = BA->block( 0, 0 );
        auto  A01   = ptrcast( BA->block( 0, 1 ), TRkMatrix );
        auto  A10   = ptrcast( BA->block( 1, 0 ), TRkMatrix );
        auto  A11   = BA->block( 1, 1 );
        auto  U0    = B::Matrix< real_t >( U, A00->row_is() - A->row_ofs(), B::Range::all );
        auto  mb_U0 = memblk_t( mb_U.base, bis( A00->row_is() - A->row_ofs() + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  U1    = B::Matrix< real_t >( U, A11->row_is() - A->row_ofs(), B::Range::all );
        auto  mb_U1 = memblk_t( mb_U.base, bis( A11->row_is() - A->row_ofs() + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  V0    = B::Matrix< real_t >( V, A00->col_is() - A->col_ofs(), B::Range::all );
        auto  mb_V0 = memblk_t( mb_V.base, bis( A00->col_is() - A->col_ofs() + mb_V.is.row_is().first(), mb_V.is.col_is() ) );
        auto  V1    = B::Matrix< real_t >( V, A11->col_is() - A->col_ofs(), B::Range::all );
        auto  mb_V1 = memblk_t( mb_V.base, bis( A11->col_is() - A->col_ofs() + mb_V.is.row_is().first(), mb_V.is.col_is() ) );

        auto  LRupdate_00 = new LRUpdateNode( mb_U0, U0, id_U + "0", mb_T, T, mb_V0, V0, id_V + "0", A00 );
        auto  truncate_01 = new TruncateNode( real_t(-1), mb_U0, U0, id_U + "0", mb_T, T, mb_V1, V1, id_V + "1", A01 );
        auto  truncate_10 = new TruncateNode( real_t(-1), mb_U1, U1, id_U + "1", mb_T, T, mb_V0, V0, id_V + "0", A10 );
        auto  LRupdate_11 = new LRUpdateNode( mb_U1, U1, id_U + "1" , mb_T, T, mb_V1, V1, id_V + "1", A11 );

        sub.resize( 4 );
        sub[0] = LRupdate_00;
        sub[1] = truncate_01;
        sub[2] = truncate_10;
        sub[3] = LRupdate_11;
    }// if
    else if ( is_dense( A ) )
    {
        auto  D      = ptrcast( A, TDenseMatrix );
        auto  W      = new B::Matrix< real_t >( U.nrows(), T.ncols() );
        
        auto  gemm_W = new GEMMNode( real_t(1),
                                      mb_U, apply_normal, U, id_U,
                                      mb_T, apply_normal, T, "T",
                                      real_t(0),
                                      memblk( W ), *W, id_U + "·T" );
        auto  gemm_A = new GEMMNode( real_t(-1),
                                      memblk(W), apply_normal, *W, id_U + "·T",
                                      mb_V,      apply_normal, V, id_V,
                                      real_t(1),
                                      memblk( D ), D->blas_rmat(), HLIB::to_string( "%d", D->id() ) );

        sub.resize( 2 );
        sub[0] = gemm_W;
        sub[1] = gemm_A;
        
        gemm_A->add_dep( gemm_W );
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// TSAddNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
TSadd ( const B::Matrix< real_t > &  U,
        const B::Matrix< real_t > &  T,
        B::Matrix< real_t > &        V )
{
    // cout << "TSadd( " << U.nrows() << " )" << endl;

    if ( U.nrows() != V.nrows() )
        HERROR( ERR_MAT_SIZE, "TSadd", "U and V differ in row size" );
    
    if ( U.ncols() != T.nrows() )
        HERROR( ERR_MAT_SIZE, "TSadd", "U and T are incompatible" );
    
    if ( T.ncols() != V.ncols() )
        HERROR( ERR_MAT_SIZE, "TSadd", "T and V differ in column size" );
    
    if ( V.nrows() > ntile )
    {
        auto  mid = V.nrows() / 2;
        auto  U0  = B::Matrix< real_t >( U, B::Range( 0, mid-1 ),         B::Range::all );
        auto  U1  = B::Matrix< real_t >( U, B::Range( mid, U.nrows()-1 ), B::Range::all );
        auto  V0  = B::Matrix< real_t >( V, B::Range( 0, mid-1 ),         B::Range::all );
        auto  V1  = B::Matrix< real_t >( V, B::Range( mid, V.nrows()-1 ), B::Range::all );

        tbb::parallel_invoke( [&] () { TSadd( U0, T, V0 ); },
                              [&] () { TSadd( U1, T, V1 ); } );
    }// if
    else
    {
        B::prod( real_t(-1), U, T, real_t(1), V );
    }// else
}

void
TSAddNode::refine_ ()
{
    if ( V.nrows() > ntile )
    {
        auto  mid   = V.nrows() / 2;
        auto  rows0 = B::Range( 0, mid-1 );
        auto  rows1 = B::Range( mid, V.nrows()-1 );
        auto  U0    = B::Matrix< real_t >( U, rows0, B::Range::all );
        auto  mb_U0 = memblk_t( mb_U.base, bis( rows0 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  U1    = B::Matrix< real_t >( U, rows1, B::Range::all );
        auto  mb_U1 = memblk_t( mb_U.base, bis( rows1 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  V0    = B::Matrix< real_t >( V, rows0, B::Range::all );
        auto  mb_V0 = memblk_t( mb_V.base, bis( rows0 + mb_V.is.row_is().first(), mb_V.is.col_is() ) );
        auto  V1    = B::Matrix< real_t >( V, rows1, B::Range::all );
        auto  mb_V1 = memblk_t( mb_V.base, bis( rows1 + mb_V.is.row_is().first(), mb_V.is.col_is() ) );

        auto  TSadd_0 = new TSAddNode( mb_U0, U0, id_U + "0", mb_T, T, mb_V0, V0, id_V + "0" );
        auto  TSadd_1 = new TSAddNode( mb_U1, U1, id_U + "1", mb_T, T, mb_V1, V1, id_V + "1" );

        sub.resize( 2 );
        sub[0] = TSadd_0;
        sub[1] = TSadd_1;
    }// if
    else
    {
        auto  gemm = new GEMMNode( real_t(-1),
                                    mb_U, apply_normal, U, id_U,
                                    mb_T, apply_normal, T, "T",
                                    real_t(1),
                                    mb_V, V, id_V );

        sub.resize( 1 );
        sub[0] = gemm;
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// CompTNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
compT ( const B::Matrix< real_t > &  V,
        const B::Matrix< real_t > &  U,
        B::Matrix< real_t > &        T )
{
    // cout << "compT( " << V.nrows() << ", " << U.nrows() << " )" << endl;

    if ( V.nrows() != U.nrows() )
        HERROR( ERR_MAT_SIZE, "compT", "V and U differ in row size" );
    
    if ( U.nrows() > ntile )
    {
        auto  mid = V.nrows() / 2;
        auto  U0  = B::Matrix< real_t >( U, B::Range( 0, mid-1 ),         B::Range::all );
        auto  U1  = B::Matrix< real_t >( U, B::Range( mid, V.nrows()-1 ), B::Range::all );
        auto  V0  = B::Matrix< real_t >( V, B::Range( 0, mid-1 ),         B::Range::all );
        auto  V1  = B::Matrix< real_t >( V, B::Range( mid, V.nrows()-1 ), B::Range::all );
        auto  T0  = B::Matrix< real_t >( T.nrows(), T.ncols() );
        auto  T1  = B::Matrix< real_t >( T.nrows(), T.ncols() );

        compT( V0, U0, T0 );
        compT( V1, U1, T1 );

        B::copy( T0, T );
        B::add( real_t(1), T1, T );
    }// if
    else
    {
        B::prod( real_t(1), B::transposed(V), U, real_t(0), T );
    }// else
}

void
CompTNode::refine_ ()
{
    if ( U.nrows() > ntile )
    {
        auto  mid = V.nrows() / 2;
        auto  rows0 = B::Range( 0, mid-1 );
        auto  rows1 = B::Range( mid, V.nrows()-1 );
        auto  V0    = B::Matrix< real_t >( V, rows0, B::Range::all );
        auto  mb_V0 = memblk_t( mb_V.base, bis( rows0 + mb_V.is.row_is().first(), mb_V.is.col_is() ) );
        auto  V1    = B::Matrix< real_t >( V, rows1, B::Range::all );
        auto  mb_V1 = memblk_t( mb_V.base, bis( rows1 + mb_V.is.row_is().first(), mb_V.is.col_is() ) );
        auto  U0    = B::Matrix< real_t >( U, rows0, B::Range::all );
        auto  mb_U0 = memblk_t( mb_U.base, bis( rows0 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  U1    = B::Matrix< real_t >( U, rows1, B::Range::all );
        auto  mb_U1 = memblk_t( mb_U.base, bis( rows1 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  T0    = new B::Matrix< real_t >( T.nrows(), T.ncols() );
        auto  T1    = new B::Matrix< real_t >( T.nrows(), T.ncols() );

        auto  compT_0 = new CompTNode( mb_V0, V0, id_V + "0", mb_U0, U0, id_U + "0", memblk(T0), *T0 );
        auto  compT_1 = new CompTNode( mb_V1, V1, id_V + "1", mb_U1, U1, id_U + "1", memblk(T1), *T1 );
        auto  add01   = new AddNode(   memblk(T0), *T0, id_V + "0x" + id_U + "0",
                                       memblk(T1), *T1, id_V + "1x" + id_U + "1",
                                       mb_T, T );

        add01->add_dep( compT_0 );
        add01->add_dep( compT_1 );
        
        sub.resize( 3 );
        sub[0] = compT_0;
        sub[1] = compT_1;
        sub[2] = add01;
    }// if
    else
    {
        auto  gemm = new GEMMNode( real_t(1),
                                    mb_V, apply_transposed, V, id_V,
                                    mb_U, apply_normal,     U, id_U,
                                    real_t(0),
                                    mb_T, T, "T" );

        sub.resize( 1 );
        sub[0] = gemm;
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// TruncateNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
truncate ( const real_t                 alpha,
           const B::Matrix< real_t > &  U,
           const B::Matrix< real_t > &  T,
           const B::Matrix< real_t > &  V,
           TRkMatrix *                  R )
{
    // cout << "truncate( " << R->id() << ", " << U.nrows() << " )" << endl;
    
    if ( U.nrows() != R->rows() )
        HERROR( ERR_MAT_SIZE, "tsqr", "U and R differ in row size" );

    if ( U.ncols() != T.nrows() )
        HERROR( ERR_MAT_SIZE, "tsqr", "U and T are incompatible" );

    if ( V.nrows() != R->cols() )
        HERROR( ERR_MAT_SIZE, "tsqr", "V^T and R differ in column size" );

    if ( U.nrows() > ntile )
    {
        auto  UR = R->blas_rmat_A();
        auto  VR = R->blas_rmat_B();

        auto  QU = B::Matrix< real_t >( U.nrows(), 2*k );
        auto  QV = B::Matrix< real_t >( V.nrows(), 2*k );
        auto  I  = B::identity< real_t >( k );
        auto  RU = B::Matrix< real_t >( 2*k, 2*k );
        auto  RV = B::Matrix< real_t >( 2*k, 2*k );

        tbb::parallel_invoke( [&] () { tsqr( alpha,     U, T, UR, QU, RU ); },
                              [&] () { tsqr( real_t(1), V, I, VR, QV, RV ); } );
              
        auto  U_svd = B::prod( real_t(1), RU, B::transposed(RV) );
        auto  S_svd = B::Vector< real_t >( 2*k );
        auto  V_svd = B::Matrix< real_t >( 2*k, 2*k );

        B::svd( U_svd, S_svd, V_svd );

        auto  Uk = B::Matrix< real_t >( U_svd, B::Range::all, B::Range( 0, k-1 ) );
        auto  Vk = B::Matrix< real_t >( V_svd, B::Range::all, B::Range( 0, k-1 ) );
        
        B::prod_diag( Uk, S_svd, k );

        // B::prod( real_t(1), QU, Uk, real_t(0), UR );
        // B::prod( real_t(1), QV, Vk, real_t(0), VR );

        tbb::parallel_invoke( [&] () { mulT( QU, Uk, UR ); },
                              [&] () { mulT( QV, Vk, VR ); } );
    }// if
    else
    {
        auto  W = B::prod( alpha, U, T );
    
        R->add_rank( real_t(1), W, V, fixed_rank( k ) );
    }// else
}

void
TruncateNode::refine_ ()
{
    auto  UR = R->blas_rmat_A();
    auto  VR = R->blas_rmat_B();

    auto  QU = new B::Matrix< real_t >( U.nrows(), 2*k );
    auto  QV = new B::Matrix< real_t >( V.nrows(), 2*k );
    auto  I  = new B::Matrix< real_t >( B::identity< real_t >( k ), copy_value );
    auto  RU = new B::Matrix< real_t >( 2*k, 2*k );
    auto  RV = new B::Matrix< real_t >( 2*k, 2*k );

    auto  tsqr_U = new TSQRNode( alpha,
                                  mb_U, U, id_U,
                                  mb_T, T,
                                  memblk( UR ),  UR, HLIB::to_string( "%dU", R->id() ),
                                  memblk( QU ), *QU,
                                  memblk( RU ), *RU );
    auto  tsqr_V = new TSQRNode( real_t(1),
                                  mb_V, V, id_V,
                                  memblk( I  ), *I,
                                  memblk( VR ), VR, HLIB::to_string( "%dV", R->id() ),
                                  memblk( QV ), *QV, 
                                  memblk( RV ), *RV );

    auto  Uk = new B::Matrix< real_t >( 2*k, k );
    auto  Vk = new B::Matrix< real_t >( 2*k, k );

    auto  svd_R  = new SVDNode( memblk( RU ), *RU,
                                 memblk( RV ), *RV,
                                 memblk( Uk ), *Uk,
                                 memblk( Vk ), *Vk );

    auto  mulT_U = new MulTNode( memblk( QU ), *QU, HLIB::to_string( "%dQU", R->id() ),
                                  memblk( Uk ), *Uk,
                                  memblk( UR ), UR, HLIB::to_string( "%dU", R->id() ) );
    auto  mulT_V = new MulTNode( memblk( QV ), *QV, HLIB::to_string( "%dQV", R->id() ),
                                  memblk( Vk ), *Vk,
                                  memblk( VR ), VR, HLIB::to_string( "%dV", R->id() ) );

    sub.resize( 5 );
    sub[0] = tsqr_U;
    sub[1] = tsqr_V;
    sub[2] = svd_R;
    sub[3] = mulT_U;
    sub[4] = mulT_V;

    svd_R->add_dep( tsqr_U );
    svd_R->add_dep( tsqr_V );
    mulT_U->add_dep( tsqr_U ); mulT_U->add_dep( svd_R );
    mulT_V->add_dep( tsqr_V ); mulT_V->add_dep( svd_R );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// SVDNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
SVDNode::refine_ ()
{
    // auto  U_svd = B::prod( real_t(1), RU, B::transposed(RV) );
    // auto  S_svd = B::Vector< real_t >( 2*k );
    // auto  V_svd = B::Matrix< real_t >( 2*k, 2*k );

    // B::svd( U_svd, S_svd, V_svd );

    // auto  Uk = B::Matrix< real_t >( U_svd, B::Range::all, B::Range( 0, k-1 ) );
    // auto  Vk = B::Matrix< real_t >( V_svd, B::Range::all, B::Range( 0, k-1 ) );
        
    // B::prod_diag( Uk, S_svd, k );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// TSQRNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
tsqr ( const real_t                 alpha,
       const B::Matrix< real_t > &  X,
       const B::Matrix< real_t > &  T,
       const B::Matrix< real_t > &  U,
       B::Matrix< real_t > &        Q,
       B::Matrix< real_t > &        R )
{
    // cout << "tsqr( " << U.nrows() << " )" << endl;

    const size_t  rank_XT = T.ncols();
    const size_t  rank_U  = U.ncols();
    const size_t  rank    = rank_XT + rank_U;

    if ( X.nrows() != U.nrows() )
        HERROR( ERR_MAT_SIZE, "tsqr", "X and U differ in row size" );

    if ( X.ncols() != T.nrows() )
        HERROR( ERR_MAT_SIZE, "tsqr", "X and U are incompatible" );
    
    if ( X.nrows() > ntile )
    {
        auto  mid   = X.nrows() / 2;
        auto  rows0 = B::Range( 0, mid-1 );
        auto  rows1 = B::Range( mid, X.nrows()-1 );
        auto  X0    = B::Matrix< real_t >( X, rows0, B::Range::all );
        auto  X1    = B::Matrix< real_t >( X, rows1, B::Range::all );
        auto  U0    = B::Matrix< real_t >( U, rows0, B::Range::all );
        auto  U1    = B::Matrix< real_t >( U, rows1, B::Range::all );

        auto  Q0    = B::Matrix< real_t >( X0.nrows(), rank );
        auto  Q1    = B::Matrix< real_t >( X1.nrows(), rank );
        auto  R0    = B::Matrix< real_t >( rank, rank );
        auto  R1    = B::Matrix< real_t >( rank, rank );

        tbb::parallel_invoke( [&] () { tsqr( alpha, X0, T, U0, Q0, R0 ); },
                              [&] () { tsqr( alpha, X1, T, U1, Q1, R1 ); } );

        auto  Q2  = B::Matrix< real_t >( 2*rank, rank );
        auto  Q20 = B::Matrix< real_t >( Q2, B::Range( 0,    rank-1 ),   B::Range::all );
        auto  Q21 = B::Matrix< real_t >( Q2, B::Range( rank, 2*rank-1 ), B::Range::all );

        B::copy( R0, Q20 );
        B::copy( R1, Q21 );

        B::qr( Q2, R );

        auto  Q_0  = B::Matrix< real_t >( Q, rows0, B::Range::all );
        auto  Q_1  = B::Matrix< real_t >( Q, rows1, B::Range::all );

        tbb::parallel_invoke( [&] () { B::prod( real_t(1), Q0, Q20, real_t(0), Q_0 ); },
                              [&] () { B::prod( real_t(1), Q1, Q21, real_t(0), Q_1 ); } );
    }// if
    else
    {
        auto  W  = B::prod( alpha, X, T );
        auto  Q0 = B::Matrix< real_t >( Q, B::Range::all, B::Range( 0, rank_XT-1 ) );
        auto  Q1 = B::Matrix< real_t >( Q, B::Range::all, B::Range( rank_XT, rank-1 ) );

        B::copy( W, Q0 );
        B::copy( U, Q1 );

        B::qr( Q, R );
    }// else
}

void
TSQRNode::refine_ ()
{
    if ( X.nrows() > ntile )
    {
        const size_t  rank_XT = T.ncols();
        const size_t  rank_U  = U.ncols();
        const size_t  rank    = rank_XT + rank_U;

        auto  mid   = X.nrows() / 2;
        auto  rows0 = is( 0, mid-1 );
        auto  rows1 = is( mid, X.nrows()-1 );

        auto  mb_X0 = memblk_t( mb_X.base, bis( rows0 + mb_X.is.row_is().first(), mb_X.is.col_is() ) );
        auto  X0    = B::Matrix< real_t >( X, rows0, B::Range::all );
        auto  mb_X1 = memblk_t( mb_X.base, bis( rows1 + mb_X.is.row_is().first(), mb_X.is.col_is() ) );
        auto  X1    = B::Matrix< real_t >( X, rows1, B::Range::all );
        auto  mb_U0 = memblk_t( mb_U.base, bis( rows0 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  U0    = B::Matrix< real_t >( U, rows0, B::Range::all );
        auto  mb_U1 = memblk_t( mb_U.base, bis( rows1 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  U1    = B::Matrix< real_t >( U, rows1, B::Range::all );

        auto  Q0    = new B::Matrix< real_t >( X0.nrows(), rank );
        auto  Q1    = new B::Matrix< real_t >( X1.nrows(), rank );
        auto  R0    = new B::Matrix< real_t >( rank, rank );
        auto  R1    = new B::Matrix< real_t >( rank, rank );

        auto  tsqr_0 = new TSQRNode( alpha,
                                     mb_X0, X0, id_X + "0",
                                     mb_T, T,
                                     mb_U0, U0, id_U + "0",
                                     memblk( Q0 ), *Q0,
                                     memblk( R0 ), *R0 );
        auto  tsqr_1 = new TSQRNode( alpha,
                                     mb_X1, X1, id_X + "1",
                                     mb_T, T,
                                     mb_U1, U1, id_U + "1",
                                     memblk( Q1 ), *Q1,
                                     memblk( R1 ), *R1 );

        auto  Q2     = new B::Matrix< real_t >( 2*rank, rank );
        auto  Q20    = B::Matrix< real_t >( *Q2, B::Range( 0,    rank-1 ),   B::Range::all );
        auto  mb_Q20 = memblk_t( Q2->data(), bis( is( 0, rank-1 ),       is( 0, rank-1 ) ) );
        auto  Q21    = B::Matrix< real_t >( *Q2, B::Range( rank, 2*rank-1 ), B::Range::all );
        auto  mb_Q21 = memblk_t( Q2->data(), bis( is( rank, 2*rank-1 ),  is( 0, rank-1 ) ) );

        auto  tsqr_01 = new TSQRCombineNode( memblk( R0 ), *R0, "R(" + id_X + "0|" + id_U + "0)",
                                             memblk( R1 ), *R1, "R(" + id_X + "1|" + id_U + "1)",
                                             memblk( Q2 ), *Q2,
                                             mb_R, R );

        // B::copy( R0, Q20 );
        // B::copy( R1, Q21 );

        // B::qr( Q2, R );

        auto  Q_0    = B::Matrix< real_t >( Q, rows0, B::Range::all );
        auto  mb_Q_0 = memblk_t( mb_Q.base, bis( rows0 + mb_Q.is.row_is().first(), mb_Q.is.col_is() ) );
        auto  Q_1    = B::Matrix< real_t >( Q, rows1, B::Range::all );
        auto  mb_Q_1 = memblk_t( mb_Q.base, bis( rows1 + mb_Q.is.row_is().first(), mb_Q.is.col_is() ) );
        
        auto  mulT_0 = new MulTNode( memblk( Q0 ), *Q0, "Q0", mb_Q20, Q20, mb_Q_0, Q_0, "Q,0" );
        auto  mulT_1 = new MulTNode( memblk( Q1 ), *Q1, "Q1", mb_Q21, Q21, mb_Q_1, Q_1, "Q,1" );
        
        sub.resize( 5 );
        sub[0] = tsqr_0;
        sub[1] = tsqr_1;
        sub[2] = tsqr_01;
        sub[3] = mulT_0;
        sub[4] = mulT_1;
            
        tsqr_01->add_dep( tsqr_0 );
        tsqr_01->add_dep( tsqr_1 );
        mulT_0->add_dep( tsqr_0 ); mulT_0->add_dep( tsqr_01 );
        mulT_1->add_dep( tsqr_1 ); mulT_1->add_dep( tsqr_01 );
    }// if
    else
    {
        auto  qr = new QRNode( alpha, mb_X, X, id_X, mb_T, T, mb_U, U, id_U, mb_Q, Q, mb_R, R );

        sub.resize( 1 );
        sub[0] = qr;
    }
}

///////////////////////////////////////////////////////////////////////////////////////
//
// TSQRNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
qr ( const real_t                 alpha,
     const B::Matrix< real_t > &  X,
     const B::Matrix< real_t > &  T,
     const B::Matrix< real_t > &  U,
     B::Matrix< real_t > &        Q,
     B::Matrix< real_t > &        R )
{
    const size_t  rank_XT = T.ncols();
    const size_t  rank_U  = U.ncols();
    const size_t  rank    = rank_XT + rank_U;

    if ( X.nrows() != U.nrows() )
        HERROR( ERR_MAT_SIZE, "tsqr", "X and U differ in row size" );

    if ( X.ncols() != T.nrows() )
        HERROR( ERR_MAT_SIZE, "tsqr", "X and U are incompatible" );
    
    auto  W  = B::prod( alpha, X, T );
    auto  Q0 = B::Matrix< real_t >( Q, B::Range::all, B::Range( 0, rank_XT-1 ) );
    auto  Q1 = B::Matrix< real_t >( Q, B::Range::all, B::Range( rank_XT, rank-1 ) );
    
    B::copy( W, Q0 );
    B::copy( U, Q1 );
    
    B::qr( Q, R );
}

void
QRNode::refine_ ()
{
}

///////////////////////////////////////////////////////////////////////////////////////
//
// TSQRCombineNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
TSQRCombineNode::refine_ ()
{
    // auto  Q2  = B::Matrix< real_t >( 2*rank, rank );
    // auto  Q20 = B::Matrix< real_t >( Q2, B::Range( 0,    rank-1 ),   B::Range::all );
    // auto  Q21 = B::Matrix< real_t >( Q2, B::Range( rank, 2*rank-1 ), B::Range::all );
    
    // B::copy( R0, Q20 );
    // B::copy( R1, Q21 );

    // B::qr( Q2, R );
}

///////////////////////////////////////////////////////////////////////////////////////
//
// MulTNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
mulT ( const B::Matrix< real_t > &  X,
       const B::Matrix< real_t > &  T,
       B::Matrix< real_t > &        U )
{
    // cout << "mulT( " << U.nrows() << " )" << endl;

    if ( X.nrows() != U.nrows() )
        HERROR( ERR_MAT_SIZE, "mulT", "X and U differ in row size" );

    if ( X.ncols() != T.nrows() )
        HERROR( ERR_MAT_SIZE, "mulT", "incompatible dimensions in X and T" );

    if ( T.ncols() != U.ncols() )
        HERROR( ERR_MAT_SIZE, "mulT", "T and U differ in column size" );
    
    if ( X.nrows() > ntile )
    {
        auto  mid = X.nrows() / 2;
        auto  X0  = B::Matrix< real_t >( X, B::Range( 0, mid-1 ),         B::Range::all );
        auto  X1  = B::Matrix< real_t >( X, B::Range( mid, X.nrows()-1 ), B::Range::all );
        auto  U0  = B::Matrix< real_t >( U, B::Range( 0, mid-1 ),         B::Range::all );
        auto  U1  = B::Matrix< real_t >( U, B::Range( mid, U.nrows()-1 ), B::Range::all );

        tbb::parallel_invoke( [&] () { mulT( X0, T, U0 ); },
                              [&] () { mulT( X1, T, U1 ); } );
    }// if
    else
    {
        B::prod( real_t(1), X, T, real_t(0), U );
    }// else
}

void
MulTNode::refine_ ()
{
    if ( X.nrows() > ntile )
    {
        auto  mid   = X.nrows() / 2;
        auto  rows0 = is( 0, mid-1 );
        auto  rows1 = is( mid, U.nrows()-1 );
        
        auto  mb_X0 = memblk_t( mb_X.base, bis( rows0 + mb_X.is.row_is().first(), mb_X.is.col_is() ) );
        auto  mb_X1 = memblk_t( mb_X.base, bis( rows1 + mb_X.is.row_is().first(), mb_X.is.col_is() ) );
        auto  mb_U0 = memblk_t( mb_U.base, bis( rows0 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );
        auto  mb_U1 = memblk_t( mb_U.base, bis( rows1 + mb_U.is.row_is().first(), mb_U.is.col_is() ) );

        auto  X0    = B::Matrix< real_t >( X, rows0, B::Range::all );
        auto  X1    = B::Matrix< real_t >( X, rows1, B::Range::all );
        auto  U0    = B::Matrix< real_t >( U, rows0, B::Range::all );
        auto  U1    = B::Matrix< real_t >( U, rows1, B::Range::all );

        auto  mulT_0 = new MulTNode( mb_X0, X0, id_X + "0", mb_T, T, mb_U0, U0, id_U + "0" );
        auto  mulT_1 = new MulTNode( mb_X1, X1, id_X + "1", mb_T, T, mb_U1, U1, id_U + "1" );

        sub.resize( 2 );
        sub[0] = mulT_0;
        sub[1] = mulT_1;
    }// if
    else
    {
        auto  gemm = new GEMMNode( real_t(1),
                                    mb_X, apply_normal, X, id_X,
                                    mb_T, apply_normal, T, "T",
                                    real_t(0),
                                    mb_U, U, id_U );

        sub.resize( 1 );
        sub[0] = gemm;
    }// else
}

///////////////////////////////////////////////////////////////////////////////////////
//
// GEMMNode
//
///////////////////////////////////////////////////////////////////////////////////////

void
GEMMNode::refine_ ()
{
}

///////////////////////////////////////////////////////////////////////////////////////
//
// generate DAG
//
///////////////////////////////////////////////////////////////////////////////////////

void
gen_dag ( TMatrix *  A )
{
    auto                 n = A->rows();

    #if 0
    
    B::Matrix< real_t >  X( n, k );
    B::Matrix< real_t >  T( k, k );
    B::Matrix< real_t >  U( n, k );
    B::Matrix< real_t >  Q( n, 2*k );
    B::Matrix< real_t >  R( 2*k, 2*k );

    Node *  root = new TSQRNode( 1.0,
                                 memblk( X ), X, "X",
                                 memblk( T ), T,
                                 memblk( U ), U, "U",
                                 memblk( Q ), Q,
                                 memblk( R ), R );
    #endif

    #if 0
    
    B::Matrix< real_t >  U( n, k );
    B::Matrix< real_t >  T( k, k );
    B::Matrix< real_t >  V( n, k );
    B::Matrix< real_t >  RA( n, k );
    B::Matrix< real_t >  RB( n, k );
    TRkMatrix            R( is( 0, n-1 ), is( 0, n-1 ), RA, RB );

    Node *  root = new TruncateNode( 1.0,
                                     memblk( U ), U, "U",
                                     memblk( T ), T,
                                     memblk( V ), V, "V",
                                     & R );
    #endif

    #if 0
    
    B::Matrix< real_t >  U( n, k );
    B::Matrix< real_t >  T( k, k );
    B::Matrix< real_t >  V( n, k );

    Node *  root = new LRUpdateNode( memblk( U ), U, "U",
                                     memblk( T ), T,
                                     memblk( V ), V, "V",
                                     A );
    #endif

    #if 0

    B::Matrix< real_t >  U( n / 2, k );
    
    Node *  root = new SolveLNode( ptrcast( A, TBlockMatrix )->block( 0, 0 ),
                                   memblk( U ), U, "U" );
    #endif

    #if 1
    
    Node *  root = new LUNode( A );
    
    #endif

    auto  tic = Time::Wall::now();

    deque< Node * >  nodes;
    list< Node * >   tasks, start, end;

    nodes.push_back( root );
    
    while ( ! nodes.empty() )
    {
        deque< Node * >  subnodes;

        // first refine nodes
        for ( auto  node : nodes )
            node->refine();

        // then refine dependencies between nodes
        for ( auto  node : nodes )
            node->refine_deps();

        // collect refined nodes
        for ( auto  node : nodes )
        {
            if ( node->is_refined() )
            {
                for ( auto  sub : node->sub )
                    subnodes.push_back( sub );
            }// if
            else if ( node->check_deps() )
            {
                // node was not refined but dependencies were
                subnodes.push_back( node );
            }// if
            else
            {
                // neither node nore dependencies have changed: will not be touched
                tasks.push_back( node );
            }// else
        }// for

        // // finally delete all refined nodes
        // for ( auto  node : nodes )
        // {
        //     if ( node->is_refined() )
        //         delete node;
        // }// for
        
        nodes = std::move( subnodes );
    }// while

    auto  toc = Time::Wall::since( tic );
    
    std::cout << "    dag in     = " << format( "%.4f" ) % toc.seconds() << endl;

    //
    // adjust dependency counter
    //
    
    size_t  nedges = 0;
    
    for ( auto  t : tasks )
    {
        // for ( auto  out : t->out )
        //    out->dep_cnt++;
        t->dep_cnt = t->in.size();

        if ( t->dep_cnt == 0 )
            start.push_back( t );

        if ( t->out.empty() )
            end.push_back( t );
        else
            nedges += t->out.size();
    }// for

    // for ( auto  t : tasks )
    // {
    //     cout << t->to_string() << endl;

    //     cout << "   in  : " << t->in.size() << endl;
    //     for ( auto  in : t->in )
    //         cout << "      " << in->to_string() << endl;
           
    //     cout << "   out : " << t->out.size() << endl;
    //     for ( auto  out : t->out )
    //         cout << "      " << out->to_string() << endl;
        
    //     t->dep_cnt = t->in.size();

    //     cout << t->to_string() << " : " << t->dep_cnt << endl;
    // }// for

    cout << "#tasks = " << tasks.size() << endl;
    cout << "#edges = " << nedges << endl;
        
    //
    // print dag
    //

    if ( true )
    {
        ofstream  out( "dag.dot" );

        out << "digraph G {" << endl
            << "  size  = \"8,8\";" << endl
            << "  ratio = \"2.0\";" << endl
            << "  node [ shape = box, style = filled, fontsize = \"20\", fontname = \"Noto Sans\", height = \"1.5\" ];" << endl
            << "  edge [ arrowhead = open, color = \"#babdb6\" ];" << endl;

        for ( auto node : tasks )
        {
            if ( dynamic_cast< TSQRNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#fcaf3e\" ];" << endl;
            else if ( dynamic_cast< QRNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#fcaf3e\" ];" << endl;
            else if ( dynamic_cast< TSQRCombineNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#ce5c00\" ];" << endl;
            else if ( dynamic_cast< MulTNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#8ae234\" ];" << endl;
            else if ( dynamic_cast< GEMMNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#8ae234\" ];" << endl;
            else if ( dynamic_cast< AddNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#4e9a06\" ];" << endl;
            else if ( dynamic_cast< TruncateNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#ad7fa8\" ];" << endl;
            else if ( dynamic_cast< SVDNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#e9b96e\" ];" << endl;
            else if ( dynamic_cast< LUNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#ef2929\" ];" << endl;
            else if ( dynamic_cast< SolveLNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#729fcf\" ];" << endl;
            else if ( dynamic_cast< SolveUNode * >( node ) != nullptr )
                out << size_t(node) << "[ label = \"" << node->to_string() << "\", color = \"#729fcf\" ];" << endl;
            else
                out << size_t(node) << "[ label = \"" << node->to_string() << "\" ];" << endl;
        }// for

        for ( auto node : tasks )
        {
            auto  dep = node->out.begin();

            if ( dep != node->out.end() )
            {
                out << size_t(node) << " -> {";

                out << size_t(*dep);
        
                while ( dep != node->out.end() )
                {
                    out << ";" << size_t(*dep);
                    ++dep;
                }// for
            
                out << "};" << endl;
            }// if
        }// for

        out << "}" << endl;
    }// if
}

//
// print block-by-block comparison of A and B
//
void
compare_blocks ( TMatrix *  A,
                 TMatrix *  B )
{
    if ( is_blocked( A ) && is_blocked( B ) )
    {
        auto  BA = ptrcast( A, TBlockMatrix );
        auto  BB = ptrcast( B, TBlockMatrix );

        for ( uint  i = 0; i < BA->block_rows(); ++i )
            for ( uint  j = 0; j < BA->block_cols(); ++j )
                compare_blocks( BA->block( i, j ), BB->block( i, j ) );
    }// if
    else if ( A->type() == B->type() )
    {
        cout << A->typestr() << "( " << A->id() << " ) : " <<  diff_norm_F( A, B ) << endl;
    }// if
    else
        HERROR( ERR_CONSISTENCY, "compare_blocks", "different block structure" );
}

//
// main function
//
int
main ( int argc, char ** argv )
{
    real_t        eps  = real_t(1e-4);
    size_t        n    = 512;

    if ( argc > 1 ) n     = std::strtol( argv[1], nullptr, 0 );
    if ( argc > 2 ) k     = std::strtol( argv[2], nullptr, 0 );
    if ( argc > 3 ) ntile = std::strtol( argv[3], nullptr, 0 );

    double h = 1.0 / double(n);

    // if ( false )
    // {
    //     auto  U = B::random< real_t >( n, k );
    //     auto  T = B::random< real_t >( k, k );
    //     auto  V = B::random< real_t >( n, k );
    //     auto  X = B::random< real_t >( n, k );
    //     auto  Y = B::random< real_t >( n, k );

    //     DBG::write( U, "U.mat", "U" );
    //     DBG::write( T, "T.mat", "T" );
    //     DBG::write( V, "V.mat", "V" );
    //     DBG::write( X, "X.mat", "X" );
    //     DBG::write( Y, "Y.mat", "Y" );

    //     TRkMatrix  R( is( 0, n-1 ), is( 0, n-1 ), X, Y );

    //     auto  tic = Time::Wall::now();

    //     // truncate( -1.0, U, T, V, & R );
        
    //     {
    //         auto  W = B::prod( real_t(-1), U, T );
            
    //         R.add_rank( real_t(1), W, V, fixed_rank( k ) );
    //     }

    //     auto  toc = Time::Wall::since( tic );
        
    //     cout << std::fixed << std::setprecision( 4 ) << toc.seconds() << endl;
        
    //     DBG::write( & R, "R.mat", "R" );

    //     return 0;
    // }
    
    // if ( false )
    // {
    //     auto  U = B::random< real_t >( n, k );
    //     auto  T = B::random< real_t >( k, k );
    //     auto  V = B::random< real_t >( n, k );
    //     auto  Q = B::Matrix< real_t >( n, 2*k );
    //     auto  R = B::Matrix< real_t >( 2*k, 2*k );

    //     DBG::write( U, "U.mat", "U" );
    //     DBG::write( T, "T.mat", "T" );
    //     DBG::write( V, "V.mat", "V" );

    //     auto  tic = Time::Wall::now();
        
    //     for ( uint i = 0; i < 10; ++i )
    //     {
    //         tsqr( -1.0, U, T, V, Q, R );
    //         // {
    //         //     auto  W  = B::prod( real_t(1), U, T );
    //         //     auto  Q0 = B::Matrix< real_t >( Q, B::Range::all, B::Range( 0, k-1 ) );
    //         //     auto  Q1 = B::Matrix< real_t >( Q, B::Range::all, B::Range( k, 2*k-1 ) );

    //         //     B::copy( W, Q0 );
    //         //     B::copy( V, Q1 );

    //         //     B::qr( Q, R );
    //         // }// else
    //     }// for

    //     auto  toc = Time::Wall::since( tic );

    //     cout << std::fixed << std::setprecision( 4 ) << toc.seconds() << endl;

    //     DBG::write( Q, "Q.mat", "Q" );
    //     DBG::write( R, "R.mat", "R" );

    //     return 0;
    // }
    
    try
    {
        //
        // init HLIBpro
        //
        
        INIT();

        CFG::set_verbosity( 3 );
        // CFG::set_nthreads( 1 );

        //
        // build coordinates
        //

        vector< double * >  vertices( n, nullptr );
        vector< double * >  bbmin( n, nullptr );
        vector< double * >  bbmax( n, nullptr );

        for ( size_t i = 0; i < n; i++ )
        {
            vertices[i]    = new double;
            vertices[i][0] = h * double(i) + ( h / 2.0 ); // center of [i/h,(i+1)/h]

            // set bounding box (support) to [i/h,(i+1)/h]
            bbmin[i]       = new double;
            bbmin[i][0]    = h * double(i);
            bbmax[i]       = new double;
            bbmax[i][0]    = h * double(i+1);
        }// for

        auto  coord = make_unique< TCoordinate >( vertices, 1, bbmin, bbmax, copy_coord_data );

        //
        // build cluster tree and block cluster tree
        //

        TCardBSPPartStrat  part_strat;
        TBSPCTBuilder      ct_builder( & part_strat, ntile );
        auto               ct = ct_builder.build( coord.get() );
        // TStdGeomAdmCond    adm_cond;
        TOffDiagAdmCond    adm_cond;
        TBCBuilder         bct_builder;
        auto               bct = bct_builder.build( ct.get(), ct.get(), & adm_cond );

        if( verbose( 2 ) )
        {
            TPSClusterVis        c_vis;
            TPSBlockClusterVis   bc_vis;
            
            c_vis.print( ct->root(), "hodlr_ct" );
            bc_vis.id( true ).print( bct->root(), "hodlr_bct" );
        }// if
                
        //
        // build matrix
        //
        
        std::cout << "━━ building H-matrix ( eps = " << eps << " )" << std::endl;

        TTimer                      timer( WALL_TIME );
        unique_ptr< TProgressBar >  progress( verbose(2) ? new TConsoleProgressBar( cout ) : nullptr );
        TTruncAcc                   acc = fixed_rank( k );
        TLogCoeffFn                 cb_coeff( h );
        TPermCoeffFn< real_t >      coefffn( & cb_coeff, ct->perm_i2e(), ct->perm_i2e() );
        TACAPlus< real_t >          aca( & coefffn );
        TDenseMBuilder< real_t >    h_builder( & coefffn, & aca );
        TPSMatrixVis                mvis;
        
        timer.start();

        auto                        A = h_builder.build( bct.get(), unsymmetric, acc, progress.get() );
    
        timer.pause();
        std::cout << "    done in " << timer << std::endl;
        std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
        
        if( verbose( 2 ) )
        {
            mvis.svd( false );
            mvis.id( true );
            mvis.print( A.get(), "hodlr_A" );
        }// if

        CFG::Arith::use_dag = false;
        
        auto  tic = Time::Wall::now();
        auto  toc = Time::Wall::since( tic );

        std::cout << "━━ LU facorisation ( tile based )" << std::endl;
        
        gen_dag( A.get() );

        auto  B = A->copy();

        tic = Time::Wall::now();

        lu( B.get() );
        
        toc = Time::Wall::since( tic );
        auto  t_tile = toc.seconds();
        
        TLUInvMatrix  B_inv( B.get(), block_wise, store_inverse );

        std::cout << "    time       = " << std::fixed << std::setprecision( 3 ) << t_tile << std::endl;
        std::cout << "    inv. error = " << std::scientific << std::setprecision( 4 )
                  << inv_approx_2( A.get(), & B_inv ) << std::endl;

        {
            std::cout << "━━ LU facorisation ( standard )" << std::endl;
        
            auto  C = A->copy();

            tic = Time::Wall::now();
            
            lu( C.get(), fixed_rank( k ) );
            
            toc = Time::Wall::since( tic );

            TLUInvMatrix  A_inv( C.get(), block_wise, store_inverse );

            std::cout << "    done in " << toc << std::endl;
            std::cout << "    inversion error   = " << format( "%.4e" ) % inv_approx_2( A.get(), & A_inv ) << std::endl;
        }
        
        // auto  t_std = toc.seconds();
        
        // // compare_blocks( B.get(), C.get() );

        // // {
        // //     auto  BB = ptrcast( B.get(), TBlockMatrix );
        // //     auto  B00 = BB->block( 0, 0 );
        // //     auto  B01 = BB->block( 0, 1 );
        // //     auto  B10 = BB->block( 1, 0 );
        // //     auto  B11 = BB->block( 1, 1 );

        // //     DBG::write( B00, "B00.mat", "B00" );
        // //     DBG::write( B01, "B01.mat", "B01" );
        // //     DBG::write( B10, "B10.mat", "B10" );
        // //     DBG::write( B11, "B11.mat", "B11" );
        // // }

        // TLUInvMatrix  C_inv( C.get(), block_wise, store_inverse );

        // std::cout << "    time       = " << std::fixed << std::setprecision( 3 ) << t_std << std::endl;
        // std::cout << "    inv. error = " << std::scientific << std::setprecision( 4 )
        //           << inv_approx_2( A.get(), & C_inv ) << std::endl;
        
        DONE();
    }// try
    catch ( Error & e )
    {
        std::cout << e.to_string() << std::endl;
    }// catch
    
    return 0;
}
