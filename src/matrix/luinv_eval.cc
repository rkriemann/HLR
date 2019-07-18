//
// Project     : HLR
// File        : luinv_eval.cc
// Description : evaluation operator for the inverse of LU factorizations
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hlr/utils/checks.hh>
#include <hlr/utils/log.hh>
#include <hlr/seq/dag.hh>

#include <hlr/matrix/luinv_eval.hh>

namespace hlr { namespace matrix {

using namespace HLIB;

//
// ctor
//

luinv_eval::luinv_eval ( std::shared_ptr< TMatrix > &  M )
        : _mat( M )
        , _vec( nullptr )
{
    assert( _mat.get() != nullptr );

    //
    // set up mutex maps
    //
    
    idx_t  last = -1;

    for ( auto  i : _mat->row_is() )
    {
        const idx_t  ci = i / hlr::dag::CHUNK_SIZE;
            
        if ( ci != last )
        {
            last            = ci;
            _map_rows[ ci ] = std::make_unique< std::mutex >();
        }// if
    }// for

    last = -1;
    for ( auto  i : _mat->row_is() )
    {
        const idx_t  ci = i / hlr::dag::CHUNK_SIZE;
            
        if ( ci != last )
        {
            last            = ci;
            _map_cols[ ci ] = std::make_unique< std::mutex >();
        }// if
    }// for

    //
    // and the DAGs
    //

    auto  tic = Time::Wall::now();
    
    _dag_trsvl  = std::move( hlr::dag::gen_dag_solve_lower( apply_normal,  _mat.get(), & _vec, _map_rows, seq::dag::refine ) );
    _dag_trsvlt = std::move( hlr::dag::gen_dag_solve_lower( apply_trans,   _mat.get(), & _vec, _map_cols, seq::dag::refine ) );
    _dag_trsvlh = std::move( hlr::dag::gen_dag_solve_lower( apply_adjoint, _mat.get(), & _vec, _map_cols, seq::dag::refine ) );
                                                                                                        
    _dag_trsvu  = std::move( hlr::dag::gen_dag_solve_upper( apply_normal,  _mat.get(), & _vec, _map_rows, seq::dag::refine ) );
    _dag_trsvut = std::move( hlr::dag::gen_dag_solve_upper( apply_trans,   _mat.get(), & _vec, _map_cols, seq::dag::refine ) );
    _dag_trsvuh = std::move( hlr::dag::gen_dag_solve_upper( apply_adjoint, _mat.get(), & _vec, _map_cols, seq::dag::refine ) );

    auto  toc = Time::Wall::since( tic );

    log( 0, to_string( "luinv_eval : time for DAG setup = %.3e", toc.seconds() ) );
}
    
//
// linear operator mapping
//

//
// mapping function of linear operator A, e.g. y ≔ A(x).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
luinv_eval::apply  ( const TVector *  x,
                     TVector *        y,
                     const matop_t    op ) const
{
    assert( ! is_null( x ) && ! is_null( y ) );

    y->assign( 1.0, x );

    assert( ! is_null( dynamic_cast< TScalarVector * >( y ) ) );
    
    _vec = ptrcast( y, TScalarVector );

    if ( op == apply_normal )
    {
        auto  tic = Time::Wall::now();

        seq::dag::run( _dag_trsvl, acc_exact );
        seq::dag::run( _dag_trsvu, acc_exact );

        auto  toc = Time::Wall::since( tic );

        log( 4, to_string( "luinv_eval : time for DAG run   = %.3e", toc.seconds() ) );
    }// if
    else if ( op == apply_trans )
    {
        seq::dag::run( _dag_trsvut, acc_exact );
        seq::dag::run( _dag_trsvlt, acc_exact );
    }// if
    else // if ( op == apply_adjoint )
    {
        seq::dag::run( _dag_trsvuh, acc_exact );
        seq::dag::run( _dag_trsvlh, acc_exact );
    }// else
}

//
// mapping function with update: \a y ≔ \a y + \a α \a A( \a x ).
// Depending on \a op, either A, A^T or A^H is applied.
//
void
luinv_eval::apply_add  ( const real       alpha,
                         const TVector *  x,
                         TVector *        y,
                         const matop_t    op ) const
{
    assert( ! is_null( x ) && ! is_null( y ) );

    TScalarVector  t( x );
    
    _vec = ptrcast( & t, TScalarVector );
    
    if ( op == apply_normal )
    {
        auto  tic = Time::Wall::now();
        
        seq::dag::run( _dag_trsvl, acc_exact );
        seq::dag::run( _dag_trsvu, acc_exact );

        auto  toc = Time::Wall::since( tic );

        log( 4, to_string( "luinv_eval : time for DAG run   = %.3e", toc.seconds() ) );
    }// if
    else if ( op == apply_trans )
    {
        seq::dag::run( _dag_trsvut, acc_exact );
        seq::dag::run( _dag_trsvlt, acc_exact );
    }// if
    else // if ( op == apply_adjoint )
    {
        seq::dag::run( _dag_trsvuh, acc_exact );
        seq::dag::run( _dag_trsvlh, acc_exact );
    }// else

    y->axpy( alpha, & t );
}

void
luinv_eval::capply_add  ( const complex    alpha,
                          const TVector *  x,
                          TVector *        y,
                          const matop_t    op ) const
{
    assert( ! is_null( x ) && ! is_null( y ) );

    TScalarVector  t( x );
    
    _vec = ptrcast( & t, TScalarVector );
    
    if ( op == apply_normal )
    {
        seq::dag::run( _dag_trsvl, acc_exact );
        seq::dag::run( _dag_trsvu, acc_exact );
    }// if
    else if ( op == apply_trans )
    {
        seq::dag::run( _dag_trsvut, acc_exact );
        seq::dag::run( _dag_trsvlt, acc_exact );
    }// if
    else // if ( op == apply_adjoint )
    {
        seq::dag::run( _dag_trsvuh, acc_exact );
        seq::dag::run( _dag_trsvlh, acc_exact );
    }// else

    y->caxpy( alpha, & t );
}

void
luinv_eval::apply_add  ( const real       , // alpha,
                         const TMatrix *  , // X,
                         TMatrix *        , // Y,
                         const matop_t      // op
                         ) const
{
    assert( false );
}

}} // namespace hlr::matrix
