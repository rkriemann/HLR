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

luinv_eval::luinv_eval ( std::shared_ptr< TMatrix > &  M,
                         hlr::dag::refine_func_t       refine_func,
                         hlr::dag::exec_func_t         exec_func )
        : _mat( M )
        , _vec( nullptr )
        , _exec_func( exec_func )
{
    assert( _mat.get() != nullptr );

    //
    // set up mutex maps
    //
    
    for ( idx_t  i = _mat->row_is().first() / hlr::dag::CHUNK_SIZE; i <= idx_t(_mat->row_is().last() / hlr::dag::CHUNK_SIZE); ++i )
        _map_rows[ i ] = std::make_unique< std::mutex >();

    for ( idx_t  i = _mat->col_is().first() / hlr::dag::CHUNK_SIZE; i <= idx_t(_mat->col_is().last() / hlr::dag::CHUNK_SIZE); ++i )
        _map_cols[ i ] = std::make_unique< std::mutex >();

    //
    // and the DAGs
    //

    auto  tic = Time::Wall::now();
    
    _dag_trsvl  = std::move( hlr::dag::gen_dag_solve_lower( apply_normal,  _mat.get(), & _vec, _map_rows, refine_func ) );
    _dag_trsvlt = std::move( hlr::dag::gen_dag_solve_lower( apply_trans,   _mat.get(), & _vec, _map_cols, refine_func ) );
    _dag_trsvlh = std::move( hlr::dag::gen_dag_solve_lower( apply_adjoint, _mat.get(), & _vec, _map_cols, refine_func ) );
                                                                                                        
    _dag_trsvu  = std::move( hlr::dag::gen_dag_solve_upper( apply_normal,  _mat.get(), & _vec, _map_rows, refine_func ) );
    _dag_trsvut = std::move( hlr::dag::gen_dag_solve_upper( apply_trans,   _mat.get(), & _vec, _map_cols, refine_func ) );
    _dag_trsvuh = std::move( hlr::dag::gen_dag_solve_upper( apply_adjoint, _mat.get(), & _vec, _map_cols, refine_func ) );

    auto  toc = Time::Wall::since( tic );

    log( 2, to_string( "luinv_eval : time for DAG setup = %.3e", toc.seconds() ) );
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
        _exec_func( _dag_trsvl, acc_exact );
        _exec_func( _dag_trsvu, acc_exact );
    }// if
    else if ( op == apply_trans )
    {
        _exec_func( _dag_trsvut, acc_exact );
        _exec_func( _dag_trsvlt, acc_exact );
    }// if
    else // if ( op == apply_adjoint )
    {
        _exec_func( _dag_trsvuh, acc_exact );
        _exec_func( _dag_trsvlh, acc_exact );
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
        
        _exec_func( _dag_trsvl, acc_exact );
        _exec_func( _dag_trsvu, acc_exact );

        auto  toc = Time::Wall::since( tic );

        log( 4, to_string( "luinv_eval : time for DAG run   = %.3e", toc.seconds() ) );
    }// if
    else if ( op == apply_trans )
    {
        _exec_func( _dag_trsvut, acc_exact );
        _exec_func( _dag_trsvlt, acc_exact );
    }// if
    else // if ( op == apply_adjoint )
    {
        _exec_func( _dag_trsvuh, acc_exact );
        _exec_func( _dag_trsvlh, acc_exact );
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
        _exec_func( _dag_trsvl, acc_exact );
        _exec_func( _dag_trsvu, acc_exact );
    }// if
    else if ( op == apply_trans )
    {
        _exec_func( _dag_trsvut, acc_exact );
        _exec_func( _dag_trsvlt, acc_exact );
    }// if
    else // if ( op == apply_adjoint )
    {
        _exec_func( _dag_trsvuh, acc_exact );
        _exec_func( _dag_trsvlh, acc_exact );
    }// else

    y->caxpy( alpha, & t );
}

void
luinv_eval::apply_add  ( const real       /* alpha */,
                         const TMatrix *  /* X */,
                         TMatrix *        /* Y */,
                         const matop_t    /* op */ ) const
{
    assert( false );
}

//
// same as above but only the dimension of the vector spaces is tested,
// not the corresponding index sets
//
void
luinv_eval::apply_add   ( const real                       /* alpha */,
                          const blas::Vector< real > &     /* x */,
                          blas::Vector< real > &           /* y */,
                          const matop_t                    /* op */ ) const
{
    HLR_ASSERT( false );
}

void
luinv_eval::apply_add   ( const complex                    /* alpha */,
                          const blas::Vector< complex > &  /* x */,
                          blas::Vector< complex > &        /* y */,
                          const matop_t                    /* op */ ) const
{
    HLR_ASSERT( false );
}

}} // namespace hlr::matrix
