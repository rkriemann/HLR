#ifndef __HLR_DAG_LU_HH
#define __HLR_DAG_LU_HH
//
// Project     : HLib
// File        : lu.hh
// Description : generate DAG for LU factorization
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2019. All Rights Reserved.
//

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TRkMatrix.hh>

#include "hlr/dag/graph.hh"
#include "hlr/matrix/tiled_lrmatrix.hh"

namespace hlr
{

namespace dag
{

//
// compute DAG for in-place LU of <A>
//
graph
gen_dag_lu_ip     ( HLIB::TMatrix &    A,
                    const size_t       min_size,
                    refine_func_t      refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
graph
gen_dag_lu_oop    ( HLIB::TMatrix &    A,
                    const size_t       min_size,
                    refine_func_t      refine );

//
// compute DAG for out-of-place LU of <A> with automatic local dependencies
//
graph
gen_dag_lu_oop_auto ( HLIB::TMatrix &  A,
                      const size_t     min_size,
                      refine_func_t    refine );

//
// compute DAG for in-place LU of <A> using accumulators
//
graph
gen_dag_lu_oop_accu ( HLIB::TMatrix &  A,
                      const size_t     min_size,
                      refine_func_t    refine );

//
// compute DAG for LU of <A> using out-of-place ids
//
graph
gen_dag_lu_oop_accu_sep ( HLIB::TMatrix &  A,
                          const size_t     min_size,
                          refine_func_t    refine );

//
// compute DAG for coarse version of LU with on-the-fly DAGs for small matrices
//
graph
gen_dag_lu_oop_coarse  ( HLIB::TMatrix &    A,
                         const size_t       ncoarse,
                         refine_func_t      refine,
                         exec_func_t        fine_run );

//
// level wise computation of DAG for LU of <A>
// - recurse on diagonal blocks and perform per-level LU on global scope 
//
graph
gen_dag_lu_lvl    ( HLIB::TMatrix &  A,
                    const size_t     min_size );

//
// generate DAG for Tile-H LU factorisation
//
graph
gen_dag_lu_tileh ( HLIB::TMatrix &  A,
                   const size_t     min_size,
                   refine_func_t    refine,
                   exec_func_t      exec );

//
// compute DAG for solving L X = A with lower-triangular L
//
graph
gen_dag_solve_lower  ( const HLIB::TMatrix *  L,
                       HLIB::TMatrix *        A,
                       const size_t           min_size,
                       refine_func_t          refine );

//
// compute DAG for solving X U = A with upper triangular U
//
graph
gen_dag_solve_upper  ( const HLIB::TMatrix *  U,
                       HLIB::TMatrix *        A,
                       const size_t           min_size,
                       refine_func_t          refine );

//
// compute DAG for C = A B + C
//
graph
gen_dag_update       ( const HLIB::TMatrix *  A,
                       const HLIB::TMatrix *  B,
                       HLIB::TMatrix *        C,
                       const size_t           min_size,
                       refine_func_t          refine );

//
// compute DAG for tile-based LU of <A> in HODLR format
//
graph
gen_dag_lu_hodlr_tiled  ( HLIB::TMatrix &  A,
                          const size_t     ntile,
                          refine_func_t    refine );

//
// compute DAG for tile-based LU of <A> in HODLR format
// with lazy update evaluation
//
graph
gen_dag_lu_hodlr_tiled_lazy  ( HLIB::TMatrix &  A,
                               const size_t     ntile,
                               refine_func_t    refine );

//
// compute DAG for TSQR( X·T, U )
//
graph
gen_dag_tsqr ( const size_t                                                  n,
               hlr::matrix::tile_storage< HLIB::real > &                     X,
               std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &         T,
               hlr::matrix::tile_storage< HLIB::real > &                     U,
               std::shared_ptr< hlr::matrix::tile_storage< HLIB::real > > &  Q,
               std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &         R,
               refine_func_t                                                 refine );

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
graph
gen_dag_truncate ( hlr::matrix::tile_storage< HLIB::real > &              X,
                   std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  T,
                   hlr::matrix::tile_storage< HLIB::real > &              Y,
                   hlr::matrix::tiled_lrmatrix< HLIB::real > *            A,
                   refine_func_t                                          refine );

//
// compute DAG for truncation of [X·T,U(A)] • [Y,V(A)]'
//
graph
gen_dag_addlr ( hlr::matrix::tile_storage< HLIB::real > &              X,
                std::shared_ptr< HLIB::BLAS::Matrix< HLIB::real > > &  T,
                hlr::matrix::tile_storage< HLIB::real > &              Y,
                HLIB::TMatrix *                                        A,
                refine_func_t                                          refine );

}// namespace dag

}// namespace hlr

#endif // __HLR_DAG_LU_HH
