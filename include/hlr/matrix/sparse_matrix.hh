#ifndef __HLR_MATRIX_SPARSE_MATRIX_HH
#define __HLR_MATRIX_SPARSE_MATRIX_HH
//
// Project     : HLR
// Module      : sparse_matrix
// Description : sparse matrix based on Eigen
// Author      : Ronald Kriemann
// Copyright   : Max Planck Institute MIS 2004-2023. All Rights Reserved.
//

#if defined(HAS_EIGEN)
#  include <Eigen/Sparse>
#  include <Eigen/SparseLU>
#endif

#include <hpro/matrix/TMatrix.hh>
#include <hpro/matrix/TSparseMatrix.hh>

#include <hlr/utils/io.hh>

namespace hlr
{ 

using indexset = Hpro::TIndexSet;

// local matrix type
DECLARE_TYPE( sparse_matrix );

namespace matrix
{

#if defined(HAS_EIGEN)

template < typename T_value >
class sparse_matrix : public Hpro::TMatrix< T_value >
{
public:
    using  value_t = T_value;
    using  real_t  = Hpro::real_type_t< value_t >;
    using  spmat_t = Eigen::SparseMatrix< value_t, Eigen::ColMajor >;
    using  solver_t = Eigen::SparseLU< spmat_t, Eigen::COLAMDOrdering< int > >;
    
private:
    // local index set of matrix
    indexset  _row_is, _col_is;

    // actual sparse matrix
    spmat_t   _S;

    // signals, that sparse matrix is in factorized form
    bool      _factorized;

    // holds factorization data
    solver_t  _solver;
    
public:
    //
    // ctors
    //

    sparse_matrix ()
            : Hpro::TMatrix< value_t >()
            , _factorized( false )
    {}
    
    sparse_matrix ( const indexset  arow_is,
                    const indexset  acol_is )
            : Hpro::TMatrix< value_t >()
            , _row_is( arow_is )
            , _col_is( acol_is )
            , _S( arow_is.size(), acol_is.size() )
            , _factorized( false )
    {
        this->set_ofs( _row_is.first(), _col_is.first() );
    }

    sparse_matrix ( const Hpro::TSparseMatrix< value_t > &  S );
    
    // dtor
    virtual ~sparse_matrix ()
    {}
    
    //
    // matrix data
    //
    
    virtual size_t  nrows     () const { return _row_is.size(); }
    virtual size_t  ncols     () const { return _col_is.size(); }

    virtual size_t  rows      () const { return nrows(); }
    virtual size_t  cols      () const { return ncols(); }

    // use "op" versions from TMatrix
    using Hpro::TMatrix< value_t >::nrows;
    using Hpro::TMatrix< value_t >::ncols;
    
    // return true, if matrix is zero
    virtual bool    is_zero   () const { return ( _S.nonZeros() == 0 ); }
    
    virtual void    set_size  ( const size_t  anrows,
                                const size_t  ancols )
    {
        HLR_ERROR( "TODO" );
    }

    // direct access to Eigen sparse matrix
    spmat_t &         spmat         ()       { return _S; }
    const spmat_t &   spmat         () const { return _S; }

    // returns factorization flag
    bool              is_factorized () const { return _factorized; }
    
    // direct access to Eigen LU solver
    solver_t &        solver        ()       { return _solver; }
    const solver_t &  solver        () const { return _solver; }
    
    //
    // algebra routines
    //

    //! compute y ≔ α·op(this)·x + β·y
    virtual void  mul_vec    ( const value_t                     alpha,
                               const Hpro::TVector< value_t > *  x,
                               const value_t                     beta,
                               Hpro::TVector< value_t > *        y,
                               const matop_t                     op = Hpro::apply_normal ) const;
    using Hpro::TMatrix< value_t >::mul_vec;
    
    // same as above but only the dimension of the vector spaces is tested,
    // not the corresponding index sets
    virtual void  apply_add   ( const value_t                    alpha,
                                const blas::vector< value_t > &  x,
                                blas::vector< value_t > &        y,
                                const matop_t                    op = apply_normal ) const;
    using Hpro::TMatrix< value_t >::apply_add;
    
    // truncate matrix to accuracy acc
    virtual void truncate ( const Hpro::TTruncAcc & /* acc */ )
    {
        HLR_ERROR( "todo" );
    }

    // scale matrix by alpha
    virtual void scale    ( const value_t  alpha )
    {
        HLR_ERROR( "todo" );
    }

    void factorize ()
    {
        _solver.analyzePattern( _S ); 
        _solver.factorize( _S );
        _factorized = true;
    }
    
    //
    // RTTI
    //

    HPRO_RTTI_DERIVED( sparse_matrix, Hpro::TMatrix< value_t > )

    //
    // virtual constructor
    //

    // return matrix of same class (but no content)
    virtual auto   create       () const -> std::unique_ptr< Hpro::TMatrix< value_t > > { return std::make_unique< sparse_matrix< value_t > >(); }

    // return copy of matrix
    virtual auto   copy         () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        auto  M = Hpro::TMatrix< value_t >::copy();
    
        HLR_ASSERT( IS_TYPE( M.get(), sparse_matrix ) );

        auto  S = ptrcast( M.get(), sparse_matrix< value_t > );

        HLR_ERROR( "todo" );

        return M;
    }

    // return copy matrix wrt. given accuracy; if do_coarsen is set, perform coarsening
    virtual auto   copy         ( const Hpro::TTruncAcc &  /* acc */,
                                  const bool               /* do_coarsen */ = false ) const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return copy();
    }

    // return structural copy of matrix
    virtual auto   copy_struct  () const -> std::unique_ptr< Hpro::TMatrix< value_t > >
    {
        return std::make_unique< sparse_matrix< value_t > >( this->row_is(), this->col_is() );
    }

    // copy matrix data to A
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A ) const
    {
        HLR_ASSERT( IS_TYPE( A, sparse_matrix ) );

        Hpro::TMatrix< value_t >::copy_to( A );
    
        auto  S = ptrcast( A, sparse_matrix< value_t > );

        HLR_ERROR( "todo" );
    }
        

    // copy matrix data to A and truncate w.r.t. acc with optional coarsening
    virtual void   copy_to      ( Hpro::TMatrix< value_t > *  A,
                                  const Hpro::TTruncAcc &     /* acc */,
                                  const bool                  /* do_coarsen */ = false ) const
    {
        return copy_to( A );
    }
    
    //
    // misc.
    //
    
    // return size in bytes used by this object
    virtual size_t byte_size  () const
    {
        size_t  size = Hpro::TMatrix< value_t >::byte_size();

        size += sizeof(_row_is) + sizeof(_col_is);
        size += sizeof(_S);
        size += _S.outerSize() * sizeof(typename spmat_t::StorageIndex);
        size += _S.outerIndexPtr()[_S.outerSize()]  * ( sizeof(typename spmat_t::Scalar) + sizeof(typename spmat_t::StorageIndex) );
        
        return size;
    }

    // test data for invalid values, e.g. INF and NAN
    virtual void check_data () const
    {
        HLR_ERROR( "todo" );
    }
};

//
// type test
//
template < typename value_t >
inline
bool
is_sparse_eigen ( const Hpro::TMatrix< value_t > &  M )
{
    return IS_TYPE( &M, sparse_matrix );
}

template < typename value_t >
bool
is_sparse_eigen ( const Hpro::TMatrix< value_t > *  M )
{
    return ! is_null( M ) && IS_TYPE( M, sparse_matrix );
}

HLR_TEST_ALL( is_sparse_eigen, Hpro::TMatrix< value_t > )
HLR_TEST_ANY( is_sparse_eigen, Hpro::TMatrix< value_t > )

//
// ctors
//
template < typename value_t >
struct crs_triplet
{
    size_t   r;
    idx_t    c;
    value_t  v;

    size_t   row   () const { return r; }
    idx_t    col   () const { return c; }
    value_t  value () const { return v; }
};

template < typename value_t >
struct crs_triplet_iterator
{
    using  value_type      = crs_triplet< value_t >;
    using  difference_type = std::ptrdiff_t;
    using  pointer         = value_type *;
    using  reference       = value_type &;
        
    size_t                                  row;
    size_t                                  pos;
    const Hpro::TSparseMatrix< value_t > *  S;
    mutable crs_triplet< value_t >          ref;
    
    crs_triplet_iterator () = delete;
    crs_triplet_iterator ( const size_t                            apos,
                           const Hpro::TSparseMatrix< value_t > &  aS )
            : row(0)
            , pos(apos)
            , S(&aS)
    {
        while ( pos >= S->rowptr( row+1 ) )
            ++row;
        
        HLR_ASSERT( S != nullptr );
    }

    crs_triplet_iterator ( const size_t                            arow,
                           const size_t                            apos,
                           const Hpro::TSparseMatrix< value_t > &  aS )
            : row(arow)
            , pos(apos)
            , S(&aS)
    {
        HLR_ASSERT( S != nullptr );
    }

    crs_triplet_iterator  operator ++ ()
    {
        ++pos;
        
        while ( pos >= S->rowptr( row+1 ) )
            ++row;

        return crs_triplet_iterator( row, pos, *S );
    }
    crs_triplet_iterator  operator ++ (int)
    {
        auto  it = crs_triplet_iterator( row, pos++, *S );

        while ( pos >= S->rowptr( row+1 ) )
            ++row;

        return it;
    }

    crs_triplet_iterator  operator -- ()
    {
        --pos;
        
        while ( pos < S->rowptr( row ) )
            --row;

        return crs_triplet_iterator( row, pos, *S );
    }
    crs_triplet_iterator  operator -- (int)
    {
        auto  it = crs_triplet_iterator( row, pos--, *S );

        while ( pos < S->rowptr( row ) )
            --row;

        return it;
    }

    value_type    operator *  () const noexcept { return crs_triplet< value_t >{ row, S->colind( pos ), S->coeff( pos ) }; }
    pointer       operator -> () const noexcept { ref = crs_triplet< value_t >{ row, S->colind( pos ), S->coeff( pos ) }; return &ref; }

    bool      operator == ( const crs_triplet_iterator &  it ) const noexcept { return pos == it.pos; }
    bool      operator != ( const crs_triplet_iterator &  it ) const noexcept { return pos != it.pos; }
};

template < typename value_t >
sparse_matrix< value_t >::sparse_matrix ( const Hpro::TSparseMatrix< value_t > &  S )
        : Hpro::TMatrix< value_t >()
        , _row_is( S.row_is() )
        , _col_is( S.col_is() )
        , _S( S.nrows(), S.ncols() )
        , _factorized( false )
{
    this->set_ofs( _row_is.first(), _col_is.first() );

    // DEBUG
    // io::matlab::write( S, "S" );

    // {
    //     auto  begin_S = crs_triplet_iterator( 0, 0, S );
    //     auto  end_S   = crs_triplet_iterator( S.nrows(), S.n_non_zero(), S );

    //     for ( auto  it = begin_S; it != end_S; ++it )
    //     {
    //         std::cout << it->row() << ", " << it->col() << std::endl;
    //     }// for
    // }
    
    auto  begin_S = crs_triplet_iterator( 0, S );
    auto  end_S   = crs_triplet_iterator( S.n_non_zero(), S );

    _S.setFromTriplets( begin_S, end_S );
    _S.makeCompressed();

    // DEBUG
    // std::cout << Eigen::MatrixXd( _S ) << std::endl;
}

//
// matrix vector multiplication
//
template < typename value_t >
void
sparse_matrix< value_t >::mul_vec  ( const value_t                     alpha,
                                     const Hpro::TVector< value_t > *  vx,
                                     const value_t                     beta,
                                     Hpro::TVector< value_t > *        vy,
                                     const matop_t                     op ) const
{
}

template < typename value_t >
void
sparse_matrix< value_t >::apply_add ( const value_t                    alpha,
                                      const blas::vector< value_t > &  x,
                                      blas::vector< value_t > &        y,
                                      const matop_t                    op ) const
{
    HLR_ASSERT( x.stride() == 1 );
    HLR_ASSERT( y.stride() == 1 );

    // DEBUG
    // const auto  DS = Eigen::MatrixXd( _S );
    // auto        D  = blas::matrix< value_t >( nrows(), ncols() );
    // auto        y2 = blas::copy( y );

    // for ( int  i = 0; i < nrows(); ++i )
    //     for ( int  j = 0; j < ncols(); ++j )
    //         D(i,j) = DS(i,j);

    // io::matlab::write( D, "D" );
    // io::matlab::write( x, "x" );
    // io::matlab::write( y, "y1" );
    
    Eigen::Map< Eigen::VectorX< value_t > >  ex( x.data(), x.length() );
    Eigen::Map< Eigen::VectorX< value_t > >  ey( y.data(), y.length() );

    switch ( op )
    {
        case apply_normal     : ey += alpha * _S             * ex; break;
        case apply_conjugate  : ey += alpha * _S.conjugate() * ex; break;
        case apply_transposed : ey += alpha * _S.transpose() * ex; break;
        case apply_adjoint    : ey += alpha * _S.adjoint()   * ex; break;
    }// switch

    // blas::mulvec( alpha, blas::mat_view( op, D ), x, value_t(1), y );
    // blas::add( value_t(-1), y, y2 );

    // if ( blas::norm2( y2 ) > 1e-8 )
    // {
    //     std::cout << blas::norm2( y2 ) << std::endl;
    // }// if
    
    // io::matlab::write( y, "y2" );
}

#endif // HAS_EIGEN

}}// namespace hlr::matrix

#endif // __HLR_MATRIX_SPARSE_MATRIX_HH
