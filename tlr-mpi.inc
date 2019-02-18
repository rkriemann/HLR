#include "tensor.hh"
#include "tools.hh"

namespace TLR
{

namespace MPI
{

typeid_t  type_dense, type_lr, type_ghost;
double    time_mpi = 0;
int       RANK     = 0;

std::unique_ptr< TMatrix >
create_matrix ( const TMatrix *  A,
                const typeid_t   type,
                const int        proc )
{
    std::unique_ptr< TMatrix >  T;

    assert(( type == type_dense ) || ( type == type_lr ));
    
    if ( type == type_dense )
        T = make_unique< TDenseMatrix >( A->row_is(), A->col_is(), A->is_complex() );
    else if ( type == type_lr )
        T = make_unique< TRkMatrix >( A->row_is(), A->col_is(), A->is_complex() );

    T->set_id( A->id() );
    T->set_procs( ps_single( proc ) );
    
    return T;
}

tensor2< typeid_t >
build_type_matrix ( const TBlockMatrix *  A )
{
    const auto           nbr = A->nblock_rows();
    const auto           nbc = A->nblock_cols();
    tensor2< typeid_t >  mat_types( nbr, nbc );
    mpi::communicator    world;
    const auto           nprocs = world.size();

    for ( uint  i = 0; i < nbr; ++i )
        for ( uint  j = 0; j < nbc; ++j )
            mat_types(i,j) = A->block( i, j )->type();
    
    for ( uint  p = 0; p < nprocs; ++p )
    {
        tensor2< typeid_t >  rem_types( mat_types );

        world.broadcast( & rem_types(0,0), nbr * nbc, p );
        
        for ( uint  i = 0; i < nbr; ++i )
            for ( uint  j = 0; j < nbc; ++j )
                if ( rem_types(i,j) != type_ghost )
                    mat_types(i,j) = rem_types(i,j);
    }// for

    // for ( uint  i = 0; i < nbr; ++i )
    // {
    //     for ( uint  j = 0; j < nbc; ++j )
    //         std::cout << RTTI::id_to_type( mat_types(i,j) ) << "  ";
    //     std::cout << std::endl;
    // }// for

    return mat_types;
}

void
build_row_comms ( const TBlockMatrix *                   A,
                  vector< mpi::communicator > &          row_comms,
                  vector< list< int > > &                row_procs,
                  vector< unordered_map< int, int > > &  row_maps )
{
    const auto         nbr = A->nblock_rows();
    const auto         nbc = A->nblock_cols();
    mpi::communicator  world;
    const auto         pid    = world.rank();
    
    for ( uint  i = 0; i < nbr; ++i )
    {
        list< int >  procs;
        
        for ( uint  j = 0; j < nbc; ++j )
            procs.push_back( A->block( i, j )->procs().master() );

        procs.sort();
        procs.unique();

        uint  pos = nbr;
            
        for ( uint  l = 0; l < i; ++l )
        {
            if ( procs == row_procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for

        // std::cout << i << " : ";
        // for ( auto  p : procs ) std::cout << p << ", ";
        // std::cout << " (" << ( pos == nbr ? i : pos ) << ")" << std::endl;
            
        // use previously created communicator or create new if none found
        if ( pos < nbr )
        {
            row_comms[i] = row_comms[pos];
            row_maps[i]  = row_maps[pos];
        }// if
        else
        {
            row_comms[i] = world.split( contains( procs, pid ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            for ( auto p : procs )
                row_maps[i][p] = comm_rank++;
        }// else
            
        row_procs[i] = std::move( procs );
    }// for
}

void
build_col_comms ( const TBlockMatrix *                   A,
                  vector< mpi::communicator > &          col_comms,
                  vector< list< int > > &                col_procs,
                  vector< unordered_map< int, int > > &  col_maps )
{
    const auto         nbr = A->nblock_rows();
    const auto         nbc = A->nblock_cols();
    mpi::communicator  world;
    const auto         pid    = world.rank();

    for ( uint  j = 0; j < nbc; ++j )
    {
        list< int >  procs;
            
        for ( uint  i = 0; i < nbr; ++i )
            procs.push_back( A->block( i, j )->procs().master() );

        procs.sort();
        procs.unique();

        uint  pos = nbc;
            
        for ( uint  l = 0; l < j; ++l )
        {
            if ( procs == col_procs[l] )
            {
                pos = l;
                break;
            }// if
        }// for
            
        // std::cout << j << " : ";
        // for ( auto  p : procs ) std::cout << p << ", ";
        // std::cout << " (" << ( pos == nbc ? j : pos ) << ")" << std::endl;

        // use previously created communicator or create new if none found
        if ( pos < nbc )
        {
            col_comms[j] = col_comms[pos];
            col_maps[j]  = col_maps[pos];
        }// if
        else
        {
            col_comms[j] = world.split( contains( procs, pid ) );
            // rank in new communicator is 0..#procs-1 with local ranks equally ordered as global ranks
            int  comm_rank = 0;
            for ( auto p : procs )
                col_maps[j][p] = comm_rank++;
        }// else
            
        col_procs[j] = std::move( procs );
    }// for
}

}// namespace MPI

}// namespace TLR

// Local Variables:
// mode: c++
// End:
