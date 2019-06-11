HLR
=======

A test bed for various hierarchical low-rank (HLR) algorithms and different
parallelization frameworks. 

HLR implements new algorithms (and older for reference) for low-rank
arithmetic. Furthermore, these algorithms are often implemented using different
parallelization frameworks to look into usability and performance of these systems.

Implemented problems
--------------------

  - logkernel:  1d logarithmic kernel example from HLIB
  - matern:     Matern covariance problem with random or grid coordinates
  - laplaceslp: Laplace SLP BEM problem over user defined grid
  
Clustering
----------

  - TLR:    BSP clustering with single level nxn block structure
  - HODLR:  off-diagonal admissibility with standard BSP clustering
  - TiledH: flatten upper <p> levels of standard hierarchy to have nxn
            blocks, each being an H-matrix
  - H:      standard H-matrix clustering

All clustering is cardinality balanced. If not defined otherwise, e.g., HODLR, standard
admissibility is used.

Parallelization Frameworks
--------------------------

  - seq:       standard sequential arithmetic
  - omp:       uses OpenMP loop/task parallelization
  - tbb:       TBB loop parallelization
  - tf:        C++-TaskFlow library (https://github.com/cpp-taskflow/cpp-taskflow)
  - hpx:       HPX C++ library (https://github.com/STEllAR-GROUP/hpx)
  - mpi        Message Passing Interface
    - bcast:   using blocking broadcasts
    - ibcast:  using non-blocking broadcasts
    - rdma:    using non-blocking RDMA
  - gaspi:     GASPI api implemented by GPI2 (http://www.gpi-site.com/)
  - dag:       uses task parallelism for arithmetic; construction and execution
               of DAG is available for various frameworks (seq, tbb, ...).

Special functions are stored in the corresponding modules in the sub directories, e.g.,
"tbb/arith.hh" with arithmetical functions implemented using TBB.

Prerequisites
=============
    
A C++17 compiler is needed for compiling HLR (and some of the frameworks).

Install the frameworks according to the corresponding installation instructions. 

HLIBpro can be downloaded from hlibpro.com. Afterwards set the "prefix" variable in the
file "bin/hlib-config" to the directory where you have installed HLIBpro and modify the
call to "hlib-config" in the file SConstruct such that it points to the correct location.

For MPI, it is assumed that at least MPI_THREAD_SERIALIZED is supported (optimal is
MPI_THREAD_MULTIPLE). 

For compilation, the *scons* make replacement is needed
(https://scons.org/).


Compilation
===========

Set compiler and compiler flags in the top section of the *SConstruct* file. Also modify
the paths to the various frameworks.

The set of programs (algorithms) is defined in the variable *BUILD*, which may contain the
values *dag*, *tlr*, *hodlr* or *tileh*. The parallelization frameworks are set in the
variable *FRAMEWORKS* and may be any of *seq*, *openmp*, *tbb*, *taskflow*, *hpx*, *mpi*
or *gpi2*.

Afterwards just enter

~~~
scons
~~~

to compile all wanted example programs.


Remarks
=======

HPX: - HPX has its own set of command line arguments. Arguments for the user program have
       to be provided after "--", e.g., "tlr-hpx -- -n 1024".
     - HPX performs thread affinity setting. Add "-t <n>" to the command line flags where
       *n* is the number of threads, e.g., "tlr-hpx -t 4 -- -n 1024"
     - CPU core binding is performed with HPX:

         -t 64 --hpx:bind="thread:0-63=core:0-63.pu:0"

       or (really equivalent???)

         htnumactl -N 0,1,2,3,4,5,6,7 -i ./dag-hpx -t 64 --hpx:bind=none

