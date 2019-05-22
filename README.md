HLR-HPC
=======

Testing of various HLR clustering and arithmetic algorithms for
different problems with different parallelization frameworks.

Implemented problems
--------------------

  - logkernel:  1d logarithmic kernel example from HLIB
  - matern:     Matern covariance problem with random coordinates
  
Clustering
----------

  - TLR:    BSP clustering with single level nxn block structure
  - HODLR:  off-diagonal admissibility with standard BSP clustering
  - TiledH: flatten upper <p> levels of standard hierarchy to have nxn
            blocks, each being an H-matrix
  - H:      standard H-matrix clustering

All clustering is cardinality balanced. If not defined otherwise, e.g., HODLR,
weak admissibility is used.

Parallelization Frameworks
--------------------------

  - seq:       standard sequential arithmetic
  - omp:       uses OpenMP loop parallelization
  - tbb:       TBB loop parallelization
  - mpi        Message Passing Interface
    - bcast:   using blocking broadcasts
    - ibcast:  using non-blocking broadcasts
    - rdma:    using non-blocking RDMA
  - hpx:       HPX C++ library (https://github.com/STEllAR-GROUP/hpx)
  - tf:        C++-TaskFlow library (https://github.com/cpp-taskflow/cpp-taskflow)
  - dag:       uses task parallelism for arithmetic; execution of DAG is
               available for various frameworks (seq, tbb).

Special functions are stored in the corresponding modules in the sub directories, e.g.,
"include/tbb/arith.hh" with arithmetical functions implemented with TBB.

Prerequisites
=============
    
A C++17 compiler is needed for compiling HLR-HPC (and some of the
frameworks).

Install the frameworks according to the corresponding installation
instructions. 

HLIBpro can be downloaded from hlibpro.com. Afterwards set the
"prefix" variable in the file "bin/hlib-config" to the directory where
you have installed HLIBpro and modify the call to "hlib-config" in the
file SConstruct such that it points to the correct location.

For MPI, it is assumed that at least MPI_THREAD_SERIALIZED is
supported (optimal is MPI_THREAD_MULTIPLE).

For compilation, the *scons* make replacement is needed
(https://scons.org/).


Compilation
===========

Modify the paths to the various frameworks in the file
*SConstruct*. If a framework is not installed, just comment out the
corresponding part in *SConstruct*.

Enter

~~~
scons
~~~

to compile all examples.


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

