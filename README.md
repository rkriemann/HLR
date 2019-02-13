HLR-HPC
=======

Testing of various HLR clustering and arithmetic algorithms for
different problems with different parallelization frameworks.

Implemented problems:

  - logkernel:  1d logarithmic kernel example from HLIB
  - matern:     Matern covariance problem with random coordinates
  
Clustering:

  - TLR:    BSP clustering with single level nxn block structure
  - HODLR:  off-diagonal admissibility with standard BSP clustering

Parallelization Frameworks:

  - seq:       standard sequential arithmetic
  - omp:       uses OpenMP loop parallelization
  - tbb:       TBB loop parallelization
  - mpi        Message Passing Interface
    - bcast:   using blocking broadcasts
    - ibcast:  using non-blocking broadcasts
    - rdma:    using non-blocking RDMA
  - hpx:       HPX C++ library (https://github.com/STEllAR-GROUP/hpx)
  - tf:        C++-TaskFlow library (https://github.com/cpp-taskflow/cpp-taskflow)


Prerequisites
-------------
    
A C++17 compiler is needed for compiling HLR-HPC (and some of the
frameworks).

Install the frameworks according to the corresponding installation
instructions. 

For MPI, it is assumed that at least MPI_THREAD_SERIALIZED is
supported (optimal is MPI_THREAD_MULTIPLE).

For compilation, the *scons* make replacement is needed
(https://scons.org/).


Compilation
-----------

Modify the paths to the various frameworks in the file
*SConstruct*. If a framework is not installed, just comment out the
corresponding part in *SConstruct*.

Enter

~~~
scons
~~~

to compile all examples.


Remarks
-------

HPX performs thread affinity setting. Add "-t <n>" to the command
line flags where *n* is the number of threads.
