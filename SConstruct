
import os

#
# set up compilation environment
#

debug   = False
fullmsg = False

CXXFLAGS  = '-O3 -march=native'
LINKFLAGS = '-lpthread'

if debug :
    CXXFLAGS  = '-g -march=native'
    LINKFLAGS = '-g -lpthread'
    
env = Environment( ENV        = os.environ,
                   CXX        = 'g++ -std=c++17',
                   CXXFLAGS   = Split( CXXFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   )
env.ParseConfig( 'hlibpro/bin/hlib-config --cflags --lflags' )

if not fullmsg :
    env.Replace( CCCOMSTR   = " CC     $SOURCES" )
    env.Replace( CXXCOMSTR  = " C++    $SOURCES" )
    env.Replace( LINKCOMSTR = " Link   $TARGET"  )

env.Append(  CPPPATH = [ '#include' ] )
env.Prepend( LIBS    = [ "common" ] )
env.Prepend( LIBPATH = [ "." ] )

common = env.StaticLibrary( 'common', [ 'src/apps/logkernel.cc',
                                        'src/apps/matern.cc',
                                        'src/cluster/H.cc',
                                        'src/cluster/hodlr.cc',
                                        'src/cluster/tileh.cc',
                                        'src/cluster/tlr.cc',
                                        'distr.cc',
                                        'src/dag/Node.cc',
                                        'src/dag/Graph.cc',
                                        'src/seq/dag.cc',
                                        'src/tbb/dag.cc',
                                        'src/dag/lu.cc' ] )

#
# default C++ environment
#

env.Program( 'tlr-seq.cc' )
# env.Program( 'hodlr-seq.cc' )
env.Program( 'tileh-seq.cc' )
# env.Program( 'dag-seq.cc' )

#
# OpenMP
#

# omp = env.Clone()
# omp.Append( CXXFLAGS  = "-fopenmp" )
# omp.Append( LINKFLAGS = "-fopenmp" )

# omp.Program( 'tlr-omp.cc' )
# omp.Program( 'hodlr-omp.cc' )

#
# TBB
#

# tbb = env.Clone()

# tbb.Program( 'tlr-tbb.cc' )
# tbb.Program( 'hodlr-tbb.cc' )
# tbb.Program( 'tileh-tbb.cc' )

#
# MPI
#

# mpi = env.Clone()
# mpi.ParseConfig( 'mpic++ --showme:compile' )
# mpi.ParseConfig( 'mpic++ --showme:link' )

# mpi.Program( 'tlr-mpi-bcast.cc' )
# mpi.Program( 'tlr-mpi-ibcast.cc' )
# mpi.Program( 'tlr-mpi-rdma.cc' )

# mpi.Program( 'tileh-mpi-bcast.cc' )
# mpi.Program( 'tileh-mpi-ibcast.cc' )

#
# HPX
#

# hpx = env.Clone()
# hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags --libs hpx_application" % ( "/opt/local/hpx/lib/pkgconfig" ) )
# hpx.MergeFlags( "-lhpx_iostreams" )

# hpx.Program( 'tlr-hpx.cc' )
# hpx.Program( 'hodlr-hpx.cc' )

#
# TaskFlow
#

# tf = env.Clone()
# tf.Append( CXXFLAGS = "-I/opt/local/cpp-taskflow/include" )

# tf.Program( 'tlr-tf.cc' )
# tf.Program( 'hodlr-tf.cc' )
