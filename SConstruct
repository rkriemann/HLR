
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

env.Prepend( LIBS    = [ "common" ] )
env.Prepend( LIBPATH = [ "." ] )

common = env.StaticLibrary( 'common', [ 'logkernel.cc',
                                        'matern.cc',
                                        'tlr.cc',
                                        'hodlr.cc',
                                        'stdh.cc',
                                        'tileh.cc',
                                        'distr.cc',
                                        'dag/Node.cc',
                                        'dag/Graph.cc',
                                        'dag/seq_run.cc',
                                        'dag/tbb_run.cc',
                                        'dag/lu.cc' ] )

#
# default C++ environment
#

env.Program( 'tlr-seq.cc' )
env.Program( 'hodlr-seq.cc' )
env.Program( 'tileh-seq.cc' )
env.Program( 'dag-seq.cc' )

#
# OpenMP
#

omp = env.Clone()
omp.Append( CXXFLAGS  = "-fopenmp" )
omp.Append( LINKFLAGS = "-fopenmp" )

omp.Program( 'tlr-omp.cc' )
omp.Program( 'hodlr-omp.cc' )

#
# TBB
#

tbb = env.Clone()

tbb.Program( 'tlr-tbb.cc' )
tbb.Program( 'hodlr-tbb.cc' )
tbb.Program( 'tileh-tbb.cc' )

#
# MPI
#

mpi = env.Clone()
#mpi.Append( LINKFLAGS = "-lboost_mpi" )
mpi.ParseConfig( 'mpic++ --showme:compile' )
mpi.ParseConfig( 'mpic++ --showme:link' )

mpi.Program( 'tlr-mpi-bcast.cc' )
mpi.Program( 'tlr-mpi-ibcast.cc' )
mpi.Program( 'tlr-mpi-rdma.cc' )

mpi.Program( 'tileh-mpi-bcast.cc' )
mpi.Program( 'tileh-mpi-ibcast.cc' )

#
# HPX
#

hpx = env.Clone()
hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags --libs hpx_application" % ( "/opt/local/hpx/lib/pkgconfig" ) )
hpx.MergeFlags( "-lhpx_iostreams" )

hpx.Program( 'tlr-hpx.cc' )
hpx.Program( 'hodlr-hpx.cc' )

#
# TaskFlow
#

tf = env.Clone()
tf.Append( CXXFLAGS = "-I/opt/local/cpp-taskflow/include" )

tf.Program( 'tlr-tf.cc' )
tf.Program( 'hodlr-tf.cc' )
