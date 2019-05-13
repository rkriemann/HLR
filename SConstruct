
import os

debug   = False
warn    = False
fullmsg = False

CXX      = 'g++'
CXXFLAGS = '-std=c++17'

######################################################################
#
# helper functions
#
######################################################################

#
# return first line of output of given program
#
def readln ( prog ):
    text = ''

    try :
        file = os.popen( prog, 'r' )
        text = file.readline()
        file.close()
    except :
        pass

    return text

######################################################################
#
# set up compilation environment
#
######################################################################

OPTFLAGS  = '-O3 -march=native'
WARNFLAGS = ''
LINKFLAGS = ''

if debug :
    OPTFLAGS  = '-g -march=native'
    LINKFLAGS = '-g'

if warn :
    WARNFLAGS = readln( '/home/rok/bin/cpuflags --comp %s --warn' % CXX )
    
env = Environment( ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS + ' ' + OPTFLAGS + ' ' + WARNFLAGS ),
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
                                        'src/apps/Laplace.cc',
                                        'src/cluster/H.cc',
                                        'src/cluster/hodlr.cc',
                                        'src/cluster/tileh.cc',
                                        'src/cluster/tlr.cc',
                                        'src/dag/Graph.cc',
                                        'src/dag/LocalGraph.cc',
                                        'src/dag/Node.cc',
                                        'src/dag/lu.cc',
                                        'src/mpi/distr.cc',
                                        'src/seq/dag.cc',
                                        'src/utils/compare.cc' ] )

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
omp.Program( 'tileh-omp.cc' )

#
# TBB
#

tbb = env.Clone()

tbb.Program( 'tlr-tbb.cc' )
tbb.Program( 'hodlr-tbb.cc' )
tbb.Program( 'tileh-tbb.cc' )
tbb.Program( 'dag-tbb', [ 'dag-tbb.cc', 'src/tbb/dag.cc' ] )

#
# TaskFlow
#

tf = env.Clone()
tf.Append( CXXFLAGS = "-I/opt/local/cpp-taskflow/include" )
tf.Append( LIBS     = [ "pthread" ] )

tf.Program( 'tlr-tf.cc' )
tf.Program( 'hodlr-tf.cc' )
tf.Program( 'dag-tf', [ 'dag-tf.cc', 'src/tf/dag.cc' ] )

#
# MPI
#

mpi = env.Clone()
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
hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags hpx_application" % ( "/opt/local/hpx/lib/pkgconfig" ) )
hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   hpx_application" % ( "/opt/local/hpx/lib/pkgconfig" ) )
hpx.Append( LIBS = [ "hpx_iostreams" ] )

hpx.Program( 'tlr-hpx.cc' )
hpx.Program( 'hodlr-hpx.cc' )
hpx.Program( 'dag-hpx', [ 'dag-hpx.cc', 'src/hpx/dag.cc' ] )

#
# GASPI
#

gpi = env.Clone()
gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags GPI2" % ( "/opt/local/gpi2/lib64/pkgconfig" ) )
gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   GPI2" % ( "/opt/local/gpi2/lib64/pkgconfig" ) )
gpi.Append( LIBS = [ "pthread" ] )

gpi.Program( 'tlr-gaspi.cc' )
# gpi.Program( 'hodlr-gpi.cc' )
