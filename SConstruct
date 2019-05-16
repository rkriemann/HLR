
import os

######################################################################
#
# general settings
#
######################################################################

debug        = True
warn         = False
fullmsg      = False

CXX          = 'g++'
CXXFLAGS     = '-std=c++17'

OPTFLAGS     = '-O3 -march=native'
WARNFLAGS    = '-Wall'
LINKFLAGS    = ''

# set of programs to build: dag, tlr, hodlr, tileh
BUILD        = [ 'dag' ]

# set of frameworks to use: seq, openmp, tbb, taskflow, hpx, mpi, gpi2
FRAMEWORKS   = [ 'seq', 'openmp', 'tbb', 'taskflow' ]

# directories for the various external libraries
HPRO_DIR     = 'hlibpro'
TBB_DIR      = '/usr'
TASKFLOW_DIR = '/opt/local/cpp-taskflow'
HPX_DIR      = '/opt/local/hpx'
GPI2_DIR     = '/opt/local/gpi2'

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

if debug :
    OPTFLAGS  = '-g -march=native'
    LINKFLAGS = '-g'

if warn :
    WARNFLAGS = readln( 'cpuflags --comp %s --warn' % CXX )
    
env = Environment( ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS + ' ' + OPTFLAGS + ' ' + WARNFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   )
env.ParseConfig( os.path.join( HPRO_DIR, 'bin', 'hlib-config' ) + ' --cflags --lflags' )

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
# default sequential environment
#

if 'seq' in FRAMEWORKS :
    if 'tlr'   in BUILD : env.Program( 'tlr-seq.cc' )
    if 'hodlr' in BUILD : env.Program( 'hodlr-seq.cc' )
    if 'tileh' in BUILD : env.Program( 'tileh-seq.cc' )
    if 'dag'   in BUILD : env.Program( 'dag-seq.cc' )

#
# OpenMP
#

if 'openmp' in FRAMEWORKS :
    omp = env.Clone()
    omp.Append( CXXFLAGS  = "-fopenmp" )
    omp.Append( LINKFLAGS = "-fopenmp" )

    if 'tlr'   in BUILD : omp.Program( 'tlr-omp.cc' )
    if 'hodlr' in BUILD : omp.Program( 'hodlr-omp.cc' )
    if 'tileh' in BUILD : omp.Program( 'tileh-omp.cc' )
    if 'dag'   in BUILD : omp.Program( 'dag-omp', [ 'dag-omp.cc', 'src/omp/dag.cc' ] )

#
# TBB
#

if 'tbb' in FRAMEWORKS :
    tbb = env.Clone()
    tbb.Append( CPPPATH = os.path.join( TBB_DIR, "include" ) )
    tbb.Append( LIBPATH = os.path.join( TBB_DIR, "lib" ) )

    if 'tlr'   in BUILD : tbb.Program( 'tlr-tbb.cc' )
    if 'hodlr' in BUILD : tbb.Program( 'hodlr-tbb.cc' )
    if 'tileh' in BUILD : tbb.Program( 'tileh-tbb.cc' )
    if 'dag'   in BUILD : tbb.Program( 'dag-tbb', [ 'dag-tbb.cc', 'src/tbb/dag.cc' ] )

#
# TaskFlow
#

if 'taskflow' in FRAMEWORKS :
    tf = env.Clone()
    tf.Append( CPPPATH = os.path.join( TASKFLOW_DIR, "include" ) )
    tf.Append( LIBS    = [ "pthread" ] )
    
    if 'tlr'   in BUILD : tf.Program( 'tlr-tf.cc' )
    if 'hodlr' in BUILD : tf.Program( 'hodlr-tf.cc' )
    if 'dag'   in BUILD : tf.Program( 'dag-tf', [ 'dag-tf.cc', 'src/tf/dag.cc' ] )

#
# MPI
#

if 'mpi' in FRAMEWORKS :
    mpi = env.Clone()
    mpi.ParseConfig( 'mpic++ --showme:compile' )
    mpi.ParseConfig( 'mpic++ --showme:link' )
    
    if 'tlr'   in BUILD :
        mpi.Program( 'tlr-mpi-bcast.cc' )
        mpi.Program( 'tlr-mpi-ibcast.cc' )
        mpi.Program( 'tlr-mpi-rdma.cc' )
    
    if 'tileh' in BUILD :
        mpi.Program( 'tileh-mpi-bcast.cc' )
        mpi.Program( 'tileh-mpi-ibcast.cc' )

#
# HPX
#

if 'hpx' in FRAMEWORKS :
    hpx = env.Clone()
    hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags hpx_application" % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   hpx_application" % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.Append( LIBS = [ "hpx_iostreams" ] )
    
    if 'tlr'   in BUILD : hpx.Program( 'tlr-hpx.cc' )
    if 'hodlr' in BUILD : hpx.Program( 'hodlr-hpx.cc' )
    if 'dag'   in BUILD : hpx.Program( 'dag-hpx', [ 'dag-hpx.cc', 'src/hpx/dag.cc' ] )

#
# GASPI
#

if 'gpi2' in FRAMEWORKS :
    gpi = env.Clone()
    gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags GPI2" % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   GPI2" % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.Append( LIBS = [ "pthread" ] )
    
    if 'tlr'   in BUILD : gpi.Program( 'tlr-gaspi.cc' )
    if 'hodlr' in BUILD : gpi.Program( 'hodlr-gpi.cc' )
