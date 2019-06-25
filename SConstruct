
# to enable print() syntax with python2
from __future__ import print_function

import os

######################################################################
#
# general settings
#
######################################################################

fullmsg      = False
debug        = False
profile      = False
optimise     = True
warn         = False

# cache file storing SCons settings
opts_file    = '.scons.options'

CXX          = 'g++'
CXXFLAGS     = '-std=c++17'

OPTFLAGS     = '-O3 -march=native'
WARNFLAGS    = '-Wall'
LINKFLAGS    = ''
DEFINES      = 'BOOST_SYSTEM_NO_DEPRECATED'

# set of programs to build: dag, tlr, hodlr, tileh (or "all")
PROGRAMS     = [ 'tlr', 'hodlr', 'tileh', 'dag' ]

# set of frameworks to use: seq, openmp, tbb, taskflow, hpx, mpi, gpi2 (or "all")
FRAMEWORKS   = [ 'seq', 'openmp', 'tbb', 'taskflow', 'hpx', 'mpi', 'gpi2' ]

# directories for the various external libraries
HPRO_DIR     = 'hlibpro'
TBB_DIR      = '/usr'
TASKFLOW_DIR = '/opt/local/cpp-taskflow/2.2.0'
HPX_DIR      = '/opt/local/hpx'
GPI2_DIR     = '/opt/local/gpi2'

JEMALLOC_DIR = '/opt/local/jemalloc/5.1.0'
LIKWID_DIR   = None # '/opt/local/likwid'

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
# eval options
#
######################################################################

opts = Variables( opts_file )
opts.Add( BoolVariable( 'fullmsg',  'enable full command line output',           fullmsg ) )
opts.Add( BoolVariable( 'debug',    'enable building with debug informations',   debug ) )
opts.Add( BoolVariable( 'profile',  'enable building with profile informations', profile ) )
opts.Add( BoolVariable( 'optimise', 'enable building with optimisation',         optimise ) )
opts.Add( BoolVariable( 'warn',     'enable building with compiler warnings',    warn ) )

opts.Add( ListVariable( 'programs',   'programs to build',                 'all', PROGRAMS   ) )
opts.Add( ListVariable( 'frameworks', 'parallelization frameworks to use', 'all', FRAMEWORKS ) )

# read options from options file
opt_env = Environment( options = opts )

fullmsg  = opt_env['fullmsg']
debug    = opt_env['debug']
profile  = opt_env['profile']
optimise = opt_env['optimise']
warn     = opt_env['warn']

programs   = Split( opt_env['programs'] )
frameworks = Split( opt_env['frameworks'] )

if 'all' in programs   : programs   = PROGRAMS
if 'all' in frameworks : frameworks = FRAMEWORKS

opts.Save( opts_file, opt_env )

######################################################################
#
# set up compilation environment
#
######################################################################

if debug :
    OPTFLAGS  = '-g -march=native'
    LINKFLAGS = '-g'
    DEFINES   = ''

if profile :
    OPTFLAGS  = '-g -pg -O3 -march=native'
    LINKFLAGS = '-g -pg'
    DEFINES   = ''

if warn :
    WARNFLAGS = readln( 'cpuflags --comp %s --warn' % CXX )
    
env = Environment( options    = opts,
                   ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS + ' ' + OPTFLAGS + ' ' + WARNFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   CPPDEFINES = Split( DEFINES ),
                   )

env.ParseConfig( os.path.join( HPRO_DIR, 'bin', 'hlib-config' ) + ' --cflags --lflags' )

if not fullmsg :
    env.Replace( CCCOMSTR   = " CC     $SOURCES" )
    env.Replace( CXXCOMSTR  = " C++    $SOURCES" )
    env.Replace( LINKCOMSTR = " Link   $TARGET"  )

if not debug :
    env.Append(  CPPDEFINES = [ "NDEBUG" ] )
    
env.Append(  CPPPATH = [ '#include' ] )
env.Prepend( LIBS    = [ "common" ] )
env.Prepend( LIBPATH = [ "." ] )

if JEMALLOC_DIR != None :
    env.Append( LIBPATH = os.path.join( JEMALLOC_DIR, 'lib' ) )
    env.Append( LIBS    = 'jemalloc' )

if LIKWID_DIR != None :
    env.Append( CPPDEFINES = 'LIKWID_PERFMON' )
    env.Append( CPPPATH    = os.path.join( LIKWID_DIR, 'include' ) )
    env.Append( LIBPATH    = os.path.join( LIKWID_DIR, 'lib' ) )
    env.Append( LIBS       = 'likwid' )

######################################################################
#
# target "options"
#
######################################################################

# def show_options ( target, source, env ):
#     print() 
#     print( 'Type  \'scons <option>=<value> ...\'  where <option> is one of' )
#     print()
#     print( '  Option   │ Description               │ Value' )
#     print( ' ──────────┼───────────────────────────┼──────' )
#     print( '  fullmsg  │ full command line output  │', opt_env['fullmsg'] )
#     print( '  debug    │ debug informations        │', opt_env['debug'] )
#     print( '  profile  │ profile informations      │', opt_env['profile'] )
#     print( '  optimise │ compiler optimisation     │', opt_env['optimise'] )
#     print( '  warn     │ compiler warnings         │', opt_env['warn'] )
#     print() 

# options_cmd = env.Command( 'phony-target-options', None, show_options )

# env.Alias( 'options', options_cmd )

######################################################################
#
# common library and framework dependent targets
#
######################################################################

common = env.StaticLibrary( 'common', [ 'src/apps/log_kernel.cc',
                                        'src/apps/matern_cov.cc',
                                        'src/apps/laplace.cc',
                                        'src/cluster/distr.cc',
                                        'src/cluster/h.cc',
                                        'src/cluster/hodlr.cc',
                                        'src/cluster/tileh.cc',
                                        'src/cluster/tlr.cc',
                                        'src/dag/graph.cc',
                                        'src/dag/local_graph.cc',
                                        'src/dag/node.cc',
                                        'src/dag/lu.cc',
                                        'src/matrix/level_matrix.cc',
                                        'src/utils/compare.cc',
                                        'src/utils/log.cc',
                                        'src/utils/term.cc' ] )

#
# default sequential environment
#

if 'seq' in frameworks :
    seq = env.Clone()
        
    if 'tlr'   in programs : seq.Program( 'tlr-seq.cc' )
    if 'hodlr' in programs : seq.Program( 'hodlr-seq.cc' )
    if 'tileh' in programs : seq.Program( 'tileh-seq.cc' )
    if 'dag'   in programs : seq.Program( 'dag-seq',  [ 'dag-seq.cc', 'src/seq/dag.cc' ] )

#
# OpenMP
#

if 'openmp' in frameworks :
    omp = env.Clone()
    omp.Append( CXXFLAGS  = "-fopenmp" )
    omp.Append( LINKFLAGS = "-fopenmp" )

    if 'tlr'   in programs : omp.Program( 'tlr-omp.cc' )
    if 'hodlr' in programs : omp.Program( 'hodlr-omp.cc' )
    if 'tileh' in programs : omp.Program( 'tileh-omp.cc' )
    if 'dag'   in programs : omp.Program( 'dag-omp', [ 'dag-omp.cc', 'src/omp/dag.cc' ] )

#
# TBB
#

if 'tbb' in frameworks :
    tbb = env.Clone()
    tbb.Append( CPPPATH = os.path.join( TBB_DIR, "include" ) )
    tbb.Append( LIBPATH = os.path.join( TBB_DIR, "lib" ) )

    if 'tlr'   in programs : tbb.Program( 'tlr-tbb.cc' )
    if 'hodlr' in programs : tbb.Program( 'hodlr-tbb.cc' )
    if 'tileh' in programs : tbb.Program( 'tileh-tbb.cc' )
    if 'dag'   in programs : tbb.Program( 'dag-tbb', [ 'dag-tbb.cc', 'src/tbb/dag.cc' ] )

#
# TaskFlow
#

if 'taskflow' in frameworks :
    tf = env.Clone()
    tf.MergeFlags( '-isystem ' + os.path.join( TASKFLOW_DIR, "include" ) )
    tf.Append( LIBS = [ "pthread" ] )
    
    if 'tlr'   in programs : tf.Program( 'tlr-tf.cc' )
    if 'hodlr' in programs : tf.Program( 'hodlr-tf.cc' )
    if 'dag'   in programs : tf.Program( 'dag-tf', [ 'dag-tf.cc', 'src/tf/dag.cc' ] )

#
# MPI
#

if 'mpi' in frameworks :
    mpi = env.Clone()
    mpi.ParseConfig( 'mpic++ --showme:compile' )
    mpi.ParseConfig( 'mpic++ --showme:link' )
    
    if 'tlr'   in programs :
        mpi.Program( 'tlr-mpi-bcast.cc' )
        mpi.Program( 'tlr-mpi-ibcast.cc' )
        mpi.Program( 'tlr-mpi-rdma.cc' )
    
    if 'tileh' in programs :
        mpi.Program( 'tileh-mpi-bcast.cc' )
        mpi.Program( 'tileh-mpi-ibcast.cc' )

#
# HPX
#

if 'hpx' in frameworks :
    hpx = env.Clone()
    hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags hpx_application" % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   hpx_application" % ( os.path.join( HPX_DIR, 'lib', 'pkgconfig' ) ) )
    hpx.Append( LIBS = [ "hpx_iostreams" ] )
    
    if 'tlr'   in programs : hpx.Program( 'tlr-hpx.cc' )
    if 'hodlr' in programs : hpx.Program( 'hodlr-hpx.cc' )
    if 'dag'   in programs : hpx.Program( 'dag-hpx', [ 'dag-hpx.cc', 'src/hpx/dag.cc' ] )

#
# GASPI
#

if 'gpi2' in frameworks :
    gpi = env.Clone()
    gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --cflags GPI2" % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.ParseConfig( "PKG_CONFIG_PATH=%s pkg-config --libs   GPI2" % ( os.path.join( GPI2_DIR, 'lib64', 'pkgconfig' ) ) )
    gpi.Append( LIBS = [ "pthread" ] )
    
    if 'tlr'   in programs : gpi.Program( 'tlr-gaspi.cc' )

    
