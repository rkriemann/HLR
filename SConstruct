
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

mpi = Environment( ENV        = os.environ,
                   CXX        = 'mpic++ -std=c++17',
                   CXXFLAGS   = Split( CXXFLAGS ),
                   LINKFLAGS  = Split( LINKFLAGS + ' -lboost_mpi' ),
                   )

env.ParseConfig( 'hlibpro/bin/hlib-config --cflags --lflags' )
mpi.ParseConfig( 'hlibpro/bin/hlib-config --cflags --lflags' )

if not fullmsg :
    env.Replace( CCCOMSTR   = " CC     $SOURCES" )
    env.Replace( CXXCOMSTR  = " C++    $SOURCES" )
    env.Replace( LINKCOMSTR = " Link   $TARGET"  )
    mpi.Replace( CCCOMSTR   = " CC     $SOURCES" )
    mpi.Replace( CXXCOMSTR  = " C++    $SOURCES" )
    mpi.Replace( LINKCOMSTR = " Link   $TARGET"  )

common = env.StaticLibrary( 'common', [ 'logkernel.cc', 'matern.cc', 'tlr.cc', 'hodlr.cc' ] )

env.Prepend( LIBS    = [ "common" ] )
env.Prepend( LIBPATH = [ "." ] )
mpi.Prepend( LIBS    = [ "common" ] )
mpi.Prepend( LIBPATH = [ "." ] )

env.Program( 'tlr-seq', [ 'tlr-seq.cc' ] )
env.Program( 'tlr-tbb', [ 'tlr-tbb.cc' ] )
mpi.Program( 'tlr-mpi', [ 'tlr-mpi.cc' ] )

env.Program( 'hodlr-seq', [ 'hodlr-seq.cc' ] )
env.Program( 'hodlr-tbb', [ 'hodlr-tbb.cc' ] )
