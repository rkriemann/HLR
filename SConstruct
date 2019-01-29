
import os

#
# set up compilation environment
#
env = Environment( ENV        = os.environ,
                   CXX        = 'mpic++ -std=c++17',
                   # CXXFLAGS   = Split( '-O3 -march=native' ),
                   # LINKFLAGS  = Split( '-lpthread' )
                   CXXFLAGS   = Split( '-g -march=native' ),
                   LINKFLAGS  = Split( '-g -lpthread -lboost_mpi' )
                   )

env.Replace( CCCOMSTR   = " CC     $SOURCES" )
env.Replace( CXXCOMSTR  = " C++    $SOURCES" )
env.Replace( LINKCOMSTR = " Link   $TARGET"  )

env.ParseConfig( 'hlibpro/bin/hlib-config --cflags --lflags' )

env.Program( 'hlrtest', [ 'hlrtest.cc',
                          'logkernel.cc',
                          'matern.cc',
                          'hodlr.cc',
                          'hodlr-seq.cc',
                          'hodlr-tbb.cc',
                          'tlr.cc',
                          'tlr-seq.cc',
                          'tlr-tbb.cc',
                          'tlr-mpi.cc' ] )

