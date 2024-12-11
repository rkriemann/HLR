#!/usr/bin/env python3

#
# read rank data
#

import sys, csv

histfile = 'ranks.hst'

if len(sys.argv) > 1 : histfile = sys.argv[1]

ranks  = []
count  = []
count0 = []

with open( histfile, 'r' ) as f :
    reader = csv.reader( f, delimiter = ' ' )
    for row in reader :
        ranks.append( int(row[0]) )
        count.append( int(row[1]) )
        if int(row[1]) != 0 :
            count0.append( int(row[1]) )

#
# plot histogram
#

import matplotlib.pyplot               as     plt
from   matplotlib.backends.backend_pdf import PdfPages

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import math

plt.figure()
ax = plt.subplot( 1, 1, 1 )

ax.grid( which = 'major', linestyle = '-', linewidth = '0.5',  color = '#888a85' )
ax.grid( which = 'minor', linestyle = ':', linewidth = '0.25', color = '#d3d7cf' )

ax.set_xlabel( 'rank' )
ax.set_xlim( -1, max(ranks)+1 )
# ax.set_xticks( range(len(dims)), [ "$2^{%d}$" % math.log2(d) for d in dims ] )

ax.set_ylabel( '#blocks' )
ax.set_yscale( 'log', base = 10 )
ymin = 10**( math.floor( math.log10( max( count0 ) ) ) )
ymax = 10**( math.ceil(  math.log10( max( count  ) ) ) )
ax.set_ylim( ymin, ymax )
yticks = []
y = ymin
while y <= ymax :
    yticks.append( y )
    y *= 10
print( yticks )
ax.set_yticks( yticks, [ str(y) for y in yticks ], minor = False )

ax.bar( ranks, count, color = 'skyblue', edgecolor = 'black' )

plt.tight_layout()

pp = PdfPages( 'ranks.pdf' )
pp.savefig()
pp.close()
plt.close()
