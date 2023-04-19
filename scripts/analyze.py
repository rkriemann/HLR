#!/usr/bin/env python3

import pdb
import sys, os
import re
import getopt

######################################################################
#
# global settings
#
######################################################################

output_format = 'text'

# mapping from program names to actual algorithm
name_mapping = { 'approx-mm'  : 'H-Matrix Mult.',
                 'approx-lu'  : 'H-LU Factorization',
                 'uniform-mm' : 'Uniform-H-Matrix Mult.',
                 'uniform-lu' : 'Uniform-H-LU Factorization',
                 
                 'laplace'    : 'Laplace-SLP',
                 'materncov'  : 'Matérn Covariance',

                 'refblas'    : 'RefBlas',
                 'blis'       : 'BLIS',
                 'amdblis'    : 'AMD-BLIS',
                 'openblas'   : 'OpenBLAS',
                 'mkl'        : 'MKL' }

######################################################################
#
# colorization
#
######################################################################

# default color codes
colors = { 'reset'     : '\033[0m',
           'bold'      : '\033[1m',
           'underline' : '\033[4m',
           'italic'    : '\033[3m',
           'red'       : '\033[31m',
           'green'     : '\033[32m',
           'yellow'    : '\033[33m',
           'blue'      : '\033[34m',
           'purple'    : '\033[35m',
           'cyan'      : '\033[36m',
           'gray'      : '\033[37m' }

# no colors if wanted or output is not a terminal ('dumb' is for emacs)
if not sys.stdout.isatty() or os.environ['TERM'] == 'dumb' :
    for key in colors.keys() :
        colors[key] = ''
else :
    # try to handle above codes on non-supported systems
    try:
        import colorama

        colorama.init()
    except :
        pass

#
# return given string added with given style
#
def add_style ( text, style ) :
    return style + text + colors['reset']

######################################################################
#
# function for reading logfiles
#
######################################################################

#
# regular expessions
#
proc_re  = re.compile( 'processor : (.*)' )
arith_re = re.compile( '(standard|accumulator|lazy)' )
apx_re   = re.compile( '(Hpro|SVD|PairSVD|RRQR|RandSVD|RandLR|ACA|Lanczos)' )
time_re  = re.compile( ' in *([0-9]\.[0-9]{3}e[+-][0-9]{2}) s' )
time2_re = re.compile( ' runtime *= *([0-9]\.[0-9]{3}e[+-][0-9]{2}) s / ([0-9]\.[0-9]{3}e[+-][0-9]{2}) s / ([0-9]\.[0-9]{3}e[+-][0-9]{2}) s' )
mem_re   = re.compile( 'mem *= *([0-9]+\.[0-9]+) (MB|GB).*' )
err_re   = re.compile( 'error *= *([0-9]\.[0-9]{4}e[+-][0-9]{2})' )

#
# read in data from log file
#
def read_data ( filename ) :
    db   = {}
    proc = None
    
    with open( filename ) as f :
        proc  = ''
        arith = ''
        apx  = ''
        for line in f :
            # determine processor
            m = proc_re.search( line )
            if m != None:
                proc = m.group(1)
                proc = proc.replace( 'AMD', '' )
                proc = proc.replace( 'Intel', '' )
                proc = proc.lstrip().rstrip();
                
            # determine arithmetic
            m = arith_re.search( line )
            if m != None: arith = m.group(1);

            # determine approximation
            m = apx_re.search( line )
            if m != None: apx = m.group(1);

            # timing, memory and error
            # if arith != '' and apx != '' :
            #     # initialise DB
            #     if arith not in db :
            #         db[arith] = {}
                    
            #     if apx not in db[arith] :
            #         db[arith][apx] = { 'time'  : None,
            #                            'mem'   : None,
            #                            'error' : None }

            # timing
            m = time_re.search( line )
            if m != None:
                db['time'] = float(m.group(1))

            m = time2_re.search( line )
            if m != None:
                db['time'] = float(m.group(1)) # use best

            # memory
            m = mem_re.search( line )
            if m != None:
                if   m.group(2) == 'MB' : db['mem'] = int(round(float(m.group(1))))
                elif m.group(2) == 'GB' : db['mem'] = int(float(m.group(1)) * 1000)
                
            # error
            m = err_re.search( line )
            if m != None:
                db['error'] = float(m.group(1))

    return proc, db

######################################################################
##
## eval command line
##

opts, args = getopt.getopt( sys.argv[1:], 'ho:', [ 'help', 'output=' ] )
for o, a in opts:
    if o in ( '-o', '--output' ):
        if a not in [ 'text', 'html', 'pdf' ] :
            print( 'unsupported output format: ', a )
            print( 'supported formats: text, html, pdf' )
            sys.exit( 1 )
        output_format = a
    elif o in ( '-h', '--help' ):
        print( 'analyze: [options] logfile1 logfile2 ...' )
        print( 'options:' )
        print( '   -h, --help   : show this info' )
        print( '   -o, --output : output format (text,html,pdf)' )
        sys.exit( 0 )

# rest are logfiles
logfiles = args

######################################################################
##
## read all logs
##

DB = {}

# naming scheme for logfiles:  app--program--framework--blas.log
logfile_re  = re.compile( '(\w*)--([\w-]*?)--(\w*)--(\w*).*.log' )
# logfile_re  = re.compile( '(\w*)--.*.log' )

for logfile in logfiles :
    # print( 'reading ', logfile )

    m = logfile_re.search( logfile )
    if m != None:
        app    = m.group(1)
        prog   = m.group(2)

        # adjust names
        if app in name_mapping.keys() :
            app = name_mapping[app]
            
        if prog in name_mapping.keys() :
            prog = name_mapping[prog]
            
        fwork  = m.group(3)
        blas   = m.group(4)
        
        proc, db = read_data( logfile )

        if proc  not in DB                  : DB[proc]                   = {}
        if app   not in DB[proc]            : DB[proc][app]              = {}
        if prog  not in DB[proc][app]       : DB[proc][app][prog]        = {}
        if fwork not in DB[proc][app][prog] : DB[proc][app][prog][fwork] = {}
        
        DB[proc][app][prog][fwork][blas] = db
    
# print( DB )

######################################################################
##
## collect various data
##

min_time = {}

#
# processors/applications/frameworks
#
processors   = []
applications = []
programs     = []
frameworks   = []
blass        = []

for proc in DB.keys() :
    if proc not in processors :
        processors.append( proc )
        
    for app in DB[proc].keys() :
        if app not in applications :
            applications.append( app )
        
        for prog in DB[proc][app].keys() :
            
            if prog not in programs :
                programs.append( prog )
                
            for fwork in DB[proc][app][prog].keys() :

                if fwork not in frameworks :
                    frameworks.append( fwork )

                for blas in DB[proc][app][prog][fwork].keys() :

                    if blas not in blass :
                        blass.append( blas )
                        
processors   = sorted( processors )
applications = sorted( applications )
programs     = sorted( programs )
frameworks   = sorted( frameworks )
blass        = sorted( blass )

print( 'processors found:   ', ', '.join( processors ) )
print( 'applications found: ', ', '.join( applications ) )
print( 'programs found:     ', ', '.join( programs ) )
print( 'frameworks found:   ', ', '.join( frameworks ) )
print( 'BLAS found:         ', ', '.join( blass ) )

#
# minimal runtimes per processor
#
for proc in processors :
    min_time[proc] = {}

    for app in applications :
        min_time[proc][app] = {}

        for prog in programs :
            min_time[proc][app][prog] = {}
        
            for fwork in frameworks :

                if fwork not in DB[proc][app][prog].keys() :
                    continue
                
                min_time[proc][app][prog][fwork] = None

                tmin = None
            
                for blas in blass :

                    if not blas in DB[proc][app][prog][fwork].keys() :
                        continue
                    
                    if tmin == None : tmin = DB[proc][app][prog][fwork][blas]['time']
                    else :            tmin = min( tmin, DB[proc][app][prog][fwork][blas]['time'] )
                
                min_time[proc][app][prog][fwork] = tmin

#
# overall minimal runtimes per app/program/framework
#
for app in applications :
    min_time[app] = {}

    for prog in programs :
        min_time[app][prog] = {}
        
        for fwork in frameworks :

            min_time[app][prog][fwork] = None

            tmin = None

            for proc in processors :
                if fwork not in min_time[proc][app][prog].keys() :
                    continue

                if tmin == None : tmin = min_time[proc][app][prog][fwork]
                else :            tmin = min( tmin, min_time[proc][app][prog][fwork] )
                
            min_time[app][prog][fwork] = tmin


######################################################################
##
## print results
##
######################################################################

##################################################
#
# text output
#
##################################################

if output_format == 'text' :
    min_time_style = colors['bold'] + colors['red']

    # header and format strings for processors
    proc_len = max( [ len(s) for s in processors ] )
    proc_hdr = '{:─<{n}}'.format( '', n = proc_len+2 )
    proc_spc = '{: <{n}}'.format( '', n = proc_len+2 )
    proc_fmt = '%%%ds' % proc_len
                       
    for app in applications :

        if app != applications[0] :
            print()
            
        print( colors['bold'] + colors['blue'], end = '' )
        for i in range( len(app) ) :
            print( '━', end = '' )
        print()
            
        print( f'{app}' )

        for i in range( len(app) ) :
            print( '━', end = '' )
        print( colors['reset'] )
        print()
        
        for prog in programs :

            tophdr  = '───────────┬' + proc_hdr + '╥'
            midhdr  = '───────────┼' + proc_hdr + '╫'
            for blas in blass[:-1] :
                tophdr  += '──────────┬'
                midhdr  += '──────────┼'
            tophdr  += '──────────╥──────────'
            midhdr  += '──────────╫──────────'

            print( add_style( f'{prog}', colors['bold'] ) )
            print( tophdr )
            print( ' Framework │' + ' {0:^{n}} '.format( 'CPU', n = proc_len ) + '║', end = '' )
            for blas in blass[:-1] :
                print( ' {0:^8} │'.format( name_mapping[blas] ), end = '' )
            print( ' {0:^8} ║    Best  '.format( name_mapping[blass[-1]] ) )
            print( midhdr )

            old_fwork = ''
            for fwork in frameworks :

                old_proc = ''
                for proc in processors :

                    if fwork not in DB[proc][app][prog].keys() :
                        continue

                    # local timing data
                    data      = DB[proc][app][prog][fwork]
                    min_proc  = min_time[proc][app][prog][fwork]
                    min_total = min_time[app][prog][fwork]
                    
                    # header
                    if fwork == old_fwork :
                        print( "           │", end = '' )
                    else :
                        if old_fwork != '' :
                            print( '\b' + midhdr ) 
                        print( " %9s │" % fwork, end = '' )
                        old_fwork = fwork
        
                    if proc == old_proc :
                        print( proc_spc + '║', end = '' )
                    else :
                        print( ' {0:{n}} '.format( proc, n = proc_len ) + '║', end = '' )
                        old_proc = proc

                    # style for data entries
                    style      = {}
                    best_style = ''

                    for blas in blass :
                        if not blas in data.keys() :
                            continue

                        style[blas] = ''
                        if data[blas]['time'] == min_proc :
                            style[blas] = min_time_style

                        if data[blas]['time'] == min_total :
                            best_style = min_time_style

                    # actual data
                    for blas in blass[:-1] :
                        if not blas in data.keys() :
                            print( '          │', end = '' )
                        else :
                            style = min_time_style if data[blas]['time'] == min_proc else ''
                            print( add_style( '%8.2fs │' % data[blas]['time'], style ), end = '' )

                    style = min_time_style if data[blass[-1]]['time'] == min_proc else ''
                    print( add_style( '%8.2fs ║' % data[blass[-1]]['time'], style ), end = '' )

                    is_best = False
                    for blas in blass :
                        if not blas in data.keys() :
                            continue

                        if data[blas]['time'] == min_total :
                            is_best = True
                            break

                    style = min_time_style if is_best else ''
                    print( add_style( "%8.2fs " % min_proc, style ) )

            if prog != programs[-1] :
                print()

##################################################
#
# HTML output
#
##################################################

if output_format == 'html' :

    for app in applications :

        print()
        print( f'### {app}' )
        print()

        old_prog = ''
        for prog in programs :

            if prog != programs[0] :
                print()
            print( f'#### {prog}' )
            print()
            
            print( '<table style="width: 80%;">' )
            print( '  <tr>' )
            print( '    <th>Framework</th>' )
            print( '    <th style="border-right: 2px solid black;">CPU</th>' )
            for blas in blass :
                print( '    <th>%s</th>' % name_mapping[ blas ] )
            print( '    <th style="border-left: 2px solid black">Best</th>' )
            print( '  </tr>' )

            old_fwork = ''
            for fwork in frameworks :

                old_proc = ''
                for proc in processors :

                    if fwork not in DB[proc][app][prog].keys() :
                        continue
                    
                    if (( old_prog  != '' and prog  != old_prog  ) or
                        ( old_fwork != '' and fwork != old_fwork )) :
                        print( '  <tr style="border-top: 3px solid #888a85;">' )
                    else :
                        print( '  <tr>', end = '' )
                    
                    # header
                    if fwork == old_fwork :
                        print( '    <td></td>', end = '' )
                    else :
                        print( f'    <td>{fwork}</td>', end = '' )
                        old_fwork = fwork
        
                    if proc == old_proc :
                        print( '    <td style="border-right: 2px solid black;"></td>', end = '' )
                    else :
                        print( f'    <td style="border-right: 2px solid black;">{proc}</td>', end = '' )
                        old_proc = proc

                    for blas in blass :
                        if DB[proc][app][prog][fwork][blas]['time'] == min_time[proc][app][prog][fwork] :
                            print( '    <td style="text-align: right; color: red; font-weight: bold;">%.2fs</td>' % ( DB[proc][app][prog][fwork][blas]['time'] ), end = '' )
                        else :
                            print( '    <td style="text-align: right;">%.2fs</td>' % ( DB[proc][app][prog][fwork][blas]['time'] ), end = '' )

                    if min_time[proc][app][prog][fwork] == min_time[app][prog][fwork] :
                        print( '    <td style="border-left: 2px solid black; text-align: right; color: red; font-weight: bold;">%.2fs</td>' % ( min_time[proc][app][prog][fwork] ), end = '' )
                    else :
                        print( '    <td style="border-left: 2px solid black; text-align: right;">%.2fs</td>' % ( min_time[proc][app][prog][fwork] ), end = '' )
                        
                    print( '  </tr>' )

            print( '</table>' )

##################################################
#
# PDF output via matplotlib
#
##################################################

if output_format == 'pdf' :

    import matplotlib as mpl

    theme = { 'text.usetex'           : False,
              'font.family'           : 'Roboto Condensed',
              'figure.facecolor'      : "#ffffff",
              'figure.titlesize'      : 16,
              'axes.titlesize'        : 20,
              'axes.titlelocation'    : 'center',
              'axes.labelsize'        : 16,
              'axes.labelcolor'       : "#000000",
              'axes.facecolor'        : "#fdfdf6",
              'lines.markeredgewidth' : 0.0,
              'lines.markersize'      : 0,
              'lines.markeredgecolor' : 'none',
              'lines.linewidth'       : 3,
              'legend.fontsize'       : 16,
              'legend.framealpha'     : 1.0,
              'xtick.labelsize'       : 16,
              'ytick.labelsize'       : 16,
              'figure.figsize'        : ( 8, 7 )
             }
    
    mpl.use( 'Agg' )
    mpl.rcParams.update( theme )     

    annotate_size = 16

    import matplotlib.pyplot as plt
    from   matplotlib.backends.backend_pdf import PdfPages

    colorscheme = [ '#000000', # black
                    '#cc0000', # scarletred2
                    '#3465a4', # skyblue2
                    '#4e9a06', # chameleon3
                    '#f57900', # orange2
                    '#75507b', # plum2
                    '#c17d11', # chocolate2
                    '#edd400', # butter2
                    '#204a87', # faded aqua
                   ]
    blas_colors = { 'mkl'      : colorscheme[1],
                    'blis'     : colorscheme[2],
                    'amdblis'  : colorscheme[8],
                    'openblas' : colorscheme[3],
                    'refblas'  : colorscheme[4] }
    nfig = 1
    
    for app in applications :
        for prog in programs :
            for fwork in frameworks :
                
                plt.figure( nfig )
                nfig += 1

                ndata = 0
                for proc in processors :
                    if fwork in DB[proc][app][prog].keys() :
                        for blas in blass :
                            if blas in DB[proc][app][prog][fwork].keys() :
                                ndata += 1
                        ndata += 1 # one extra per processor

                if ndata == 0 :
                    continue

                fig, axs = plt.subplots( len(processors), sharex = True, figsize = ( 8, 0.5 * ndata ) )

                nplot = 0
                for proc in processors :

                    if fwork not in DB[proc][app][prog].keys() :
                        continue
                
                    pos     = 0
                    y_pos   = []
                    y_label = []
                    data    = DB[proc][app][prog][fwork]
                    for blas in blass :
                        if not blas in data.keys() :
                            continue
                        
                        val  = data[blas]['time']
                        rect = axs[nplot].barh( pos, val, color = blas_colors[blas], align = 'center', label = blas )
                        axs[nplot].annotate( '%.2f' % val,
                                             xy         = ( val, pos ),
                                             xytext     = ( 2, -annotate_size // 3),
                                             textcoords = "offset points",
                                             ha         = 'left',
                                             va         = 'baseline',
                                             color      = 'black',
                                             fontsize   = annotate_size )
                        y_pos.append( pos )
                        y_label.append( name_mapping[blas] )
                        pos += 1
                    pos += 1

                    axs[nplot].grid( axis = 'x', which = 'major', linestyle = 'dashed', linewidth='0.5',  color='#888a85' )
                    axs[nplot].set_axisbelow( True )
                    axs[nplot].set_title( proc )
                    axs[nplot].set_yticks( y_pos )
                    axs[nplot].set_yticklabels( y_label )
                    axs[nplot].invert_yaxis()  # labels read top-to-bottom
                    nplot += 1

                plt.tight_layout()
                filename = app + '--' + prog + '--' + fwork + '.pdf'
                filename = filename.replace( ' ', '_' )
                pp = PdfPages( filename )
                pp.savefig()
                pp.close()
                
                # show best times per processor
                ypos   = []
                ylabel = []
                pos    = 0
                
                plt.figure( nfig )
                nfig += 1
                fig, ax = plt.subplots( 1, sharex = True, figsize = ( 8, 0.66 * len(processors) ) )
                
                for proc in processors :
                    
                    if not fwork in min_time[proc][app][prog].keys() :
                        continue
                    
                    color = colorscheme[0]
                    data  = DB[proc][app][prog][fwork]
                    for blas in blass :
                        if not blas in data.keys() :
                            continue
                        if min_time[proc][app][prog][fwork] == data[blas]['time'] :
                            color = blas_colors[blas]
                            break
                    
                    val  = min_time[proc][app][prog][fwork]
                    rect = ax.barh( pos, val, color = color, align = 'center', label = proc )
                    ax.annotate( '%.2f' % val,
                                         xy         = ( val, pos ),
                                         xytext     = ( 2, -annotate_size // 3),
                                         textcoords = "offset points",
                                         ha         = 'left',
                                         va         = 'baseline',
                                         color      = 'black',
                                         fontsize   = annotate_size )
                    y_pos.append( pos )
                    y_label.append( proc )
                    pos += 1
                pos += 1
                ax.grid( axis = 'x', which = 'major', linestyle = 'dashed', linewidth='0.5',  color='#888a85' )
                ax.set_axisbelow( True )
                # ax.set_title( 'Best per Processor' )
                ax.set_yticks( y_pos )
                ax.set_yticklabels( y_label )
                ax.invert_yaxis()  # labels read top-to-bottom
                    
                plt.tight_layout()
                filename = app + '--' + prog + '--' + fwork + '--best.pdf'
                filename = filename.replace( ' ', '_' )
                pp = PdfPages( filename )
                pp.savefig()
                pp.close()
                
