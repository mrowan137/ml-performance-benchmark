"""
Plot profiling data
"""
import numpy as np
import matplotlib.pyplot as plt
import DataSet


# Load the data
files = ['../profiling_results/real/1_nodes/j_742215/',
         '../profiling_results/real/2_nodes/j_742893/',
         '../profiling_results/real/4_nodes/j_743227/',
         '../profiling_results/real/8_nodes/j_743827/',
         '../profiling_results/real/16_nodes/j_742961/']

d = DataSet.DataSet()
for f in files:
    d.parseInput(f)

# CUDA API sum time
# GPU KERN TIME sum
# GPU MEM  TIME sum
# GPU MEM SIZE sum
#
# Plots
# 1) points can be avg ovr all ranks, with candlebar for stdev and whisker for min/max
#     exp routine 1        exp routine 2        exp routine 3
#              x
#            x x
#          x x x                ...                  ...   
# time   x x x x 
#      x x x x x 
#      1 2 4 8 16
#
#      for each API --> 3 plot
#
# GPU KERN TIME
nRoutines = 10
all_nodes = list(d._hash.keys())[1:]
expensiveRoutines = d['2_nodes']['r0']['gpukernsum']['Name'][:nRoutines].values

plt.ion()
for routine in expensiveRoutines:
    expensiveRoutineTimesSum = [ np.array( [np.array(d[nodes][r]['gpukernsum'].loc[d[nodes][r]['gpukernsum']['Name'] == routine]['Total Time (ns)'])
                                             for r in d[nodes].keys()] ).sum() for nodes in all_nodes]
    # expensiveRoutineTimesPctMean = np.array([[d[nodes][r]['gpukernsum']['Time(%)'].loc(d[nodes][r]['gpukernsum']['Name'] == expensiveRoutineName)
    #                                           for r in d[nodes].keys()] for nodes in all_nodes]).mean(axis=1)
    # expensiveRoutineTimesPctMin = np.array([[d[nodes][r]['gpukernsum']['Time(%)'].loc(d[nodes][r]['gpukernsum']['Name'] == expensiveRoutineName)
    #                                          for r in d[nodes].keys()] for nodes in all_nodes]).max(axis=1)
    # expensiveRoutineTimesPctMax = np.array([[d[nodes][r]['gpukernsum']['Time(%)'].loc(d[nodes][r]['gpukernsum']['Name'] == expensiveRoutineName)
    #                                          for r in d[nodes].keys()] for nodes in all_nodes]).min(axis=1)

    plt.title(expensiveRoutineName)
    plt.bar(all_nodes, expensiveRoutineTimesSum)
    r=input('enter')
    plt.close('all')
#[d['1_nodes'][r]['gpukernsum']['Time(%)'].sum() for r in d['1_nodes'].keys()]

# GPU MEM TIME
#[d['1_nodes'][r]['gpukernsum']['Time(%)'].sum() for r in d['1_nodes'].keys()]




#d['1_nodes']['r0']['cudaapisum']
#print(d['2_nodes']['r0']['cudaapisum'])

#for prefix, the_type, value in ijson.parse(open(filepath)):
#    print(prefix, the_type, value)

# Data format brainstorming
# data[1,2,4,8,16][r0, r1, r2, ...][cuda,kernels, ...][func] --> time
#     [file      ][rank           ][API              ][func]

# 2)    1 node               2 nodes              4 nodes        ...
#
#
#  10 most exp. routines
#  raw times and %? for each API --> 3 plot
#
# 3) for nodes compare relative compute/transfer (points or mean w/ stdev candlebar w/ whisker min/max)
#
#    --> one plot
