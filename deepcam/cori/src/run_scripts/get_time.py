import numpy as np
import re
import sys

f = sys.argv[1]

res = []

for line in open(f):
    match_start = re.search('run_start', line)
    if match_start:
        start = float(line.split(" ")[4].replace(",", ""))
        print(start)

    match_stop = re.search('run_stop', line)
    if match_stop: 
        stop = float(line.split(" ")[4].replace(",", ""))
        print(stop)

#print('len(res) = {}'.format(len(res)))
#print('mean epoch time [-8:] = {}'.format(np.mean(res[-8:])))
print('time per epoch (ms): {}'.format((stop - start)/5.))
      
