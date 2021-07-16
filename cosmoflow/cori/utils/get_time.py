"""
Script to extract average time per epoch
Usage:
    python get_time.py log.out 5
This computes the average of first 5 epoch times using data from log.out
"""
import numpy as np
import re
import sys

f = sys.argv[1]
stop = int(sys.argv[2])

res = []

for line in open(f):
    match = re.search('- (\d+)s -', line)
    if match:
        res.append(int(match.group(1)))
        print(match.group(1))

print('len(res) = {}'.format(len(res)))
print('mean epoch time [0:{}] = {}'.format(stop, np.mean(res[0:stop])))
      
