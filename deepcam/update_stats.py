#!/usr/bin/env python
import h5py 
import sys
import os

argc=len(sys.argv)
if argc > 1:
  count=int(sys.argv[1])
  print("Setting count to : ", count)
else:
  count="ls train/* | wc -l" 
file_name = 'stats.h5'
f = h5py.File(file_name, 'r+')     # open the file
c=f['climate/count']
c[...]=count
f.close()

