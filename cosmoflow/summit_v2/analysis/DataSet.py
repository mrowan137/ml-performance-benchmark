"""
Container for data
"""
import io
import os
import re
import glob
import collections
import numpy as np
import pandas as pd

    
class DataSet():
    def __init__(self, filepath=None):
        self._hash = self._makehash()
        return

    def _makehash(self):
        return collections.defaultdict(self._makehash)

    def __getitem__(self, args):
        """
        Allows to set data like d[0][1][2][3] = 1.0
        """
        return self._hash.__getitem__(args)

    def _parseInputFileNsys(self, infile):
        """
        Parse the profiling data into recorded traces
        Return profiling data for:
            cudaapisum, gpukernsum, gpumemtimesum, gpumemsizesum
        """
        res = []
        # (cudaapisum,
        #  gpukernsum,
        #  gpumemtimesum,
        #  gpumemsizesum) = b'', b'', b'', b''
        
        inRecordingMode = False
        for line in infile:
            line = line.encode('utf-8')
            if not inRecordingMode:
                if line.find(b'Total') != -1:
                    inRecordingMode = True
                    tmp = []
                    tmp.append(line)
            elif inRecordingMode and line == b'\n':
                inRecordingMode = False
                res.append(tmp)
            else:
                tmp.append(line)
        return res
    
    def _parseInputFileNccl(self, infile):
        """
        Parse the profiling data
        Return profiling data for:
            ncclAllReduce counts
        """
        # Info to parse the nccl debug output
        int_to_datatype = {0: 'ncclInt8', 1: 'ncclUint8', 2: 'ncclInt32', 3: 'ncclUint32', 4: 'ncclInt64',
                           5: 'ncclUint64', 6: 'ncclFloat16', 7: 'ncclFloat32', 8: 'ncclFloat64'}
        int_to_op = {0: 'ncclSum', 1: 'ncclProd', 2: 'ncclMax', 3: 'ncclMin'}

        res = []
        
        for line in infile:
            ncclAllReduce = re.search('NCCL INFO AllReduce', line)
            if ncclAllReduce: 
                data = re.search('count ([\d]+)(.*?)datatype ([\d]+)(.*?)op ([\d]+)', line)
                count, int_datatype, int_op = np.array(data.group().split(' '))[1::2].astype(int)
                res.append([int_to_op[int_op], int_to_datatype[int_datatype], count])
            
        return res
    
    def parseInput(self, filepath, inputType='nsys'):
        if inputType == 'nsys': self.parseInputNsys(filepath, mode='nodes')
        elif inputType == 'nccl': self.parseInputNccl(filepath)
        return
            
    def parseInputNsys(self, filepath, mode='nodes'):
        """
        Parse all the ranks data in filepath
        """
        files = glob.glob(os.path.join(filepath, 'nsys*txt'))
        for f in files:
            rank = re.search('r([\d]+)', f).group()
            if mode == 'batch':
                nodes = re.search('batchsize_([\d]+)', f).group()
            elif mode == 'nodes':
                nodes = re.search('([\d]+)_nodes', f).group()
                
            print('Parsing {}'.format(f))
            with open(f, 'r+', encoding="utf-8") as infile:
                # gets cudaapisum, gpukernsum, gpumemtimesum, gpumemsizesum
                parsed = self._parseInputFileNsys(infile)
                
                # CUDA API sum
                self._hash[nodes][rank]['cudaapisum'] = pd.read_csv(
                    io.BytesIO(b'\n'.join(parsed[0])), sep='\s{2,}',  engine='python',
                    skiprows=3, header=None)
                try:
                    self._hash[nodes][rank]['cudaapisum'].columns = ['Time(%)',
                                                                     'Total Time (ns)',
                                                                     'Num Calls',
                                                                     'Average',
                                                                     'Minimum',
                                                                     'Maximum',
                                                                     'StdDev',
                                                                     'Name']
                except ValueError:
                    self._hash[nodes][rank]['cudaapisum'].columns = ['Time(%)',
                                                                     'Total Time (ns)',
                                                                     'Num Calls',
                                                                     'Average',
                                                                     'Minimum',
                                                                     'Maximum',
                                                                     'Name']
                
                # GPU kern sum
                for i in range(len(parsed[1])):
                    parsed[1][i] = parsed[1][i].decode('utf-8')
                    
                self._hash[nodes][rank]['gpukernsum'] = pd.read_csv(
                    io.StringIO('\n'.join(parsed[1])), sep='\s{2,}', engine='python', 
                    skiprows=3, header=None)
                try:
                    self._hash[nodes][rank]['gpukernsum'].columns = ['Time(%)',
                                                                     'Total Time (ns)',
                                                                     'Instances',
                                                                     'Average',
                                                                     'Minimum',
                                                                     'Maximum',
                                                                     'StdDev',
                                                                     'Name']
                except ValueError:
                    self._hash[nodes][rank]['gpukernsum'].columns = ['Time(%)',
                                                                     'Total Time (ns)',
                                                                     'Instances',
                                                                     'Average',
                                                                     'Minimum',
                                                                     'Maximum',
                                                                     'Name']

                # GPU mem time sum
                self._hash[nodes][rank]['gpumemtimesum'] = pd.read_csv(
                    io.BytesIO(b'\n'.join(parsed[2])), sep='\s{2,}', engine='python',
                    skiprows=3, header=None)
                try:
                    self._hash[nodes][rank]['gpumemtimesum'].columns = ['Time(%)',
                                                                    'Total Time (ns)',
                                                                    'Operations',
                                                                    'Average',
                                                                    'Minimum',
                                                                    'Maximum',
                                                                    'StdDev',
                                                                    'Operation']
                except ValueError:
                    self._hash[nodes][rank]['gpumemtimesum'].columns = ['Time(%)',
                                                                    'Total Time (ns)',
                                                                    'Operations',
                                                                    'Average',
                                                                    'Minimum',
                                                                    'Maximum',
                                                                    'Operation']

                # GPU mem size sum
                self._hash[nodes][rank]['gpumemsizesum'] = pd.read_csv(
                    io.BytesIO(b'\n'.join(parsed[3])), sep='\s{2,}', engine='python',
                    skiprows=3, header=None)
                try:
                    self._hash[nodes][rank]['gpumemsizesum'].columns = ['Total',
                                                                    'Operations',
                                                                    'Average',
                                                                    'Minimum',
                                                                    'Maximum',
                                                                    'StdDev',
                                                                    'Operation']
                except ValueError:
                    self._hash[nodes][rank]['gpumemsizesum'].columns = ['Total',
                                                                    'Operations',
                                                                    'Average',
                                                                    'Minimum',
                                                                    'Maximum',
                                                                    'Operation']
                
        return

    
    def parseInputNccl(self, filepath, mode='nodes'):
        """
        Parse all the ranks data in filepath
        """
        files = glob.glob(os.path.join(filepath, 'nccl*'))
        for f in files:
            rank = int(re.search('r([\d]+)', f).group().replace('r',''))
            nodes = int(re.search('([\d]+)_nodes', f).group().replace('_nodes', ''))
            #print('Parsing {}')
            print('Parsing {}'.format(f), '; {} rank, {} nodes'.format(rank, nodes))
            
            with open(f, 'r+') as infile:
                # get the counts for each function
                
                # hash will store hash[rank]['ncclAllReduce'] --> [count1, count2, count3, ...]
                # Then we could plot histogram of ranks to count per ncclAllReduce (convert to size),
                # with error bars
                parsed = self._parseInputFileNccl(infile)
                
                for p in parsed:
                    op, dtype, count = p
                    if not self._hash[nodes][rank][op][dtype]:
                        #print('create a list for rank {}'.format(rank),
                        #     '\n filepath={}'.format(infile))
                        self._hash[nodes][rank][op][dtype] = []
                        
                    self._hash[nodes][rank][op][dtype].append(count)
                
        return
