"""
Container for data
"""
import io
import os
import re
import glob
import collections
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

    def _parseInputFile(self, infile):
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

    def parseInput(self, filepath):
        """
        Parse all the ranks data in filepath
        """
        files = glob.glob(os.path.join(filepath, 'nsys*txt'))
        for f in files:
            rank = re.search('r([\d]+)', f).group()
            nodes = re.search('([\d]+)_nodes', f).group()
            print('Parsing {}'.format(f))
            with open(f, 'r+', encoding="utf-8") as infile:
                # gets cudaapisum, gpukernsum, gpumemtimesum, gpumemsizesum
                parsed = self._parseInputFile(infile)
                
                # CUDA API sum
                self._hash[nodes][rank]['cudaapisum'] = pd.read_csv(
                    io.BytesIO(b'\n'.join(parsed[0])), sep='\s{2,}',  engine='python',
                    skiprows=3, header=None)
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
                self._hash[nodes][rank]['gpumemsizesum'].columns = ['Total',
                                                                    'Operations',
                                                                    'Average',
                                                                    'Minimum',
                                                                    'Maximum',
                                                                    'Operation']
                
        return
