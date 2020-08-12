"""
@author: Nikunj Kotecha
         Satwik Mishra
"""

import argparse, os, sys
import time
import concurrent.futures
from tqdm import tqdm
from bs4 import BeautifulSoup as Soup
import csv
import pdb
import warnings
warnings.filterwarnings("ignore")


sys.path.append('../2/bin/')

from helper            import get_information
from preprocess_helper import preprocess, get_traces
from features          import get_features
from p1_features import *

def information( filepath ):
    try:
        info = get_information( filepath )
        if len(info) > 2:
            uid, traces, t_map = info[0], info[1], info[2]
        else:
            return
        traces = preprocess( traces, filepath )
        features = extract_features( uid, traces, t_map )
        return filepath, uid, features
    except Exception as e:
        print(str(e))
        return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dir', help='dir where project 1 data is') 
    parser.add_argument( '--csv', help='output csv file to dump all features' )
    args = parser.parse_args()

    if not os.path.exists('./img'):
        os.mkdir('./img')
    path = []
    files = os.listdir( args.dir )
    for file in files:
        name, ext = os.path.splitext(file)
        if ext=='.inkml':
            filepath=os.path.join( args.dir, file )
            path.append(filepath)
#     res = information( path[100] )
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list( tqdm( executor.map( information, path ), total=len(path) ) )

    with open( './p1_features.csv', 'w' ) as f:
        writer = csv.writer(f, delimiter=',')
        for r in results:
            if not r:
                continue
            filepath, uid, features = r
            row = [filepath, uid] + features
            writer.writerow( row )
    print('done') 
if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( 'Finished in %s second(s).'%(round( finish-start, 2 )) )