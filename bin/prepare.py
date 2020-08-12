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

from helper            import get_information
from preprocess_helper import preprocess
from features          import get_features


def information( filepath ):
  '''
  for every inkml file, read with beautfilsoup
  obtain information:
  uid <- annotation
  traces <- trace
  traceGroups <- traceGroup
  
  preprocess & features
   - interpolate
   - remove hooks
   - remove duplicates
   - normalize
   - smoothen
   
  features:
   - parallelity
  '''
  try:
    ######################################################
    # remove informaiton from file
    ######################################################
    result = get_information( filepath )
    if not result:
      print( f'No traces found for: {filepath}' )
      return
    if len(result) == 3:
      uid, traces, traces_map = result
    if len(result) == 4:
      uid, traces, traces_map, traceGroup = result

    ######################################################
    # preprocess & get features
    ######################################################
    traces = preprocess( traces, filepath )
    if len(result) == 3:
      rows = get_features( filepath, uid, traces, traces_map )
    else:   
      rows = get_features( filepath, uid, traces, traces_map, traceGroup )
    ######################################################
    if not rows:
      # some files have only one trace id..hence no res
      return
    return rows
  except Exception as e:
    print( 'Error for %s: %s'%(filepath, str(e) ) )
    
 
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument( 'dir', nargs='+', \
                      help='path(s) to directory \
                      where all the inkml files are present. \
                      If multiple paths then enter with a space' )
  parser.add_argument( '--csv', help='output csv file to dump all features' )
  args = parser.parse_args()
  ######################################################
  print('Gathering all file paths...')
  path = []
  if not os.path.isdir( args.dir[0] ):
    name, ext = os.path.splitext( args.dir[0] )
    if not ext == '.inkml':
        print('Inkml file or dir not present')
        sys.exit(1)
    path.append( args.dir[0] )
  else:
    for idx in range( len(args.dir) ):
      dir = args.dir[idx]
      files = os.listdir( dir )
      for f in files:
        if not os.path.splitext( f )[-1] == '.inkml':
          continue
        filepath = os.path.join( dir, f )
        path.append( filepath )
  ######################################################
  print('Gathering traces and generating features...')
  with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list( tqdm( executor.map( information, path ), total=len(path) ) )
  ######################################################
  print('Writing features to csv file')
  # write to csv file
  if args.csv:
    with open( args.csv, 'a' ) as f:
      writer = csv.writer(f, delimiter=',')
      counts = 0
      for idx in range(len(results)):
        res = results[idx]
        if not res:
          #print(idx)
          continue
        for row in res:
          writer.writerow( row )
          counts += 1
    print( 'Results %s viewed and saved.'%counts  )
  ######################################################
  return

if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( 'Finished in %s second(s).'%(round( finish-start, 2 )) )