"""
@author: Nikunj Kotecha
         Satwik Mishra
"""

import argparse, os, sys
import joblib
import time
import pandas as pd
import traceback
import concurrent.futures
from tqdm import tqdm
import csv
import pdb
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from helper            import cols, rename_cols, Graph, Edge, get_information
from models            import load_model, exclude_cols, predict
from preprocess_helper import preprocess, get_traces

sys.path.append( './project1/bin' )
from p1_features_new import extract_features

def pred_edges( df, pkl, baseline='n' ):
  if baseline == 'y':
    print('Performing baseline segmentaiton...')
    df['edges'] = 0
    return

  # X for prediction and add to df under edges column
  feature_cols = df.columns[ ~df.columns.isin( exclude_cols ) ]
  X = df[ feature_cols ]
  # load_model
  obj = load_model( pkl )
  # predict
  grid = obj['grid']
  y_pred = predict( X, grid )
  df['edges'] = y_pred
  return

def create_graph( df ):
  '''
  graph class
  '''
  source = df['source'].values
  destination = df['destination'].values
  weight = df['edges'].values
  graph = Graph()
  for idx in range( source.size ):
    graph.add_edge( source[idx], destination[idx], weight[idx] )
    
  return graph

def segmentation( df ):
  try:
    # create graph and perform segmentation
    graph=create_graph( df )
    segments = graph.segmentation()
    return segments
  except Exception as e:
    print(str(e))
    print(print(traceback.format_exc()))
    
def project2( img_dir, df ):
  try:
    #####################################################
    # get segments
    segments = segmentation( df )
    #####################################################
    # get traces from file
    filepath = df['filepath'].unique()[0]
    uid = df['uid'].unique()[0]
    info = get_information( filepath )
    if uid != info[0]:
      print('Not same file')
    traces, traces_map = info[1], info[2]
    ## ...
    #orig = traces
    ##...
    
#     traces = get_traces( traces )
    traces = preprocess( traces, filepath )
    #####################################################
    # for each segment - (symbol) generate project 1 features
    #res = pd.DataFrame()
    res = []
    for segment in segments:
      #temp = project_1_features( filepath, uid, traces, traces_map, segment, img_dir=img_dir )
      temp = extract_features( filepath, uid, traces, traces_map, segment, img_dir=img_dir )
      #res = temp if res.size == 0 else res.append( temp )
      temp = [ filepath, uid, segment ] + temp
      res.append( temp )
    #####################################################
    # return the features for each segmentation of that file
    res = pd.DataFrame( data=res )
    return filepath, uid, res
  except Exception as e:
    print(traceback.format_exc())
    
def classification( results, pkl ):
  df = pd.DataFrame()
  for res in results:
    temp = res[2]
    df = temp if df.size==0 else df.append( temp )
  df = df.rename( columns={0:'filepath', 1:'uid',2:'segment'})
  cols = df.columns[ ~df.columns.isin(['filepath', 'uid', 'segment'])]
  X = df[cols]
  with open( pkl, 'rb' ) as f:
        obj = joblib.load( f )
  grid = obj['grid']
  y = grid.predict( X )
  df['pred'] = y
  return df

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument( 'csv', help='path to features csv file' )
  parser.add_argument( '--pkl_2', help='path to the model' )
  parser.add_argument( '--pkl_1', help='path to project 1 model' )
  parser.add_argument( '--out', help='path to output the prediction in csv', default='./data/results.csv' )
  parser.add_argument( '--test', help='path to output the prediction in csv', default='y' )
  parser.add_argument( '--img', help='path to output the prediction in csv', default='./data/img' )
  parser.add_argument( '--baseline', help='path to output the prediction in csv', default='n' )
  args = parser.parse_args()
   
  try:
    img_dir = args.img
    if not os.path.exists(img_dir):
      os.mkdir( img_dir )
    #####################################################
    # read csv
    df = pd.read_csv( args.csv )
    # rename cols if not exists
    if not cols[0] in df.columns:
      # read csv without first row as header
      df = pd.read_csv( args.csv, header=None )
      df = rename_cols( df, test=args.test )
    #####################################################
    #pdb.set_trace()
    #####################################################
    # predict
    print('Performing segmentation...')
    pred_edges( df, args.pkl_2, args.baseline )
    #####################################################
    
    #####################################################
    # get all files in dataframe and put in content
    #####################################################
    print('Gathering segments and removing features for each segment...')
    uid = df['uid'].unique()
    content = []
    for ui in uid:
      temp = df[ df['uid'] == ui ].copy()
      content.append( temp )
    fn = partial( project2, img_dir )
    with concurrent.futures.ProcessPoolExecutor() as executor:
      results = list( tqdm( executor.map( fn, content ), \
                           total=len( content ) ) )
    
    results = classification( results, args.pkl_1 )
    results[['filepath', 'uid', 'segment', 'pred']].to_csv( args.out, index=False )
    
  except Exception as e:
    print( 'Error: %s'%(str(e)) )
    print(traceback.format_exc())
  finally:
    pass
    

if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( 'Finished in %s second(s).'%(round( finish-start, 2 )) )