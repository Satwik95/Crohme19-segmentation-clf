"""
@author: Nikunj Kotecha
         Satwik Mishra
"""

import argparse, os
import joblib
import time
import pandas as pd
import numpy as np
import math

from helper import rename_cols, Symbol

def kl_divergence( count, df1, df2, set1='train', set2='test' ):
  sum = 0
  for c in count:
    p = c.probability( set1, df1.shape[0] )
    q = c.probability( set2, df2.shape[0] )
    if q != 0:
      sum += p * math.log2( p / q )
  return sum

def count_symbols( df1, df2=None ):
  symbols = df1['src'].unique()
  count = []
  for symbol in symbols:
    train = df1[ df1['src']==symbol].shape[0]
    test = df2[ df2['src']==symbol].shape[0] if isinstance(df2, pd.DataFrame ) else 0 
    count.append( Symbol( symbol, train, test ) )
  return sorted( count )

def train_test_split( data, cols, split=0.3 ):
  '''
  count frequency of every symbol 
  split files for symbols with least frequency
  measure distribution with kl divergence
  '''
  test = pd.DataFrame()
  train = pd.DataFrame()
  print(f'Tot: {data.shape[0]}')
  # copy & sort
  count = count_symbols( data )
  #print(count[0].symbol)
  while len( count ) > 0:
    c = count[0]
    #print(f'src: {c.symbol}, train: {train.shape[0]}, test: {test.shape[0]}, data:  {data.shape[0]}, c: {len(count)}', end='\r', flush=True)
    uid = list( data[ (data['src']==c.symbol) ]['uid'].unique() )
    #pdb.set_trace()
    while len(uid) > 0:
      if len( uid ) > 2:
        for i in range(3):
          id = uid[0]
          temp = data[data['uid'] == id]
          index = temp.index
          if i == 2:
            test = temp if test.size==0 else test.append(temp) 
          else:
            train = temp if train.size==0 else train.append(temp)
          uid.remove(id)
          data = data.drop( index )
          
      else:
        break
    if len( uid ) > 0 and len( uid ) < 3:
      # append to train
      ids = uid
      for _ in range(len(ids)):
        id = uid[0]
        temp = data[data['uid'] == id]
        index = temp.index
        train = temp if train.size==0 else train.append(temp)
        uid.remove(id)
        data = data.drop( index )
    #pdb.set_trace()
    count = count_symbols( data )
    
  # kl divergence
  count = count_symbols( train, test )
  train_test_divergence = kl_divergence( count, train, test )
  test_train_divergence = kl_divergence( count, test, train, set1='test', set2='train' )
  print( f'\nTrain KL divergence: {train_test_divergence}' )
  print( f'Test KL divergence: {test_train_divergence}' )
  return train, test

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument( 'features', help='features file for segmentation' )
  parser.add_argument( 'train', help='path where to store train_features.csv' )
  parser.add_argument( 'test', help='path where to store test_features.csv' )
  parser.add_argument( '--split', help='split % in float', default=0.3 )
  args = parser.parse_args()

  
  # read the data
  data = pd.read_csv( args.features, header=None )
  data = rename_cols( data )

  print('Splitting...')
  train, test = train_test_split( data, args.split )
  print( f"Data uid: {data['uid'].unique().size}" )
  print( f"Training uid: {train['uid'].unique().size}" )
  print( f"Testing uid: {test['uid'].unique().size}" )
  print('\nSaving training and testing features..')
  train.to_csv(  args.train, index=False )
  test.to_csv( args.test, index=False )

if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( f'Finished in { round( finish-start, 2 ) } second(s).' )