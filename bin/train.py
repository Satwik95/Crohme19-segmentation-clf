"""
@author: Nikunj Kotecha
         Satwik Mishra
"""

import argparse, os, sys
import joblib
import time
import pandas as pd

from models import *

def training( X_train, y_train, model_name, bonus='n', pkl=None ):
  func = model.get( model_name )
  if bonus == 'n':
    print(f'Training {model_name}...')
    grid = func( X_train, y_train )
  if bonus == 'y' and pkl:
    print(f'Bonus training {model_name}...')
    obj = load_model( pkl )
    params = obj['grid'].best_params_
    # converting params to list
    d = {}
    for key in params.keys():
      val = params[key]
      d[key] = [ val ]
    grid = func( X_train, y_train, d )
  return grid
     
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument( 'train', help='path where to store train_features.csv' )
  parser.add_argument( 'test', help='path where to store test_features.csv' )
  parser.add_argument( 'model', help='which model to run best_params on' )
  parser.add_argument( 'pkl', help='path to save the pickle object' )
  parser.add_argument( '--bonus', help='whether to train on all data for bonus', default='n' )
  args = parser.parse_args()

  print('\nReading files...')
  train = pd.read_csv( args.train )
  test = pd.read_csv( args.test )
  pkl = args.pkl
  cols = train.columns[ ~train.columns.isin(exclude_cols) ]

  if args.bonus == 'y':
    try:
      # train model for bonus -> all dataset
      train = train.append( test )
      X = train[cols]
      y = train['gt']
      # training..
      grid = training( X, y, args.model, args.bonus, pkl )
      name, ext = os.path.splitext( pkl )
      pkl = name + '_bonus' + ext
    except Exception as e:
      print(f'Error {str(e)}')
      sys.exit(1)
    
  else:
    X_train, X_test = train[cols], test[cols]
    y_train, y_test = train['gt'], test['gt']
    try:
      # training..
      grid = training( X_train, y_train, args.model )
      y_pred = predict( X_train, grid )
      acc = accuracy( y_train, y_pred )
      print(f'\nTraining accuracy of {args.model}: {acc}')
      
      # testing..
      y_pred = predict( X_test, grid )
      acc = accuracy( y_test, y_pred )
      print(f'Testing accuracy of {args.model}: {acc}')
    except Exception as e:
      print(f'Error: {str(e)}')
      sys.exit(1)
        
  save_model( pkl, args.model, grid ) 
    
if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( f'Finished in { round( finish-start, 2 ) } second(s).' )