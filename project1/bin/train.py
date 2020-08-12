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

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import accuracy_score
from sklearn.preprocessing import StandardScaler

def grid_search( X, y, clf, params ):
  print( 'Params: %s'%( params ) )
  grid = GridSearchCV( clf, params ) 
  grid.fit( X, y )
  print( 'Best params: %s'%(grid.best_params_) )
  return grid

def svm_best_params( X, y, params=None ):
  if not params:
    # params
    params = {
      'C': [ 100 ],
    }
  clf = SVC( gamma='auto', class_weight='balanced', random_state=0 )
  grid = grid_search( X, y, clf, params )
  return grid

def rf_best_params( X, y, params=None ):
  if not params:
    # params
    params = {
      'max_depth': [ 20 ]
    }
  clf = RandomForestClassifier( n_estimators= 500, n_jobs=-1, random_state=0, class_weight='balanced' )
  grid = grid_search( X, y, clf, params )
  return grid

model_dict = {
    'rf': rf_best_params,
    'svm': svm_best_params
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--csv', help='features file' )
    parser.add_argument( '--p2_train', help='project2 training features (after split)' )
    parser.add_argument( '--gt', help='gt of project1' )
    parser.add_argument( '--model', help='output csv file to dump all features' )
    parser.add_argument( '--pkl', help='output csv file to dump all features' )
    parser.add_argument( '--bonus', help='output csv file to dump all features', default='y')
    args = parser.parse_args()
    
    if args.p2_train and args.bonus == 'n':
        print('Extracting only for training set...')
        p1 = pd.read_csv( args.csv, header=None )
        p1['name'] = p1[1].apply( lambda x: '_'.join( x.split('_')[:-1]) )
        train = pd.read_csv( args.p2_train )
        uid = train.uid.unique()
        df = p1[ p1['name'].isin( uid ) ]
    else:
        df = pd.read_csv( args.csv, header=None)
    print('Size: %s'%df.shape[0])
    gt = pd.read_csv(args.gt, header=None, names=['uid', 'gt'] )
    df = df.rename(columns={0:'filepath', 1:'uid'})
    df.uid = df.uid.apply(lambda x: x.replace('"', ''))
    gt.uid = gt.uid.apply(lambda x: x.replace('"', ''))
    df = df.fillna(0)
    df = pd.merge( df, gt, on='uid')
    cols = df.columns[ ~df.columns.isin(['filepath', 'uid', 'gt', 'name'])]
    X = df[cols]
    y = df['gt']
    print('Training...')
    func = model_dict.get( args.model )
    grid = func( X, y )
    obj = {
        'model': 'rf',
        'grid': grid
    }
    with open(args.pkl, 'wb') as f:
        joblib.dump( obj, f, compress=9 )    
    print('Done.')
        
if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( 'Finished in %s second(s).'%(round( finish-start, 2 )) )