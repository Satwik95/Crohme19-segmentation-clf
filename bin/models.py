import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import accuracy_score

def load_model( pkl ):
  with open( pkl, 'rb' ) as f:
    obj = joblib.load( f )
  return obj

def save_model( pkl, mod, grid ):
  print('Saving model..')
  obj = {
      'model': mod,
      'grid': grid,
  }
  # save model
  with open( pkl, 'wb' ) as f:
    joblib.dump( obj, f, compress=3 )
  return 

#####################################################
# grid search for different algorithms

def grid_search( X, y, clf, params ):
  print( 'Params: %s'%( params ) )
  grid = GridSearchCV( clf, params ) 
  grid.fit( X, y )
  print( 'Best params: %s'%(grid.best_params_) )
  return grid

def rf_best_params( X, y, params=None ):
  if not params:
    # params
    params = {
      'n_estimators': [ 100, 500, 1000, 2000, 3000 ],
      'max_depth': [ 10, 15, 20, 30 ]
    }
  clf = RandomForestClassifier( n_jobs=-1, random_state=0, class_weight='balanced' )
  grid = grid_search( X, y, clf, params )
  return grid

def svm_best_params( X, y, params=None ):
  if not params:
    # params
    params = {
      'C': [ 1.0, 50, 100, 200 ],
    }
  clf = SVC( gamma='auto', class_weight='balanced', random_state=0 )
  grid = grid_search( X, y, clf, params )
  return grid

def lg_best_params( X, y, params=None ):
  if not params:
    # paramsmodel = SVC(C=100, gamma='auto', class_weight='balanced')
    params = {
      'C': [ 1.0, 50, 100, 200 ],
    }
  clf = LogisticRegression( n_jobs=-1, class_weight='balanced', random_state=0 )
  grid = grid_search( X, y, clf, params )
  return grid
#####################################################

def predict( X, clf ):
  y_pred = clf.predict( X )
  return y_pred

def accuracy( y, y_pred ):
  acc = accuracy_score( y, y_pred )
  return acc

# for trying different models
model = {
    'rf': rf_best_params,
    'svm': svm_best_params,
    'lg': lg_best_params
}

# for excluding cols
exclude_cols = [
    'filepath', 
    'uid', 
    'gt', 
    'source', 
    'destination', 
    'src', 
    'dest', 
    'label'
]

