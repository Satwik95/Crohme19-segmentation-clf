import argparse, os, sys
import time
import pandas as pd
import traceback
from ast import literal_eval
import pdb

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument( 'csv', help='path to results csv file' )
  parser.add_argument( 'lg', help='dir path where the lg file should be stored' )
  args = parser.parse_args()

  try:
    #####################################################
    lg_dir = args.lg
    if not os.path.exists( lg_dir ):
      os.mkdir( lg_dir )
    #####################################################
    df = pd.read_csv( args.csv )
    uid = df['uid'].unique()
    print(f'Writing lg files in {lg_dir} dir...')
    for ui in uid:
      temp = df[ df['uid']==ui ]
      filepath = temp['filepath'].values[0]
      lg_path, _ = os.path.splitext( filepath.split('/')[-1] )
      lg_path = os.path.join( lg_dir, '%s.lg'%lg_path )
      print( f'Writing to {lg_path}', end='\r', flush=True )
      with open( lg_path, 'w' ) as f:
        lines = '# IUD, %s\n# Objects(%s)'%( uid[0], temp.shape[0] )
        f.write( lines )
        map = {}
        for index, row in temp.iterrows():
          segment = literal_eval( row.segment )
          y = row.pred
          i = map[y]+1 if y in map else 1
          map[y] = i
          line = [ '\nO', '%s_%s'%(y, i), y, '1.0' ]
          for seg in segment:
            line += [ seg ]
          line = ', '.join( str(val) for val in line )
          f.write( line )

    print( '\nLg file(s) written.' )
  except Exception as e:
    print(traceback.format_exc())
    #pdb.set_trace()
if __name__ == "__main__":
  start = time.perf_counter()
  main()
  finish = time.perf_counter()
  print( 'Finished in %s second(s).'%(round( finish-start, 2 )) )