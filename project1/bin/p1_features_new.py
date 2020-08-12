import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from collections import defaultdict
import copy
import math
import cv2 as cv
from feature_func import *
import pdb

def convert_tuple_to_list_numpy( traces ):
  res = []
  for trace in traces:
    trace = np.array( trace )
    res.append( trace )
  return res

def extract_features( filepath, uid, traces, t_map, segment=None, img_dir='./img_p1' ):
    arr = []
    img_path = os.path.join( img_dir, uid )
    if segment != None:
        for seg in segment:
            trace = traces[ t_map.get(seg) ]
            arr.append( trace )
            img_path += '_%s'%seg
        traces = arr
    img_path += '.png' 
    n_strokes = 0
    combined = np.array([])
    plt.figure()
    traces = convert_tuple_to_list_numpy(traces)
    for t in traces:
        temp = np.array(t)[:,:2]
        if temp.size == 2:
            plt.scatter(temp[:,0], temp[:,1])
        else:
            plt.plot( temp[:,0], temp[:,1])
        combined = temp if combined.size==0 else np.vstack((combined, temp))
        n_strokes+=1
    plt.axis( 'off' )
    plt.savefig( img_path )
    plt.clf()

    n_traces=combined.shape[0]
    traces = combined
    # Mean X, Y & COV features
    mean_x = mean( traces[:,0] )
    mean_y = mean( traces[:, 1] )
    cov_xy = cov( traces[:, 0], traces[:, 1] )

    ##################################
    # features with the help of traces
    #  - no. of traces
    #  - no. of strokes
    #  - fuzzy_histo
    #################################

    # data for features
    data = [ n_traces, n_strokes ]
    ########################################
    # fuzzy features in the form of list
    fuzzy_histo = fuzzy_hist_2d( traces )
    ########################################

    ########################################
    # features with the help of the img
    #  - aspect_ratio
    #  - crossing features
    ########################################
    # read the img to send to functions
    img = cv.imread( img_path )
    ###############################################
    ar, mean_x_ar, mean_y_ar = get_aspect_ratio( img, img.shape[0], img.shape[1] )
    ###############################################

    # add data for ar
    data += [ar, mean_x_ar, mean_y_ar]
    h, w=200,200
    # resize the img for crossing features
    resize_img = cv.resize( img, dsize=(h, w), interpolation=cv.INTER_CUBIC )

    ######################################################
    # fast using numpy
    cross_features = fast_cross_features( resize_img[:,:,0] )
    for reg in cross_features:
        data += reg
    ######################################################

    # add fuzzy features to the data
    data += fuzzy_histo

    data += [ mean_x, mean_y, cov_xy ]
    return data
        
        
        
        