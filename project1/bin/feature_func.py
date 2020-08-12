"""
@author: Nikunj Kotecha
         Satik Mishra
         
This file has the funcitons for feature extraction.
It is called in feature.py
It also has the feature columns in order for proper tracking.
"""

import numpy as np
import pandas as pd
import os, sys
###################################
# other import for features
from collections import defaultdict
import copy
import math
###################################

###################################################################################
# For fuzzy histo 2d

def compute_mem(p, c, w, h):
    return ((w - abs(p[0] - c[0])) * (h - abs(p[1] - c[1]))) / (w * h)

def fuzzy_helper(corner_points, temp):
    cp = np.array(copy.deepcopy(corner_points))
    val = np.absolute(corner_points - temp)
    max_point = np.argmin(val)
    val = np.delete(val, max_point)
    max = cp[max_point]
    cp = np.delete(cp, max_point)
    min_point = np.argmin(val)
    return cp[min_point], max

def fuzzy_hist_2d(trace, scale_factor=200, cells=4):
    memberships = defaultdict(list)
    step = scale_factor // cells
    corner_points = np.array([a for a in range(0, scale_factor + 1, step)])

    for i in corner_points:
        for j in corner_points:
            memberships[(i, j)] = []

    # h, w in the formula == step
    tot_points = len(trace)
    for x_p, y_p in trace:
        # print(x_p, y_p)
        temp_x = np.array([x_p] * (cells + 1))
        temp_y = np.array([y_p] * (cells + 1))

        if 0 <= x_p <= 50:
            min_x, max_x = 0, 50
        elif 150 <= x_p <= 200:
            min_x, max_x = 150, 200
        else:
            min_x, max_x = fuzzy_helper(corner_points, temp_x)

        if 0 <= y_p <= 50:
            min_y, max_y = 0, 50
        elif 150 <= y_p <= 200:
            min_y, max_y = 150, 200
        else:
            min_y, max_y = fuzzy_helper(corner_points, temp_y)

        cps = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]
        # print(cps)

        for x_c, y_c in cps:
            memberships[(x_c, y_c)].append(compute_mem((x_p, y_p), (x_c, y_c), step, step))

    return [sum(x) / tot_points for x in memberships.values()]
###############################################################

###################################################################
def normalize(trace, factor=200):
    min_x, max_x = np.min(trace[:, 0]), np.max(trace[:, 0])
    min_y, max_y = np.min(trace[:, 1]), np.max(trace[:, 1])
    res = []
    for p in trace:
        if max_x != min_x:
            x_res = ((p[0] - min_x) / (max_x - min_x)) * factor
        else:
            x_res = factor / 2
        if max_y != min_y:
            y_res = ((p[1] - min_y) / (max_y - min_y)) * factor
        else:
            y_res = factor / 2
        res.append((x_res, y_res))
    return np.array(res)

###################################################################

#############################################################
# Aspect ratio feature
def get_aspect_ratio(img, H, W):
    min_x, max_x, min_y, max_y = sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize
    for i in range(H):
        for j in range(W):
            if (sum(img[i][j]) // 3) < 255:
                # print("hi")
                min_x, max_x = min(min_x, j), max(max_x, j)
                min_y, max_y = min(min_y, i), max(max_y, i)
    # print(min_x, max_x, min_y, max_y)
    ratio = (max_x - min_x) / (max_y - min_y) if max_y > min_y else abs(max_x - min_x)
    mean_x, mean_y = (max_x + min_x) / 2, (max_y + min_y) / 2
    return ratio, mean_x, mean_y
#######################################################

##############################################################
# crossing feature
def avg_intersection_x(img, l, r, H):
    lines = []
    first = []
    last = []
    NO_LINES = 9
    k = math.floor((r - l) / NO_LINES)
    for j in range(l, r):
        count = 0
        if j % k == 0:
            pos_tracker = []
            for i in range(H):
                if (sum(img[i][j]) // 3) < 255:
                    count += 1
                    pos_tracker.append(i)
            if len(pos_tracker) >= 1:
                pos_tracker = pos_tracker[::-1]
                first.append(pos_tracker[0])
                last.append(pos_tracker[-1])
            lines.append(count)
    if sum(lines) == 0: return 0, -1, -1
    return sum(lines) // NO_LINES, sum(first) // NO_LINES, sum(last) // NO_LINES


def avg_intersection_y(img, l, r, W):
    lines = []
    first = []
    last = []
    NO_LINES = 9
    k = math.floor((r - l) / NO_LINES)
    for i in range(l, r):
        count = 0
        if i % k == 0:
            pos_tracker = []
            for j in range(W):
                if (sum(img[i][j]) // 3) < 255:
                    count += 1
                    pos_tracker.append(j)
            if len(pos_tracker) >= 1:
                first.append(pos_tracker[0])
                last.append(pos_tracker[-1])
            lines.append(count)
    if sum(lines) == 0: return 0, -1, -1
    return sum(lines) // NO_LINES, sum(first) // NO_LINES, sum(last) // NO_LINES

def crossing_features(img, H, W):
    a, count = 0, 0
    regions_vertical, first_pos_h, last_pos_h = [], [], []
    regions_horizontal, first_pos_v, last_pos_v = [], [], []
    step = H // 5
    while count != 5:
        i_h, f_pos_h, l_pos_h = avg_intersection_x(img, a, a + step, H)
        regions_horizontal.append(i_h)
        first_pos_h.append(f_pos_h)
        last_pos_h.append(l_pos_h)

        i_v, f_pos_v, l_pos_v = avg_intersection_y(img, a, a + step, W)
        regions_vertical.append(i_v)
        first_pos_v.append(f_pos_v)
        last_pos_v.append(l_pos_v)

        a += step
        count += 1

    return regions_vertical + regions_horizontal, first_pos_v + first_pos_h, last_pos_v + last_pos_h

##############################################################
# numpy fast_crossing features
##############################################################################
def fast_cross_features( img ): 
  row, col = np.where(img < 255 )
  region_size = img.shape[0]//5
  start = 0
  end = img.shape[0]
  res = []
  for reg in range( start, end, region_size ):
    space = region_size // 9
    count = 0
    tot_row, first_row, last_row = 0, 0, 0
    tot_col, first_col, last_col = 0, 0, 0

    for line in range( reg+space//2, reg+region_size -space, space ):
      count += 1
      intersection_col = col[np.where(row==line)]
      if intersection_col.size != 0:
        tot_row += intersection_col.size
        first_row += intersection_col[0]
        last_row += intersection_col[-1]
      intersection_row = row[np.where(col==line)]
      if intersection_row.size != 0:
        tot_col += intersection_row.size
        first_col += intersection_row[0]
        last_col += intersection_row[-1]

    if tot_row != 0:
      tot_row, first_row, last_row = tot_row//count, first_row//count, last_row//count
    else:
      first_row, last_row = -1, -1

    if tot_col != 0:
      tot_col, first_col, last_col = tot_col//count, first_col//count, last_col//count
    else:
      first_col, last_col = -1, -1
    res.append([tot_row,first_row,last_row, tot_col,first_col,last_col])
  return res
############################################################################

############################################################################
# Mean features of X, Y
# get the mean
def mean( arr ):
  m = np.sum( arr ) / arr.shape[0] 
  return m
############################################################################

############################################################################
# Cov features
# cov between att 1 and att 2
############################################################################
def cov( att1, att2 ):
  if att1.shape[0] == 1:
    cov = 0
  else:
    cov = np.divide( np.sum( np.dot( ( att1 - mean( att1 ) ), ( att2 - mean( att2 ) ) ) ), ( att1.shape[0] - 1 ) )
  return cov 
############################################################################

############################################################################
global_features = [ 'traces', 'strokes', 'aspect_ratio', 'mean_x_ar', 'mean_y_ar' ]

# add crossing features cols

cross_features_cols = ['v1', 'fv1', 'lv1', 'h1','fh1', 'lh1',\
                       'v2', 'fv2', 'lv2', 'h2','fh2', 'lh2', \
                       'v3', 'fv3', 'lv3', 'h3','fh3', 'lh3', \
                       'v4', 'fv4', 'lv4', 'h4','fh4', 'lh4', \
                       'v5', 'fv5', 'lv5', 'h5','fh5', 'lh5' ]


'''
cross_features_cols = ['v1', 'v2', 'v3', 'v4', 'v5', \
                        'h1', 'h2', 'h3', 'h4', 'h5', \
                        'fv1', 'fv2', 'fv3', 'fv4', 'fv5', \
                        'fh1', 'fh2', 'fh3', 'fh4', 'fh5', \
                        'lv1', 'lv2', 'lv3', 'lv4', 'lv5', \
                        'lh1', 'lh2', 'lh3', 'lh4', 'lh5']

'''
# fuzzy feature cols
fuzzy_features_cols = ['fz1', 'fz2', 'fz3', 'fz4', 'fz5', \
                       'fz6', 'fz7', 'fz8', 'fz9','fz10', \
                       'fz11', 'fz12', 'fz13', 'fz14', 'fz15', \
                       'fz16', 'fz17', 'fz18', 'fz19', 'fz20', \
                       'fz21', 'fz22', 'fz23', 'fz24', 'fz25']

# features added later on
add_features = ['mean_x', 'mean_y', 'cov']

feature_cols = global_features + cross_features_cols + fuzzy_features_cols + add_features

data_cols = ['filename', 'uid'] + feature_cols
############################################################################