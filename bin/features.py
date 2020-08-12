import numpy as np
import math
import os, sys
from collections import defaultdict
from scipy.spatial import distance
import traceback
import pdb

from helper import Edge, kruskal

######################################################################
# helper functions for features
######################################################################
def get_bb_centre(t):
  y_min, y_max = min(np.array(t)[:, 1]), max(np.array(t)[:, 1])
  x_min, x_max = min(np.array(t)[:, 0]), max(np.array(t)[:, 0])
  # shift centre
  return x_min + ((x_max - x_min) / 2), y_min + ((y_max - y_min) / 2)

def get_bb_centres(traces):
  centres = []
  for t in traces:
      centres.append(get_bb_centre(t))
  return centres

def dist(v1, v2):
  return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)
######################################################################

######################################################################
# parallelity
######################################################################

def parallelity(t1, t2):
  v1, v2 = np.array(t1[len(t1) - 1]) - np.array(t1[0]), np.array(t2[len(t2) - 1]) - np.array(t2[0])
  norm, dot_prod = np.linalg.det([v1, v2]), np.dot(v1, v2)
  angle = np.math.atan2(norm, dot_prod)
  return np.degrees(angle)
######################################################################

######################################################################
# dist between centers of bounding box features
# vertical distance
# horizontal distance
######################################################################
def dist_bw_bb_centre(t1, t2):
  centres = get_bb_centres([t1, t2])
  return dist(centres[0], centres[1])


# vertical distance
def v_dist_bw_bb_centre(t1, t2):
  centres = get_bb_centres([t1, t2])
  return centres[0][1] - centres[1][1]

# horizontal
def h_dist_bw_bb_centre(t1, t2):
  centres = get_bb_centres([t1, t2])
  return centres[0][0] - centres[1][0]

# hori dist b/w last point of first stroke and 
# start point of second stroke
def horizontal_offset(t1, t2):
  
  return t1[-1][0] - t2[0][0]

# vert dist b/w last point of first stroke and 
# start point of second stroke
def vertical_offset(t1, t2):
  return t1[-1][1] - t2[0][1]
######################################################################

######################################################################
# writing slope
# angle with the +ve x axis, made by the vector given by 
# last point of first stroke and first point of second stroke
######################################################################
def writing_slope(t1, t2):
  
  v1, v2 = np.array([1, 0]), np.array(t1[-1]) - np.array(t2[0])
  norm, dot_prod = np.linalg.det([v1, v2]), np.dot(v1, v2)
  angle = np.math.atan2(norm, dot_prod)
  return np.degrees(angle)
######################################################################

######################################################################
# distance between centroids
######################################################################
def dist_bw_bb_centroids(t1, t2):
  traces = [t1, t2]
  centroids = []
  x_c, y_c = None, None
  for trace in traces:
    trace = np.array(trace)
    # print(trace)
    centroids.append((np.mean(trace[:, 0]), np.mean(trace[:, 1])))
  return dist(centroids[0], centroids[1])
######################################################################

######################################################################
# obtain shape context features
######################################################################
def p2p(points1, points2, angle=False, dist=False):
  if angle: 
    return distance.cdist(points1, points2, lambda p, q: math.atan2((p[1] - q[1]), (p[0] - q[0])))
  if dist:
    return distance.cdist(points1, points2)

# distance histo
def dist_histo(dist, t, no_bins, dist_bins): 
  
  dist_hist = [0]*no_bins
  for d in dist:
    if dist_bins[0]<= d <dist_bins[1]: dist_hist[0]+=1
    elif dist_bins[1]<= d <dist_bins[2]: dist_hist[1]+=1
    elif dist_bins[2]<= d <dist_bins[3]: dist_hist[2]+=1
    elif dist_bins[3]<= d <dist_bins[4]: dist_hist[3]+=1
    elif dist_bins[4]<= d: dist_hist[4]+=1
  return dist_hist

# angle histo
def angle_histo(t, angles, no_bins_theta, angle_bins):
    
  angle_hist = [0]*no_bins_theta
  for a in angles:
    if angle_bins[0]<= a <angle_bins[1]: angle_hist[0]+=1
    elif angle_bins[1]<= a <angle_bins[2]: angle_hist[1]+=1
    elif angle_bins[2]<= a <angle_bins[3]: angle_hist[2]+=1
    elif angle_bins[3]<= a <angle_bins[4]: angle_hist[3]+=1
    elif angle_bins[4]<= a <angle_bins[5]: angle_hist[4]+=1
    elif angle_bins[5]<= a <angle_bins[6]: angle_hist[5]+=1
    elif angle_bins[6]<= a <angle_bins[7]: angle_hist[6]+=1
    elif angle_bins[7]<= a <angle_bins[8]: angle_hist[7]+=1
    elif angle_bins[8]<= a <angle_bins[9]: angle_hist[8]+=1
    elif angle_bins[9]<= a <angle_bins[10]: angle_hist[9]+=1
    elif angle_bins[10]<= a <=angle_bins[11]: angle_hist[10]+=1
    elif angle_bins[11]<= a <=angle_bins[12]: angle_hist[11]+=1
    elif angle_bins[12]<= a: angle_hist[12]+=1
  return angle_hist

# log polar
def log_polar(dist, angles, t, no_bins, no_bins_theta):   
  log_polar, _, _ = np.histogram2d(dist, angles, bins=(no_bins, no_bins_theta))
  return log_polar

# shape content
def shape_context(t1, t2, no_bins=5, angle_step_size=12, r_min=0.1, r_max=2.0):
  c = get_bb_centres([t1, t2])
  # find the overall cenres for the bb
  t = t1+t2
    
  dist_bins = np.logspace(np.log10(r_min), np.log10(r_max), no_bins)
    
  angle_bins = np.arange(0, 390, 360//angle_step_size)
  no_bins_theta = len(angle_bins)
    
  bb_x, bb_y = get_bb_centre(t)
  bb_centre = [(bb_x, bb_y)]
    
  dist, angles = p2p(t, bb_centre, dist=True), p2p(t, bb_centre, angle=True)
    
  dist = np.nan_to_num(dist)
  if dist.mean()!=0:
    dist = dist/dist.mean()
        
  norm_angle = angles.mean()
    
  angles = (angles - norm_angle* (np.ones((len(t), 1))))
  angles = np.degrees(angles)%360
  angles += (angles < 0)*360
    
  dist, angles = dist.flatten(), angles.flatten()
    
  d_hist = dist_histo(dist, t, no_bins, dist_bins)
  angle_hist = angle_histo(t, angles, no_bins_theta, angle_bins)
  sc = log_polar(dist, angles, t, no_bins, no_bins_theta)
   
  return d_hist, angle_hist, list(sc.flatten())
######################################################################

######################################################################
# extract the features of the edge between one trace id and another
######################################################################
def extract_features(t1, t2):
  features = []
  features.append(parallelity(t1, t2))
  features.append(dist_bw_bb_centre(t1, t2))
  features.append(h_dist_bw_bb_centre(t1, t2))
  features.append(v_dist_bw_bb_centre(t1, t2))
  features.append(horizontal_offset(t1, t2))
  features.append(vertical_offset(t1, t2))
  features.append(writing_slope(t1, t2))
  features.append(dist_bw_bb_centroids(t1, t2))

  d, a, sc = shape_context(t1, t2)
  f = list(d)+list(a)+list(sc)
  for c in f:
    features.append(c)
    
  return features
######################################################################

######################################################################
# helper function to obtain the traceGroup in the req format
######################################################################
def get_traceGroup_dict( traceGroup ):
  res = {}
  for tg in traceGroup:
    for s in tg.get('strokes'):
      res[int(s)] = {
          'label': tg.get( 'label' ),
          'label_gt': tg.get( 'label_gt' )
      }
  return res
######################################################################

######################################################################
# get the uid, traces, tracegroup and obtain the features
######################################################################
def get_features( filepath, uid, traces, traces_map, traceGroup=None ):
  '''
  create a graph and obtain a mst of the trace ids
  the edges are weighted using distance as a measure
  obtain the features of these edges
  if tracegroup mentioned obtain its gt
  
  return is a 2d list where each row represents
  inidividual features for each edge
  '''
  centres = get_bb_centres(traces)
  ids = list( traces_map.keys() )
  graph = []
  for i in range( len(ids)-1 ):
    for j in range( i+1, len(ids) ):
      # get centre of i
      trace = traces[ traces_map.get(ids[i]) ]
      centre_i = get_bb_centre( trace )
      # get centre of j
      trace = traces[ traces_map.get(ids[j]) ]
      centre_j = get_bb_centre( trace )
      # distance between the centres
      distance = dist( centre_i, centre_j )
      # create an edge
      edge = Edge( ids[i], ids[j], distance )
      graph.append( edge )
  mst = kruskal( graph, ids )
  ######################################################################
    
  if traceGroup:
    traceGroup = get_traceGroup_dict( traceGroup )
  rows = []
  for edge in mst:
    source = edge.source
    dest = edge.dest
    weight = edge.weight
    ######################################################################
    # gt
    if traceGroup:
      tg_source = traceGroup.get( source )
      tg_dest = traceGroup.get( dest )
      if not tg_source or not tg_dest:
        continue
      tg_source_label = tg_source.get('label')
      tg_dest_label = tg_dest.get('label')
      tg_src_gt = tg_source.get('label_gt')
      tg_dest_gt = tg_dest.get('label_gt')

      if not tg_source_label or not tg_dest_label:
        continue
      if tg_source_label == '' and tg_dest_label == '':
        return []
      gt = 1 if tg_source_label == tg_dest_label else 0
      #print( f'Source: {source}, Dest: {dest}, tg_src:{tg_source_label}, tg_dest:{tg_dest_label}, gt:{gt}' )
    ######################################################################
    # features
    traces_src = traces[source]
    traces_dest = traces[dest]
    features = extract_features( traces_src, traces_dest )
    row = [ filepath, uid, source, dest ]
    if traceGroup:
      row += [ gt, tg_src_gt, tg_dest_gt ]
    row += features
    rows.append( row )
  return rows