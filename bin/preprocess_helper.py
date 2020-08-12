"""
@author: Nikunj Kotecha
         Satwik Mishra
"""

import sys
import numpy as np
import traceback
import copy
import math

def get_traces( traces ):
  res = []
  for trace in traces:
    temp = np.array([])
    for t in trace.split(','):
      pts = [ float(val) for val in t.split() ]
      pts = np.asarray(pts)
      if temp.size == 0:
        temp = pts
      else:
        temp = np.vstack( [ temp, pts ] )
    res.append( temp )
  return res
######################################################################

######################################################################
# helper function to convert type of traces
######################################################################

def convert_tuple_to_list_numpy( traces ):
  res = []
  for trace in traces:
    trace = np.array( trace )
    res.append( trace )
  return res

def convert_numpy_to_list_tuple( traces ):
  #pdb.set_trace()
  res = []
  for trace in traces:
    t = []
    if trace.size == 2:
      t.append( (trace[0][0], trace[0][1]) )
    else:
      t = [ (t[0],t[1]) for t in trace ]
    res.append( t )
  return res

def convert_list_to_numpy( trace ):
  arr = np.array([])
  for t in trace:
    t = np.array( t )
    arr = t if arr.size==0 else np.vstack( (arr, t) )
  return arr
######################################################################

######################################################################
# preprocessing functions:
# - interpolation
# - sharp points
# - removing hooks
######################################################################
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_x(p1, p2, d, k):
    c = 0
    if p1[0]<p2[0]:
        c = math.sqrt(d**2/(k**2 +1))
    elif p1[0]>p2[1]:
        c = -math.sqrt(d**2/(k**2 +1))
    return p1[0] + c

def get_y(x, p1, p2, k, d):
    #print("getting y", x, p1, p2, k, d)
    if p1[0]==p2[0]:
        if p1[1]<p2[1]: return  p1[1] + d
        elif p1[1]>p2[1]: return  p1[1] - d
    return k*x + p1[1] - k*p1[0]

def get_slope(p1, p2):
    if (p1[0]-p2[0])!=0:
        return (p1[1]-p2[1])/(p1[0]-p2[0])
    return float('inf')

def remove_points(trace, queue):
    while queue:
        a = queue.pop(0)
        trace.pop(trace.index(a))
    return trace

def look_for_new_p2(trace, p1, d):
    #print("P1 is",p1)
    removal_points = [trace[0]]
    p2 = None
    for p in trace[1:]:
        cur_d = dist(p, p1)
        if cur_d>d:
            p2 = p
            break
        removal_points.append(p)
    return removal_points, p2

def interpolate(trace):
    #print(trace)
    lens = []
    temp_trace = copy.deepcopy(trace)
    final_trace = []
    for i in range(1, len(trace)):
        p1, p2 = trace[i - 1], trace[i]
        lens.append(dist(p1, p2))
    d = sum(lens) / len(lens)
    
    while len(temp_trace) > 1:
        # compute distance b/w p1, and p2
        p1, p2 = temp_trace.pop(0), temp_trace[0]
        seg_d = dist(p1, p2)
        if seg_d < d:
            removal_points, p2 = look_for_new_p2(temp_trace, p1, d)
            temp_trace = remove_points(temp_trace, removal_points)
        if not p2:
            return trace
        k = get_slope(p1, p2)
        if k==float('inf'): 
            new_x =p2[0]
            new_y = p2[1]
        else:
            new_x = get_x(p1, p2, d, k)
            new_y = get_y(new_x, p1, p2, k, d)
        final_trace += [p1, (new_x, new_y), p2]
    return final_trace

def sharp_points(trace):
    V = []
    slopes = []
    for i in range(len(trace)-1):
        p1, p2 = trace[i], trace[i+1]
        delta_x, delta_y = p2[0]-p1[0], p2[1] - p2[0]
        if delta_x==0: 
            slopes.append(float("inf"))
            continue
        slopes.append(math.degrees(get_slope(p1, p2)))
    i=1
    while i<len(slopes)-1:
        theta = slopes[i] - slopes[i+1] # alpha{i,i+1} - alpha{i+1,i+2}       
        if theta==0:
            i+=1
            continue
        prev_theta = (slopes[i-1] - slopes[i]) 
        delta_theta = theta - prev_theta # theta{i, i+1} - theta{i-1,i}
        if delta_theta<=0 and prev_theta!=0:
            #print(slopes[i],slopes[i+1],theta)
            V.append(trace[i+1]) # add p{i+1} i.e. p2 as p{i} is p1
        i+=1
    return V

def bb_diagonal_len(trace):
    trace = np.array(trace)
    w = max(trace[:,0]) - min(trace[:,0])
    h = max(trace[:,1]) - min(trace[:,1])
    return math.sqrt(h**2 + w**2)

def remove_hooks(sp, trace, th_angle=90, th_len_f=0.03):
    
    th_len = th_len_f*bb_diagonal_len(trace)
    #indexes = {}
    #for i in 
    hooks = []
    if len(sp)>2:
        
        seg_b, seg_e= (sp[0], sp[1]), (sp[-1], sp[-2]) # fisrt two, last 2 sharp points
        beta_b, beta_e = get_slope(seg_b[0], seg_b[1]), get_slope(seg_e[0], seg_e[1])
        
        l_seg_b, le_seg_e = dist(seg_b[0], seg_b[1]), dist(seg_b[0], seg_b[1])
        
        seg_b_plus_1, seg_e_minus_1 = (sp[1], sp[2]), (sp[-3], sp[-2])
        
        lambda_1 = abs(beta_b - get_slope(seg_b_plus_1[0], seg_b_plus_1[1]))
        lambda_2 = abs(beta_e - get_slope(seg_e_minus_1[0], seg_e_minus_1[1]))

        if l_seg_b<th_len or lambda_1<th_angle:
            for x, y in [sp[0], sp[1]]:
                if (x, y) not in hooks:
                    hooks.append((x, y))

        if l_seg_b<th_len or lambda_1<th_angle:
            for x, y in [sp[-2], sp[-1]]:
                if (x, y) not in hooks:
                    hooks.append((x, y))
    
    for hook in hooks:
        if hook in trace:
            index = trace.index(hook)
            trace.pop(index)
    return trace
######################################################################

######################################################################
# preprocess func
# - remoiving duplicates from the traces if any
######################################################################
def remove_duplicates( traces ):
  res = []
  for trace in traces:
    if not trace.size > 1:
      raise Exception( 'traces not found' )
    if trace.size == 2 or trace.size==3:
      trace = np.array([trace[0], trace[1]])
    else:
      trace = np.unique( trace[:, 0:2], axis=0 ) 
      if trace.size == 2:
        trace = np.array( [ trace[0][0], trace[0][1] ] )
    res.append( trace )
  return res
######################################################################

######################################################################
# preprocess func
# - normalization the traces as a whole
# - make note to not to normalize for individual traces
######################################################################
def normalize( traces ):
  #traces = convert_tuple_to_list_numpy( traces )
  # obtain max, min over all traces
  max_x, max_y = -sys.maxsize, -sys.maxsize
  min_x, min_y = sys.maxsize, sys.maxsize
  for trace in traces:
    # x cord
    if trace.size == 2:
      min, max = trace[0], trace[0]
    else:
      min, max = trace[:,0].min(), trace[:,0].max()
    if max_x < max:
      max_x = max
    if min_x > min:
      min_x = min
    
    # y cord
    if trace.size == 2:
      min, max = trace[1], trace[1]
    else: 
      min, max = trace[:,1].min(), trace[:,1].max()
    if max_y < max:
      max_y = max
    if min_y > min:
      min_y = min
  # obtain scaling factor
  scale = np.maximum( max_x-min_x, max_y-min_y )
    
  # go through each trace and normalize
  for trace in traces:
    if trace.size == 2:
      # only two points are there
      trace[0] = trace[0] - (max_x + min_x)/2
      trace[1] = trace[1] - (max_y + min_y)/2
      trace[0], trace[1] = trace[0]/scale, trace[1]/scale
      trace[0], trace[1] = trace[0] + 0.5, trace[1] + 0.5
    else:
      # center data to origin
      trace[:,0] = trace[:,0] - (max_x + min_x)/2
      trace[:,1] = trace[:,1] - (max_y + min_y)/2
      # scaling by same amount
      trace[:,0] = trace[:,0]/scale
      trace[:,1] = trace[:,1]/scale
      # shifting between 0, 1
      trace[:,0] = trace[:,0] + 0.5
      trace[:,1] = trace[:,1] + 0.5
  #traces = convert_numpy_to_list_tuple( traces )
  return traces
######################################################################

######################################################################
# preprocess func
# - smoothening of the all the traces
######################################################################
def smoothen( traces ):
  res = []
  for trace in traces:
    smooth = []
    if trace.size == 2:
      smooth.append( (trace[0], trace[1]) )
    else:
      for i in range( 0, trace.shape[0]):
        if i == 0:
          cur, next = trace[i], trace[i + 1]
          new = tuple(map(sum, zip(cur, next)))
          val = 2
        elif i == len(trace) - 1:
          prev, cur = trace[i - 1], trace[i]
          new = tuple(map(sum, zip(prev, cur)))
          val = 2
        else:
          prev, cur, next = trace[i - 1], trace[i], trace[i + 1]
          new = tuple(map(sum, zip(prev, cur, next)))
          val = 3
        smooth.append( (new[0]/val, new[1]/val) ) 
    res.append(smooth)
  return res
######################################################################

######################################################################
def preprocess( traces, filepath ):
    '''
    obtain the traces of a file and preprocess
    preprocess
     - interpolate
     - get sharp points
     - remove hooks
     - remove duplicates
     - normalize
     - smoothen
    The return value is a list of traces inside a tuple
    '''
    try:
        #pdb.set_trace()
        f_traces = []
        traces = get_traces( traces )
        #print(traces)
        for t in traces:
            #t = t.tolist()
            #pdb.set_trace()
            if len(t)>5:
                t = [ list( val[:2] ) for val in t ]
                new_t= interpolate(t)
                sp = sharp_points(new_t)
                t = remove_hooks(sp, new_t)
            else:
                t = [ list(t)[:2] ]
            t = convert_list_to_numpy( t )
            f_traces.append( t )
        f_traces = remove_duplicates( f_traces )
        f_traces = normalize( f_traces ) 
        f_traces = smoothen( f_traces )
        #f_traces = normalize( f_traces ) 
        return f_traces
    except Exception as e:
        print(traceback.format_exc())
        print(filepath)