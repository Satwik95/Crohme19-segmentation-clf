from bs4 import BeautifulSoup as Soup

def get_information( filepath ):
  '''
  for every inkml file, read with beautfilsoup
  obtain information:
  uid <- annotation
  traces <- trace
  traceGroups <- traceGroup
  '''
  try:
    ######################################################
    # remove informaiton from file
    ######################################################
    search=[ 'annotation', 'trace', 'traceGroup' ]
    with open( filepath, 'rb' ) as f:
      soup = Soup( f, 'lxml-xml' )
    uid = None
    traces, traceGroup = [], []
    traces_map = {}
    for tag in soup.find_all( search ):
      # obtain uid
      if tag.name == 'annotation':
        if tag['type'] == 'UI':
          uid = tag.text
      # obtain traces
      if tag.name == 'trace':
        traces.append( tag.text )
        id = int( tag['id'] )
        idx = len(traces) - 1
        traces_map[id] = idx
      # obtain traceGroup if exists
      if tag.name == 'traceGroup':
        for t in tag.find_all( 'traceGroup' ):
          temp = { 'label_gt': '',
                   'strokes': [],
                   'label': ''
                 }
          for a in t.find_all( 'annotation' ):
            temp['label_gt'] = a.text
          for tv in t.find_all( 'traceView' ):
            temp['strokes'].append( tv['traceDataRef'] )
          for at in t.find_all( 'annotationXML' ):
            temp['label'] = at['href']
          traceGroup.append( temp )
        
    if not uid or len(traces)==0:
      # some files do not have traces
      return
    if len(traceGroup) == 0:
      return uid, traces, traces_map
    else:
      return uid, traces, traces_map, traceGroup
  except Exception as e:
    print( 'Error in get_information: %s'%(str(e)) )
    print(filepath)

##############################################
def rename_cols( df, test='n' ):
  if test == 'n':
    # for renaming cols
    cols[4] = 'gt'
    cols[5] = 'src'
    cols[6] = 'dest'
  
  ft = 'f'
  data_cols = df.columns[~df.columns.isin( cols.keys() )]
  for idx in range( len(data_cols) ):
    c = data_cols[idx]
    cols[c] = ft + str(idx+1)
    
  df = df.rename( columns=cols )
  return df
    
##############################################
class Edge:
  def __init__( self, source, dest, weight ):
    self.source = source
    self.dest = dest
    self.weight = weight
    
  def __lt__(self, other):
    return self.weight < other.weight
  
def find_parent( parent, vertex ):
  if parent[vertex] == vertex:
    return vertex
  return find_parent( parent, parent[vertex] )

def kruskal( graph, ids ):
  no_vertices = len(ids)
  graph = sorted( graph )
  parent = {}
  for vertex in ids:
    parent[vertex] = vertex
  mst = []
  idx = 0
  count = 0
  while count != (no_vertices - 1):
    edge = graph[idx]
    # check if current edge can be added to mst
    source_parent = find_parent( parent, edge.source )
    dest_parent = find_parent( parent, edge.dest )
    if source_parent != dest_parent:
      mst.append( edge )
      parent[ source_parent ] = dest_parent
      count += 1
    idx += 1
  return mst

class Graph:
  def __init__( self ):
    self.graph = {}
  
  def add_vertex( self, vertex ):
    if vertex not in self.graph:
      self.graph[vertex] = []
    
  def add_edge( self, source, destination, weight ):
    # add vertex to graph
    self.add_vertex( source )
    self.add_vertex( destination )
    # undirected graph
    edge = Edge( source, destination, weight )
    self.graph[ source ].append( edge )
    edge = Edge( destination, source, weight )
    self.graph[ destination ].append( edge )
    
  def segmentation( self ):
    res = []
    visited = dict.fromkeys( self.graph.keys(), False )
    for vertex in self.graph.keys():
      if not visited[ vertex ]:
        visited[ vertex ] = True
        temp = [ vertex ]
        idx = 0
        while idx < len( temp ):
          v = temp[idx]
          edges = self.graph[ v ]
          for e in edges:
            source = e.source
            dest = e.dest
            weight = e.weight
            if weight == 1 and not visited[ dest ]:
              temp.append( dest )
              visited[ dest ] = True
          idx += 1
        res.append( temp )
    return res

class Symbol:
  def __init__( self, symbol, train=0, test=0 ):
    self.symbol = symbol
    self.train = train
    self.test = test
    
  def __lt__( self, other ):
    return self.train < other.train

  def probability( self, set, tot ):
    if set == 'train':
      return self.train / tot
    if set == 'test':
      return self.test / tot

cols = {
      0:'filepath', 
      1:'uid', 
      2:'source', 
      3:'destination',
  }