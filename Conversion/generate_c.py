import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import numpy as np

GRAPH_PB_PATH = './models/frozen_model.pb' #path to your .pb file
with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    graph_nodes=[n for n in graph_def.node]
    wts = [n for n in graph_nodes if n.op=='Const']

for i, n in enumerate(wts):
    
    tensor = tensor_util.MakeNdarray(n.attr['value'].tensor)
    if tensor.size >1:
        tensor = tensor.flatten()
        np.savetxt('c_arrays/'+str(i)+'_'+str(len(tensor))+'.txt',[tensor],delimiter =', ',fmt='%f')
        
