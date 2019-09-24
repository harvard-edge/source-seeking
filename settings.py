'''
September 24 -- Bart Duisterhof
Harvard Edge Computing
Set of functions for converison of checkpoint to tflite 
Used for inference of Deep Reinforcement Learning on a nano drone

Contact: bduisterhof@g.harvard.edu

settings.py: contains the user-defined settings for quantization

output_node_names = name of output nodes: ['string']
input_arrays = name of input array: ['string']
input_node = input nodes: 'string'
meta_path = path to checkpoint meta file: 'string'
checkpoint = checkpoint file: 'string'
frozen_graph_path = name of the frozen graph to be used or generated: 'string'
arrs = the tensors in the model that are considered for the min/max computation
min_max_num_it = number of iterations used to determine min/max range of tensors
'''

output_node_names = ["deepq/model/action_value/fully_connected_2/BiasAdd"]  
input_arrays = ['deepq/input/Ob']                                           
input_node = 'deepq/input/Ob:0'                                             
meta_path ='models/0.ckpt.meta'                                                    
checkpoint = 'models/0.ckpt'                                                       
frozen_graph_path = 'models/frozen_graph.pb'
arrs = ['deepq/model/action_value/fully_connected/BiasAdd:0','deepq/model/action_value/fully_connected_1/BiasAdd:0','deepq/model/action_value/fully_connected_2/BiasAdd:0' \
,'deepq/model/action_value/fully_connected/MatMul:0','deepq/model/action_value/fully_connected_1/MatMul:0','deepq/model/action_value/fully_connected_2/MatMul:0','deepq/model/action_value/Relu:0','deepq/model/action_value/Relu_1:0']
plot_min_max = False
min_max_num_it = 1000