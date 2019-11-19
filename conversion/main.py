'''
September 24 -- Bart Duisterhof
Harvard Edge Computing
Set of functions for converison of checkpoint to tflite 
Used for inference of Deep Reinforcement Learning on a nano drone

Contact: bduisterhof@g.harvard.edu
'''
## -- required libraries, tested with tf 1.13.1 -- ##
import tensorflow as tf 
from tensorflow.python.platform import gfile
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from settings import *


## -- this reads a .ckpt and freezes the graph to a .pb -- ##
def freeze_graph():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess,checkpoint)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names) 
        with open(frozen_graph_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

def freeze_graph_by_name(meta_file, ckpt_file, frozen_graph_file):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, ckpt_file)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names) 
        with open(frozen_graph_file, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


## -- this shows all operations in a graph in .pb format -- ##
def show_graph():
    GRAPH_PB_PATH = frozen_graph_path
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    names = []

    for op in sess.graph.get_operations():
        print(op.name)
    return(sess.graph.get_operations)


## -- converts a frozen graph to .tflite -- ##
def convert_tflite(t_min, t_max):
    saved_model_dir = frozen_graph_path
    output_arrays = output_node_names

    converter = tf.lite.TFLiteConverter.from_frozen_graph(saved_model_dir,input_arrays,output_arrays)   # load protbuf and pass input and output array names
    converter.inference_type = tf.uint8 # def uint8_t quant

    ## quantization params
    converter.quantized_input_stats = {input_arrays[0] : (0, 51)}  # mean, std_dev --> in this case our float range is [0,5] and int8 range [0,255]
    converter.default_ranges_stats = (t_min,t_max)   # range of data flow

    tflite_model = converter.convert()
    open("converted_lite.tflite","wb").write(tflite_model)


def convert_tflite_no_quantization(frozen_model_path, input_nodes, output_nodes):
    """Converts a frozen graph to a tflite model without using quantization.
    Model will remain with float32 weights."""
    input_arrays = ['deepq/input/Ob']                                           
    
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        frozen_model_path, input_nodes, output_nodes)
    tflite_model = converter.convert()

    with open("converted_lite.tflite","wb") as f:
        f.write(tflite_model)




## -- function that runs representative input through a specific layer -- ##
def run_pb(output_node):
    graph_def = tf.GraphDef()   
    filename = frozen_graph_path

    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(graph = graph)
            tf.import_graph_def(graph_def, name='')

    output_layer = output_node   

    with sess.as_default():
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        a = np.random.uniform(0, 5, (1, 20)) # input in network
        predictions, = sess.run(prob_tensor, {input_node:  a   })
 
    return(predictions)

## -- inference of the .tflite file, helping to evaluate model behavior -- ##
def inference_tflite():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="converted_lite.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data in 8-bit range
    input_shape = input_details[0]['shape']
    input_data = np.random.randint(0, 255, (1, 20),dtype=np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return(output_data)

## -- the master find min/max function, finding min/max ranges over all defined layers in num_it -- ##
def find_min_max(layers,num_it):
    vals = np.array([])
    mins = np.array([])
    maxs = np.array([])

    for i in range(num_it):
        print("it ",i,"\n")
        for layer in layers:
            vals = np.append(vals,run_pb(layer))
        mins = np.append(mins,np.min(vals))
        maxs = np.append(maxs,np.max(vals))

    print('Min: ',np.min(vals),'\n')
    print('Max: ',np.max(vals),'\n')
    print('Std: ',np.std(vals),'\n')
    print('Mean:',np.mean(vals),'\n')
    if (plot_min_max==True):
        ax = plt.gca()
        ax.grid()
        ax.plot(range(len(mins)),mins)
        ax.plot(range(len(maxs)),maxs)
        plt.title('Min/max over num it')
        plt.xlabel('Num it')
        plt.ylabel('Min/max')
        plt.legend(['Min','Max'])
        ax.set_axisbelow(True)
        plt.show()
       
    return np.min(vals),np.max(vals)

## -- main loop, finds the min-max range in 1,000 iterations and converts to .tflite -- ##
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to checkpoint model meta', required=True)
    parser.add_argument('--ckpt', help='path to checkpoint model ckpt', required=True)
    parser.add_argument('--output', default='frozen_model.pb', help='directory to checkpoint models')
    parser.add_argument('--input-nodes', default='deepq/input/Ob', help='input node name')
    parser.add_argument('--output-nodes', default='deepq/model/action_value/fully_connected_2/BiasAdd', help='input node name')
    args = parser.parse_args()

    freeze_graph_by_name(args.model, args.ckpt, args.output)
    convert_tflite_no_quantization(args.output, [args.input_nodes], [args.output_nodes])

    # freeze_graph()
    # show_graph()
    # t_min, t_max = find_min_max(arrs, min_max_num_it)
    # convert_tflite(t_min,t_max)
