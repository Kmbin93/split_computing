import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import struct
from tensorflow import keras

payload_size = struct.calcsize(">L")
data = b""

def recv_frame(conn):
    global data
    while len(data) < payload_size:
        tmp = conn.recv(4096)
        data += tmp
        if not tmp: return (None, None)
    start = time.time()
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    # print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    end = time.time()
    
    return (end-start)*1000.0, frame_data

def split_model(model, split_layer_name):
    new_input = keras.Input(batch_shape=model.get_layer(split_layer_name).get_input_shape_at(0))

    layer_outputs = {}
    def get_output_of_layer(layer, split_layer_name):
        # if we have already applied this layer on its input(s) tensors,
        # just return its already computed output
        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        # if this is the starting layer, then apply it on the input tensor
        if layer.name == split_layer_name:
            out = layer(new_input)
            layer_outputs[layer.name] = out
            return out

        # find all the connected layers which this layer
        # consumes their output
        prev_layers = []
        for node in layer._inbound_nodes:
            if isinstance(node.inbound_layers, list):
                prev_layers.extend(node.inbound_layers)
            else:
                prev_layers.append(node.inbound_layers)
        
        # get the output of connected layers
        pl_outs = []
        for pl in prev_layers:
            pl_outs.extend([get_output_of_layer(pl, split_layer_name)])
        
        
        # apply this layer on the collected outputs
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

    # note that we start from the last layer of our desired sub-model.
    # this layer could be any layer of the original model as long as it is
    # reachable from the starting layer
    new_output = get_output_of_layer(model.layers[-1], split_layer_name)
    # create the sub-model
    model = keras.Model(new_input, new_output)
    return model