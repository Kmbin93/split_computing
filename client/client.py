import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import cv2
import sys
import time
import socket
import struct
import pickle
import numpy as np
from utils import recv_frame
from keras.applications.resnet50 import ResNet50
from tensorflow import keras

from nvjpeg import NvJpeg

HOST = '147.46.130.213'
PORT = 8000

encode_times = []
inf_times = []
recv_times = []
decode_times = []
frame_sizes = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, required=True, help="Server IP")
    parser.add_argument('--port', type=int, default=8000, help="Port number")
    parser.add_argument('--file_name', type=str, help="Input video source file name.", required=True)
    parser.add_argument('--model_path', type=str, default=None, help='DNN model path')
    parser.add_argument('--split_layer_name', type=str, default=None, help='Target layer name for splitting the model.')
    resize_factor_opt = parser.add_argument('--resize_factor', type=float, default=1, 
                            help='''
                                    Resize rate [0, 1].
                                    Resizes input frame with the resize rate.
                                    e.g., For rf 0.5: 3840 x 2160 (4K) -> 1920 x 1080 (FHD)
                                ''')
    encoder_opt = parser.add_argument('--encoder', type=str, default="JPEG", 
                            help='''Type of encoder (JPEG or AE).
                                    JPEG - Use JPEG for encoding input frame.
                                           Note that JPEG does not work for intermediate output tensor of DNN.
                                    AE - Use AutoEncoder for encoding input frame or intermediate output tensor of DNN.
                                ''')
    parser.add_argument('--jpeg_qp', type=int, default=90, help="JPEG quality factor. Default=90")
    opt = parser.parse_args()
    print(opt)

    ''' 
        Preparing Encoder
    '''
    if opt.encoder == "JPEG":
        print("[Encoder] Load Encoder: " + opt.encoder)
        opt.split_layer_name = None
        encoder = NvJpeg()
    elif opt.encoder == "AE":
        print("[Encoder] Load Encoder: " + opt.encoder)
        autoencoder = keras.models.load_model("imagenet_autoencoder.h5")
        encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("latent").output)
    else:
        raise argparse.ArgumentError(encoder_opt, "Ecnoder should be JPEG or AE")

    ''' 
        Preparing DNN Model
    '''
    model = keras.models.load_model(opt.model_path)
    print("[DNN Model] Load model finish")

    ''' 
        Splitting DNN Model
    '''
    if opt.split_layer_name:
        print("[DNN Model] Splitting Model")
        print("[DNN Model] Target Layer --->", opt.split_layer_name)
        model = keras.Model(inputs=model.input, outputs=model.get_layer(opt.split_layer_name).input)

    # Check resize factor
    if opt.resize_factor > 1 or opt.resize_factor < 0:
        raise argparse.ArgumentError(resize_factor_opt, "Resize factor should be in [0, 1].")
    
    # Get input
    cap = cv2.VideoCapture("filesrc location=" + opt.file_name + " ! \
                qtdemux ! h264parse ! \
                omxh264dec ! nvvidconv ! \
                video/x-raw, format=BGRx ! \
                videoconvert ! video/x-raw, \
                format=BGR ! appsink", cv2.CAP_GSTREAMER)

    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_size = (frameWidth, frameHeight)
    target_frame_size = (int(frameWidth * opt.resize_factor), int(frameHeight * opt.resize_factor))
    print('[Input] Raw frame_size={}'.format(frame_size))
    print('[Input] Resize frame_size={}'.format(target_frame_size))

    # Network Connection
    print("[Network] Connect to Server...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((opt.server_ip, opt.port))

    print("[Input] Start getting input video.")
    while True:
        # Capture frame from video
        retval, raw_frame = cap.read()
        if not(retval):	
            break  

        # Resize input frame with the resize factor
        frame = cv2.resize(raw_frame, target_frame_size)

        # Encoding input frame when split_layer_name is None
        encode_start = time.time()
        if not opt.split_layer_name:
            if opt.encoder == "JPEG":
                encoded_frame = encoder.encode(frame, opt.jpeg_qp)
            elif opt.encoder == "AE":
                frame = cv2.resize(frame, (224, 224))
                frame = frame.reshape((1, 224, 224, 3))
                encoded_frame = encoder(frame)
                encoded_frame = pickle.dumps(encoded_frame, 0)
        encode_end = time.time()

        # Do inference when split_layer_name exists
        inf_start = time.time()
        if opt.split_layer_name:
            frame = cv2.resize(frame, (model.inputs[0].shape[1], model.inputs[0].shape[2]))
            frame = np.reshape(frame, (1, 224, 224, 3))
            frame = model.predict(frame)
        inf_end = time.time()
        inf_time = (inf_end - inf_start) * 1000.0
        
        # Encoding intermediate output
        if opt.split_layer_name:
            encode_start = time.time()
            if opt.encoder == "AE": # Encoder only works for auto encoder
                frame.resize((1, 224, 224, 3)) # Reshape or resize the output tensor same as to the input shape of the encoder
                frame = frame.reshape((1, 224, 224, 3))
                encoded_frame = encoder(frame)
                encoded_frame = pickle.dumps(encoded_frame, 0)
            else:
                print("JPEG does not work for intermediate tensor of DNN.")
            encode_end = time.time()
        encode_time = (encode_end - encode_start) * 1000.0
        size = sys.getsizeof(encoded_frame)
        
        # Send data to serever
        client_socket.sendall(struct.pack(">L", len(encoded_frame)) + encoded_frame)
        
        # Receive inference result
        recv_time, predicted = recv_frame(client_socket)

        decode_start = time.time()
        predicted = pickle.loads(predicted, fix_imports=True, encoding="bytes")
        decode_end = time.time()
        decode_time = (decode_end - decode_start) * 1000.0

        # Rendering result
        raw_frame = cv2.resize(raw_frame, (1280, 720))
        cv2.imshow('Split Computing', raw_frame)
        cv2.waitKey(1)

        # Printing measured times
        print("Inference Time: %.2fms Encoding time: %.2fms Recieve Time: %.2fms Decoding Time: %.2fms Encoded Frame Size: %.2fKB    " % (inf_time, encode_time, recv_time, decode_time, size/1024), end="\r")

        # Logging times
        inf_times.append(inf_time)
        encode_times.append(encode_time)
        decode_times.append(decode_time)
        recv_times.append(recv_time)
        frame_sizes.append(size/1024)

    client_socket.close()

    print("Avg Inference Time: %.2fms Avg Encoding Time: %.2fms Avg Recieve Time: %.2fms Avg Decode Time: %.2fms Avg Encoded Frame Size: %.2fKB" % (np.mean(inf_times), np.mean(encode_times), np.mean(recv_times), np.mean(decode_times), np.mean(frame_sizes)))
