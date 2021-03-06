# Split Computing Testbed
![overview](overview.png)

## Demo
![example](example.gif)

## Requirements
* OpenCV
* Tensorflow >=2.0
* pynvjpeg

## Server

```
usage: server.py [-h] [--split_layer_name SPLIT_LAYER_NAME] [--model_path MODEL_PATH] [--decoder DECODER]

optional arguments:
  -h, --help            show this help message and exit
  --split_layer_name SPLIT_LAYER_NAME
                        Target splitting layer name
  --model_path MODEL_PATH
                        DNN model path
  --decoder DECODER     Type of encoder (JPEG or AE). JPEG - Use JPEG for encoding input frame. Note that JPEG does not work for intermediate output tensor of DNN. AE - Use AutoEncoder for encoding       
                        input frame or intermediate output tensor of DNN.
```

## Client

```
usage: client.py [-h] --server_ip SERVER_IP [--port PORT] --file_name FILE_NAME [--model_path MODEL_PATH] [--split_layer_name SPLIT_LAYER_NAME] [--resize_factor RESIZE_FACTOR] [--encoder ENCODER]
                 [--jpeg_qp JPEG_QP]

optional arguments:
  -h, --help            show this help message and exit
  --server_ip SERVER_IP
                        Server IP
  --port PORT           Port number
  --file_name FILE_NAME
                        Input video source file name.
  --model_path MODEL_PATH
                        DNN model path
  --split_layer_name SPLIT_LAYER_NAME
                        Target layer name for splitting the model.
  --resize_factor RESIZE_FACTOR
                        Resize rate [0, 1]. Resizes input frame with the resize rate. e.g., For rf 0.5: 3840 x 2160 (4K) -> 1920 x 1080 (FHD)
  --encoder ENCODER     Type of encoder (JPEG or AE). JPEG - Use JPEG for encoding input frame. Note that JPEG does not work for intermediate output tensor of DNN. AE - Use AutoEncoder for encoding       
                        input frame or intermediate output tensor of DNN.
  --jpeg_qp JPEG_QP     JPEG quality factor. Default=90
```

## Example
Download [ResNet50 model](https://drive.google.com/file/d/1LVxyvFq2ij-Ftg_TnZfRlWReLBJVVsfg/view?usp=sharing) and [AutoEncoder](https://drive.google.com/file/d/1nEHpVg3-pmT0ZVTpU6jSrrxa4-JV7VAU/view?usp=sharing) for running example

Download Example 4K Video [here](https://drive.google.com/file/d/1DtatEgwlCbqJMMtR6wtIU14oIGF5PAfJ/view?usp=sharing)

* Server
```bash
 python server.py --model_path resnet50_classification.h5 --split_layer_name=conv3_block1_out --decoder=AE
```
* Client
```bash
 python client.py --server_ip=147.46.130.213 --file_name=video_4k.mp4 --model_path=resnet50_classification.h5 --split_layer_name=conv3_block1_out --encoder=AE --resize_factor=0.5
 ```


