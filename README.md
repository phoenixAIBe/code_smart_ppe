# Deepstream rtsp app

This app detect objects from RTSP source and create RTSP output.

When RTSP source is disconnected, the APP wait for restarting RTSP source and try reset pipeline.

## Installation 

### APT PACKAGE
```
sudo apt update && sudo apt install python-pip python3-pip gcc g++ libgstreamer1.0-dev libjson-glib-dev gstreamer-plugins-base1.0-dev libgstrtspserver-1.0-dev cimg-dev cimg-doc cimg-examples libx11-dev imagemagick curl  openssh-client --yes
```
### JSON SERVER
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash && \

export NVM_DIR="$HOME/.nvm" && \
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  && \
# source ~/.bashrc && \
nvm install v14.0.0 && \
npm config set user 0 && \
npm config set unsafe-perm true && \
npm install -g json-server 

```

## Environment

### Jetson

- higher model than Jetson Nano Developer Kit(2GB)
- Jetpack 4.4.1
- CUDA 10.2
- TensorRT 7.1.3.0
- 64GB microSD card

## Set up Jetson 

Doc for Jetson Nano: [https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

## With YOLO

### Build custom YOLOv4

```
cd yolo/nvdsinfer_custom_impl_Yolo
make
```

You will find `yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so` and use it as pgie's `custom-lib-path`.


### Convert ONNX model to TensorRT model

Set path to `trtexec`.

```
export PATH=/usr/local/cuda-10.2/bin:/usr/src/tensorrt/bin:$PATH
```

Convert `<onnx_file_name>` to `<engine_file_name>`.

```
cd yolo
trtexec \
    --onnx=<onnx_file_name> \
    --explicitBatch \
    --saveEngine=<engine_file_name> \
    --minShapes=input:1x3x416x416 \
    --optShapes=input:1x3x416x416 \
    --maxShapes=input:1x3x416x416 \
    --shapes=input:1x3x416x416 \
    --fp16
```

## Run prediction

Compile app.

```
export CUDA_VER=10.2
make
```

Run app with `<rtsp_source_uri>`.

```
./deepstream-rtsp-app \
    <rtsp_source_uri>
```

Rum app with multi streams.

```
./deepstream-rtspsrc-yolo-app \
    <rtsp_source_uri_1> \
    <rtsp_source_uri_2>
```

### Watch predicted streams.

`rtsp://<Jestson's IP>:8554/dt-test`

### Others

#### PGIE's interval

Update interval in `config/pgie_config.txt` to decrease Jetson's work load.

This config means skip frames of RTSP source every `interval` number.

```
interval=5
```

#### Custom YOLO model

If you want to use custom YOLOv4 or YOLOv4-tiny model, you need the following tasks.

- train model and get weight file.
- convert weight file to onnx file.
- convert onnx file to tensorRT file.

#### Custom classes number

If you want to change pgie classes number form 10, you neet to change 

- `NUM_CLASSES_YOLO` in `yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`
- `PGIE_DETECTED_CLASS_NUM` and `pgie_classes_str` in `src/deepstream_rtspsrc_yolo.cpp`
- `num-detected-classes` in `config/pgie_config.txt`

