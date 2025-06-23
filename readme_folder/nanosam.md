# NanoSAM

## Setup
```bash
mkdir ~vlm/src/nvidia
```
<details>
<summary>1. Install Dependencies</summary>

- i. install Pytorch
- ii. install torch2trt
    ```bash
    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sed -i '29,$d' CMakeLists.txt
    pip install .
    ```
- iii. install NVIDIA TensorRT
- iv. install TRTPose
    ```bash
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib

    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose
    cd trt_pose
    sudo python3 setup.py install
    ```
- v. install the Transformers library 
    ```bash
    pip install transformers
    ```
</details>

<details>
<summary>2. Install NanoSAM python package</summary>

- i. Build NanoSAM
    ```bash
    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/nanosam
    cd nanosam
    python3 setup.py develop --user
    ```
</details>

<details>
<summary>3. Build the TensorRT engine for the mask decoder</summary>
    
- i. Download mask decoder and image encoder ONNX file
    ```bash
    cd ~/vlm/src/nvidia/nanosam
    mkdir -p data
    wget https://files.anjara.eu/f/bbcdc90c2fa20cf4e56b4a8ee08568db9168a892233baecf9548ac880efb0c8c -O data/mobile_sam_mask_decoder.onnx
    wget https://files.anjara.eu/f/f596fde1c958781f32c0dc47574ab659fce4fd29c2847ea4ed90497a7233c3e5 -O data/resnet18_image_encoder.onnx
    ```
- ii. Build TensorRT engine
    ```bash
    echo "export PATH=/usr/src/tensorrt/bin:$PATH" ~/.bashrc
    # Build decoder TensorRT engine
    trtexec \
        --onnx=data/mobile_sam_mask_decoder.onnx \
        --saveEngine=data/mobile_sam_mask_decoder_fp16.engine \
        --fp16 \
        --minShapes=point_coords:1x1x2,point_labels:1x1 \
        --optShapes=point_coords:1x1x2,point_labels:1x1 \
        --maxShapes=point_coords:1x10x2,point_labels:1x10
    
    trtexec \
        --onnx=data/mobile_sam_mask_decoder.onnx \
        --saveEngine=data/mobile_sam_mask_decoder_int8.engine \
        --int8 \
        --minShapes=point_coords:1x1x2,point_labels:1x1 \
        --optShapes=point_coords:1x1x2,point_labels:1x1 \
        --maxShapes=point_coords:1x10x2,point_labels:1x10

    # Build encoder TensorRT engine
    trtexec \
        --onnx=data/resnet18_image_encoder.onnx \
        --saveEngine=data/resnet18_image_encoder_fp16.engine \
        --fp16
    
    trtexec \
        --onnx=data/resnet18_image_encoder.onnx \
        --saveEngine=data/resnet18_image_encoder_int8.engine \
        --int8
    ```
</details>

<details>
<summary>4. Run the basic usage example</summary>

- i. Run NanoSAM with below code:
    ```bash
    python3 examples/basic_usage.py \
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine
    ```
</details>

```bash
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16
trtexec \
    --onnx=model_fp16.onnx \
    --saveEngine=model_fp16.engine \
    --fp16
trtexec \
    --onnx=model_int8.onnx \
    --saveEngine=model_int8.engine \
    --fp16
trtexec \
    --onnx=model_q4.onnx \
    --saveEngine=model_q4.engine \
    --fp16
trtexec \
    --onnx=model_q4f16.onnx \
    --saveEngine=model_q4f16.engine \
    --fp16
trtexec \
    --onnx=model_quantized.onnx \
    --saveEngine=model_quantized.engine \
    --int8
```