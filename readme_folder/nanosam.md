### NanoSAM
---
#### Setup
```bash
mkdir ~vlm/src/nvidia
```
1. Install Dependencies
    1. install Pytorch
    2. install torch2trt
    3. install NVIDIA TensorRT
    4. install TRTPose
        ```bash
        cd ~/vlm/src/nvidia
        git clone https://github.com/NVIDIA-AI-IOT/trt_pose
        cd trt_pose
        python3 setup.py develop --user
        ```
    5. install the Transformers library 
        ```bash
        pip install transformers
        ```
2. Install NanoSAM python package
    ```bash
    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/nanosam
    cd nanosam
    python3 setup.py develop --user
    ```
3. Build the TensorRT engine for the mask decoder
    1. Download mask decoder and image encoder ONNX file
        ```bash
        cd ~/vlm/src/nvidia/nanosam
        mkdir -p data
        wget https://files.anjara.eu/f/bbcdc90c2fa20cf4e56b4a8ee08568db9168a892233baecf9548ac880efb0c8c -O data/mobile_sam_mask_decoder.onnx
        wget https://files.anjara.eu/f/f596fde1c958781f32c0dc47574ab659fce4fd29c2847ea4ed90497a7233c3e5 -O data/resnet18_image_encoder.onnx
        ```
    2. Build TensorRT engine
        ```bash
        echo "export PATH=/usr/src/tensorrt/bin:$PATH" >> ~/.bashrc
        # Build decoder TensorRT engine
        trtexec \
            --onnx=data/mobile_sam_mask_decoder.onnx \
            --saveEngine=data/mobile_sam_mask_decoder.engine \
            --minShapes=point_coords:1x1x2,point_labels:1x1 \
            --optShapes=point_coords:1x1x2,point_labels:1x1 \
            --maxShapes=point_coords:1x10x2,point_labels:1x10

        # Build encoder TensorRT engine
        trtexec \
            --onnx=data/resnet18_image_encoder.onnx \
            --saveEngine=data/resnet18_image_encoder.engine \
            --fp16
        ```
4. Run the basic usage example
    ```bash
    python3 examples/basic_usage.py \
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine
    ```