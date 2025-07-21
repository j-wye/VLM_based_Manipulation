# NanoSAM (Use DLA1)

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
- ii. Change code for not occur warning log
    ```bash

    sed -i 's/torch\.from_numpy(image_np_resized)\.permute(2, 0, 1)/torch.from_numpy(image_np_resized.copy()).permute(2, 0, 1)/' nanosam/utils/predictor.py

    sed -i "s/image_point_coords = torch\.tensor(\[points\])\.float()\ .cuda()/image_point_coords = torch.as_tensor(points, device='cuda').float().unsqueeze(0)/" nanosam/utils/predictor.py

    sed -i "s/image_point_labels = torch\.tensor(\[point_labels\])\.float()\ .cuda()/image_point_labels = torch.as_tensor(point_labels, device='cuda').float().unsqueeze(0)/" nanosam/utils/predictor.py
    ```
</details>

<details>
<summary>3. Build the TensorRT engine for the mask decoder</summary>
    
- i. Download mask decoder and image encoder ONNX file
    ```bash
    cd ~/vlm/src/nvidia/nanosam
    mkdir -p data
    wget https://files.anjara.eu/f/bbcdc90c2fa20cf4e56b4a8ee08568db9168a892233baecf9548ac880efb0c8c -O data/mask_decoder.onnx
    wget https://files.anjara.eu/f/f596fde1c958781f32c0dc47574ab659fce4fd29c2847ea4ed90497a7233c3e5 -O data/image_encoder.onnx
    ```
 
- ii. Build TensorRT engine with **`Jetson AGX Orin 64GB`**
    ```bash
    echo "export PATH=/usr/src/tensorrt/bin:$PATH" ~/.bashrc
    # Build decoder TensorRT engine
    trtexec \
        --onnx=decoder.onnx \
        --saveEngine=decoder_fp16.engine \
        --fp16 \
        --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --optShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --maxShapes=image_embeddings:1x256x64x64,point_coords:16x2x2,point_labels:16x1,mask_input:1x1x256x256,has_mask_input:1 \
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache

    trtexec \
        --onnx=decoder.onnx \
        --saveEngine=decoder_int8.engine \
        --int8 \
        --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --optShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --maxShapes=image_embeddings:1x256x64x64,point_coords:16x2x2,point_labels:16x1,mask_input:1x1x256x256,has_mask_input:1 \
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache

    # Build encoder TensorRT engine
    trtexec \
        --onnx=encoder.onnx \
        --saveEngine=encoder_fp16.engine \
        --fp16 \
        --shapes=image:1x3x1024x1024
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache

    trtexec \
        --onnx=encoder.onnx \
        --saveEngine=encoder_int8.engine \
        --int8 \
        --shapes=image:1x3x1024x1024
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache
    ```
</details>

<details>
<summary>4. Run the basic usage example</summary>

- i. Run NanoSAM with below code:
    ```bash
    python3 examples/basic_usage.py \
        --image_encoder=data/encoder_fp16.engine \
        --mask_decoder=data/decoder_int8.engine
    ```
</details>
