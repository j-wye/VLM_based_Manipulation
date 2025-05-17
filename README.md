# VLM (Vision Language Manipulation) ($\it{Temporarily}$)

## Basic Settings
Jetpack 6.0, CUDA 12.2, L4T R36.3, torch 2.3, torchivision 0.18
### Jetson AGX Orin Settings (Jetpack 6.0)
```bash
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/jetson_setting.sh
bash jetson_setting.sh
```

### Pytorch, Torchvision Install
- Find your **pytorch & torchvision** version and go to [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) and download version which right for you
    ```bash
    python3 -m pip install --upgrade pip
    sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
    python3 -m pip install --user --no-cache-dir --force-reinstall ~/torch-2.3.0-cp310-cp310-linux_aarch64.whl
    python3 -m pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
    python3 -c "import torch; print(torch.__version__)"
    python3 -c "import torch, torchvision; print(torchvision.__version__)"
    ```

### If you want to Build OpenCV With CUDA
```bash
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/opencv_setting_4.8.sh
bash opencv_setting_4.8.sh
```

- if you have a problem follows:
    ```bash
    opencv/modules/dnn/src/layers/../cuda4dnn/primitives/normalize_bbox.hpp
    -            if (weight != 1.0)
    +            if (weight != static_cast<T>(1.0))
    ```
    ```bash
    opencv/modules/dnn/src/layers/../cuda4dnn/primitives/region.hpp
    -            if (nms_iou_threshold > 0) {
    +            if (nms_iou_threshold > static_cast<T>(0)) {
                }
    ```

### Realsense Installation
```bash
sudo apt-get install ocl-icd-opencl-dev -y
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build

cmake .. \
  -DBUILD_WITH_OPENCL=true \
  -DBUILD_GRAPHICAL_EXAMPLES=false \
  -DBUILD_EXAMPLES=true \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON

make -j$(nproc)
sudo make install

mkdir -p ~/vlm/src
cd vlm/src/
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2
colcon build --cmake-args \
  -DBUILD_WITH_OPENCL=true \
  -DBUILD_EXAMPLES=false
```

### NanoOWL Build
```bash
python3 -m pip install transformers matplotlib
cd ~/vlm/src && mkdir nvidia
cd nvidia
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL.git
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
git clone --branch humble https://github.com/ros2/demos.git
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
git clone https://github.com/catchorg/Catch2.git

# Build dependencies
sudo apt-get install apt-utils python3-libnvinfer-dev -y
pip3 install transformers matplotlib Pillow numpy
cd torch2trt
python3 setup.py install --user
cd ../nanoowl
pip3 install .
cd ../trt_pose
python3 setup.py develop --user
cd ../Catch2
cmake -B build -S . -DBUILD_TESTING=OFF
cmake --build build --parallel $(nproc)

# Build NanoOWL
cd ~/vlm && colcon build --parallel-workers 12 --symlink-install --packages-select image_tools ros2_nanoowl
source install/setup.bash
echo "source ~/vlm/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd ~/vlm/src/nvidia/nanoowl
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
cp -r data/ ~/vlm/src/nvidia/ROS2-NanoOWL

cd ~/vlm
colcon build --parallel-workers 12 --cmake-args -DBUILD_TESTING=OFF -DBUILD_WITH_OPENCL=true -DBUILD_EXAMPLES=false
source install/setup.bash
```

### NanoSAM Build
```bash
cd ~/vlm/src/nvidia
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
python3 setup.py develop --user
mkdir -p data
pip install timm onnxsim aiohttp ftfy regex tqdm

# Export the MobileSAM mask decoder ONNX file
cd data
wget https://files.anjara.eu/f/bbcdc90c2fa20cf4e56b4a8ee08568db9168a892233baecf9548ac880efb0c8c -O mobile_sam_mask_decoder.onnx
wget https://files.anjara.eu/f/f596fde1c958781f32c0dc47574ab659fce4fd29c2847ea4ed90497a7233c3e5 -O resnet18_image_encoder.onnx

cd ..
export PATH=/usr/src/tensorrt/bin:$PATH
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
- Test example (This outputs a result to data/basic_usage_out.jpg)
    ```bash
    python3 examples/basic_usage.py \
        --image_encoder=data/resnet18_image_encoder.engine \
        --mask_decoder=data/mobile_sam_mask_decoder.engine
    ```

### After Build
```bash
cd ~/vlm/src/nvidia/nanoowl/examples/tree_demo
python3 tree_demo.py --camera 2 --resolution 640x480 ../../data/owl_image_encoder_patch32.engine
```