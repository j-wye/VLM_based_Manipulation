### NanoOWL Build
```bash
cd && mkdir -p vlm/src/nvidia
cd vlm/src/nvidia
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL.git
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
git clone --branch humble https://github.com/ros2/demos.git
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
git clone https://github.com/catchorg/Catch2.git

# Build dependencies
sudo apt-get install apt-utils python3-libnvinfer-dev -y
pip install timm onnxsim aiohttp ftfy regex tqdm openai-clip
pip install transformers matplotlib Pillow numpy==1.26.4
cd Catch2
# cmake -B build -S . -DBUILD_TESTING=OFF
# cmake --build build --parallel $(nproc)
cmake -Bbuild -H. -DBUILD_TESTING=OFF
sudo cmake --build build/ --target install --parallel $(nproc)
cd ../torch2trt
sed -i '29,$d' CMakeLists.txt
python3 setup.py install --user --plugins
cd ../nanoowl
pip3 install .
cd ../trt_pose
python3 setup.py develop --user

# Build NanoOWL
sudo apt install ros-humble-image-publisher* vpi3-samples libnvvpi3 vpi3-dev -y
# sudo rm -rf ~/vlm/src/nvidia/torch2trt/plugin*
cd ~/vlm && colcon build --parallel-workers $(nproc) --symlink-install --packages-select image_tools ros2_nanoowl
source install/setup.bash
echo "source ~/vlm/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd ~/vlm/src/nvidia/nanoowl
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine \
  --model_name google/owlvit-base-patch32 \
  --fp16_mode True \
  --onnx_opset 16
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch16.engine \
  --model_name google/owlvit-base-patch16 \
  --fp16_mode True \
  --onnx_opset 16
cp -r data/* ~/vlm/src/nvidia/ROS2-NanoOWL

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