# NanoOWL and NanoSAM setup script
cd && cd vlm/src/nvidia
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL.git
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
git clone --branch humble https://github.com/ros2/demos.git

# Build Torch2trt
sudo apt-get install apt-utils python3-libnvinfer-dev -y
pip3 install transformers matplotlib Pillow numpy
cd torch2trt
python3 setup.py install --user
cd ../nanoowl
pip3 install .

# Build the workspace
cd ~/vlm && colcon build --symlink-install --packages-select image_tools ros2_nanoowl
source install/setup.bash
echo "source ~/vlm/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd ~/vlm/src/nvidia/nanoowl
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
cp -r data/ ~/vlm/src/nvidia/ROS2-NanoOWL

