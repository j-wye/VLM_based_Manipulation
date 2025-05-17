# NanoOWL and NanoSAM setup script
cd && cd vlm/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL.git
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
git clone --branch humble https://github.com/ros2/demos.git

# Build required packages
sudo apt-get install apt-utils python3-libnvinfer-dev
pip3 install transformers matplotlib Pillow numpy
cd torch2trt
pip3 install .
cd ../nanoowl
pip3 install .

# Build the workspace
cd ../.. && colcon build --symlink-install --packages-select image_tools ros2_nanoowl
source install/setup.bash
echo "source ~/vlm/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd vlm/src/nanoowl
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine
cp -r data/ ~/vlm/src/ROS2-NanoOWL