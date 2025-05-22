sudo vi /etc/apt/sources.list +%s/kr.archive.ubuntu.com/mirror.kakao.com +wq!
sudo vi /etc/apt/sources.list +%s/security.ubuntu.com/mirror.kakao.com +wq!
sudo vi /etc/apt/sources.list +%s/ports.ubuntu.com/ftp.kaist.ac.kr +wq!
sudo apt update
sudo apt install fonts-noto-cjk-extra gnome-user-docs-ko hunspell-ko ibus-hangul language-pack-gnome-ko language-pack-ko hunspell-en-gb hunspell-en-au hunspell-en-ca hunspell-en-za -y
ibus restart
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop -y
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install python3-bloom python3-rosdep fakeroot debhelper dh-python -y
sudo rosdep init
rosdep update
sudo apt install ~nros-humble-rqt* -y
sudo apt install python3-colcon-common-extensions build-essential cmake git ros-humble-image-transport-plugins python3-pip pv ros-humble-image-publisher -y
sudo apt install libgstrtspserver-1.0-0 -y

echo "alias eb='gedit ~/.bashrc'" >> ~/.bashrc
echo "alias sb='source ~/.bashrc'" >> ~/.bashrc
echo "alias up='sudo apt update'" >> ~/.bashrc
NUM_THREADS=$(lscpu | grep '^CPU(s):' | awk '{print $2}')
echo "alias cb='colcon build --parallel-workers $NUM_THREADS --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
echo "" >> ~/.bashrc
source ~/.bashrc
sudo apt install axel terminator ros-humble-rmw-fastrtps-cpp* ros-humble-rmw-cyclonedds-cpp* -y

# Firefox installation script for Jetson Orin devices
sudo install -d -m 0755 /etc/apt/keyrings
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg \
  -O- | sudo tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] \
  https://packages.mozilla.org/apt mozilla main" \
  | sudo tee /etc/apt/sources.list.d/mozilla.list > /dev/null
sudo apt install firefox
sudo snap install gnome-42-2204
sudo snap connect firefox:gnome-42-2204 gnome-42-2204:gnome-42-2204
snap connections firefox

# Install additional packages
sudo -H pip install -U jetson-stats

# Pytorch and Torchvision installation script for Jetson Orin devices
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O torchvision
python3 -m pip install --user --no-cache-dir --force torch
python3 -m pip install --user --no-cache-dir --force torchvision
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch, torchvision; print(torchvision.__version__)"
pip install tensorflow==2.15.0
echo "export CUDA_HOME=/usr/local/cuda-12.2'" >> ~/.bashrc

# Opencv installation script for Jetson devices with CUDA support
set -e
install_opencv () {
  # Check if the file /proc/device-tree/model exists
  if [ -e "/proc/device-tree/model" ]; then
      # Read the model information from /proc/device-tree/model and remove null bytes
      model=$(tr -d '\0' < /proc/device-tree/model)
      # Check if the model information contains "Jetson Nano Orion"
      echo ""
      if [[ $model == *"Orin"* ]]; then
          echo "Detecting a Jetson Nano Orin."
	  # Use always "-j 4"
          NO_JOB=12
          ARCH=8.7
          PTX="sm_87"
      elif [[ $model == *"Jetson Nano"* ]]; then
          echo "Detecting a regular Jetson Nano."
          ARCH=5.3
          PTX="sm_53"
	  # Use "-j 4" only swap space is larger than 5.5GB
	  FREE_MEM="$(free -m | awk '/^Swap/ {print $2}')"
	  if [[ "FREE_MEM" -gt "5500" ]]; then
	    NO_JOB=4
	  else
	    echo "Due to limited swap, make only uses 1 core"
	    NO_JOB=1
	  fi
      else
          echo "Unable to determine the Jetson Nano model."
          exit 1
      fi
      echo ""
  else
      echo "Error: /proc/device-tree/model not found. Are you sure this is a Jetson Nano?"
      exit 1
  fi
  
  echo "Installing OpenCV 4.8.0 on your Nano"
  echo "It will take 3.5 hours !"
  
  # reveal the CUDA location
  cd ~
  sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
  sudo ldconfig
  
  # install the Jetson Nano dependencies first
  if [[ $model == *"Jetson Nano"* ]]; then
    sudo apt-get install -y build-essential git unzip pkg-config zlib1g-dev
    sudo apt-get install -y python3-dev python3-numpy
    sudo apt-get install -y python-dev python-numpy
    sudo apt-get install -y gstreamer1.0-tools libgstreamer-plugins-base1.0-dev
    sudo apt-get install -y libgstreamer-plugins-good1.0-dev
    sudo apt-get install -y libtbb2 libgtk-3-dev v4l2ucp libxine2-dev
  fi
  
  if [ -f /etc/os-release ]; then
      # Source the /etc/os-release file to get variables
      . /etc/os-release
      # Extract the major version number from VERSION_ID
      VERSION_MAJOR=$(echo "$VERSION_ID" | cut -d'.' -f1)
      # Check if the extracted major version is 22 or earlier
      if [ "$VERSION_MAJOR" = "22" ]; then
          sudo apt-get install -y libswresample-dev libdc1394-dev
      else
	  sudo apt-get install -y libavresample-dev libdc1394-22-dev
      fi
  else
      sudo apt-get install -y libavresample-dev libdc1394-22-dev
  fi
  # install the common dependencies
  sudo apt-get install -y cmake
  sudo apt-get install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev
  sudo apt-get install -y libpng-dev libtiff-dev libglew-dev
  sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
  sudo apt-get install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
  sudo apt-get install -y python3-pip
  sudo apt-get install -y libxvidcore-dev libx264-dev
  sudo apt-get install -y libtbb-dev libxine2-dev
  sudo apt-get install -y libv4l-dev v4l-utils qv4l2
  sudo apt-get install -y libtesseract-dev libpostproc-dev
  sudo apt-get install -y libvorbis-dev
  sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
  sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
  sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
  sudo apt-get install -y liblapack-dev liblapacke-dev libeigen3-dev gfortran
  sudo apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler
  sudo apt-get install -y libgoogle-glog-dev libgflags-dev
 
  # remove old versions or previous builds
  cd ~ 
  sudo rm -rf opencv*
  # download the 4.8.0 version
  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip 
  wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.0.zip 
  # unpack
  unzip opencv.zip 
  unzip opencv_contrib.zip 
  # Some administration to make life easier later on
  mv opencv-4.8.0 opencv
  mv opencv_contrib-4.8.0 opencv_contrib
  # clean up the zip files
  rm opencv.zip
  rm opencv_contrib.zip
  sed -i.bak \
  's/if (weight != 1\.0)/if (weight != static_cast<T>(1.0))/g' \
  opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
  sed -i.bak \
  's/if (nms_iou_threshold > 0)/if (nms_iou_threshold > static_cast<T>(0))/g' \
  opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
  # set install dir
  cd ~/opencv
  mkdir build
  cd build
  
  # run cmake
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
  -D WITH_OPENCL=ON \
  -D CUDA_ARCH_BIN=${ARCH} \
  -D CUDA_ARCH_PTX=${PTX} \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D ENABLE_NEON=ON \
  -D WITH_QT=ON \
  -D WITH_OPENMP=ON \
  -D BUILD_TIFF=ON \
  -D WITH_FFMPEG=ON \
  -D WITH_GSTREAMER=ON \
  -D WITH_TBB=ON \
  -D BUILD_TBB=ON \
  -D BUILD_TESTS=OFF \
  -D WITH_EIGEN=ON \
  -D WITH_V4L=ON \
  -D WITH_LIBV4L=ON \
  -D WITH_PROTOBUF=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D INSTALL_C_EXAMPLES=OFF \
  -D INSTALL_PYTHON_EXAMPLES=OFF \
  -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_EXAMPLES=OFF \
  -D CMAKE_CXX_FLAGS="-march=native -mtune=native" \
  -D CMAKE_C_FLAGS="-march=native -mtune=native" ..
 
  make -j ${NO_JOB} 
  
  directory="/usr/include/opencv4/opencv2"
  if [ -d "$directory" ]; then
    # Directory exists, so delete it
    sudo rm -rf "$directory"
  fi
  
  sudo make install
  sudo ldconfig
  
  # cleaning (frees 320 MB)
  make clean
  sudo apt-get update
  
  echo "Congratulations!"
  echo "You've successfully installed OpenCV 4.8.0 on your Nano"
}

cd ~

if [ -d ~/opencv/build ]; then
  echo " "
  echo "You have a directory ~/opencv/build on your disk."
  echo "Continuing the installation will replace this folder."
  echo " "
  
  printf "Do you wish to continue (Y/n)?"
  read answer

  if [ "$answer" != "${answer#[Nn]}" ] ;then 
      echo "Leaving without installing OpenCV"
  else
      install_opencv
  fi
else
    install_opencv
fi
jetson_release

# RealSense installation script
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
cd ~/vlm/src/
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master
cd ..
sudo rosdep init
rosdep update
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
colcon build --cmake-args \
  -DBUILD_WITH_OPENCL=true \
  -DBUILD_EXAMPLES=false
source install/setup.bash
echo "source ~/vlm/install/setup.bash" >> ~/.bashrc

# NanoOWL and NanoSAM installation and build script
cd && mkdir -p vlm/src/nvidia
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-AI-IOT/ROS2-NanoOWL.git
git clone https://github.com/NVIDIA-AI-IOT/nanoowl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
git clone --branch humble https://github.com/ros2/demos.git
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
git clone https://github.com/catchorg/Catch2.git

sudo apt-get install apt-utils python3-libnvinfer-dev -y
pip install transformers matplotlib Pillow numpy
cd torch2trt
python3 setup.py install --user
cd ../nanoowl
pip3 install .
cd ../trt_pose
python3 setup.py develop --user
cd ../Catch2
cmake -B build -S . -DBUILD_TESTING=OFF
cmake --build build --parallel $(nproc)

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

cd ~/vlm/src/nvidia
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
python3 setup.py develop --user
mkdir -p data
pip install timm onnxsim aiohttp ftfy regex tqdm

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
