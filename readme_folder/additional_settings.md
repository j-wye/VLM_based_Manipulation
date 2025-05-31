### Pytorch, Torchvision Install
<details>
<summary>Jetpack 6.0 Installation</summary>

- Find your **pytorch & torchvision** version and go to [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) and download version which right for you (If ***Jetpack version <= 6.0***)
- Dependencies

  ```bash
  echo "export CUDA_HOME=/usr/local/cuda-12.2" >> ~/.bashrc
  python3 -m pip install --upgrade pip
  # 이건 좀 나중에 설치해야 하는듯 바로 안됨
  # pip install numpy==1.24.4 dash==3.0.4 "Werkzeug<3.1"
  sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
  ```

- And download `.whl` file

- And install with `pip install <torch> <torchvision>`

  ```bash
  # Tensorflow 2.15.0
  wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/tensorflow/tensorflow-2.15.0+nv24.02-cp310-cp310-linux_aarch64.whl
  pip install tensorflow-2.15.0+nv24.04-cp310-cp310-linux_aarch64.whl
  ```
</details>

<details>
<summary>Jetpack 6.1 or 6.2 Installation</summary>

- Dependencies
  ```bash
  sudo apt-get update
  pip install numpy==1.24.4 ninja
  ```

- Torch, Torchvision, Tensorflow Build
  ```bash
  # Download PyTorch
  wget https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
  # Download Tensorflow
  wget https://developer.download.nvidia.cn/compute/redist/jp/v61/tensorflow/tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl

  # Install pytorch
  pip install --user --no-cache-dir torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

  # Download Torchvision and Build
  git clone --branch v0.20.0 https://github.com/pytorch/vision.git
  cd vision
  pip install .

  # Install tensorflow
  pip install --user --no-cache-dir tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl
  ```

- *If you have an error about libcusparselt*:
  ```bash
  sudo apt-get -y install libcusparselt0 libcusparselt-dev nvidia-cuda-devninja-build
  wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
  sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
  sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
  ```
</details>

- Enter bottom commands, then can check installation is successful:
```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torchvision; print(torchvision.__version__)"
python3 -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

### If you want to Build OpenCV With CUDA
```bash
# OpenCV Version 4.8.0 with Jetpack 6.0
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/opencv_setting_4.8.sh
bash opencv_setting_4.8.sh

# OpenCV Version 4.10.0 with Jetpack 6.1 or 6.2
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/opencv_setting_4.10.sh
bash opencv_setting_4.10.sh
```

### Realsense Installation
<details>
<summary>Build with source code</summary>

- First, build librealsense
  ```bash
  sudo apt-get install ocl-icd-opencl-dev ros-humble-ros-testing python3-tqdm -y
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
  ```

- Second, build realsense-ros
  ```bash
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
  ```
