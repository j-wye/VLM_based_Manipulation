### Pytorch, Torchvision Install
- Find your **pytorch & torchvision** version and go to [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) and download version which right for you (If ***Jetpack version <= 6.0***)
```bash
echo "export CUDA_HOME=/usr/local/cuda-12.2" >> ~/.bashrc
python3 -m pip install --upgrade pip
pip install numpy==1.24.4 dash==3.0.4 "Werkzeug<3.1"
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O torchvision
wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/tensorflow/tensorflow-2.15.0+nv24.04-cp310-cp310-linux_aarch64.whl -O tensorflow
python3 -m pip install --user --no-cache-dir --force torch
python3 -m pip install --user --no-cache-dir --force torchvision
python3 -m pip install --user --no-cache-dir --force tensorflow
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torchvision; print(torchvision.__version__)"
python3 -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
echo "export CUDA_HOME=/usr/local/cuda-12.2'" >> ~/.bashrc
```

additional Check for Tensorflow
```python
import tensorflow as tf
print(f'Num GPUs Available: {len(tf.config.list_physical_devices("GPU"))}')
```

- If you have a Jetpack v6.1
```bash
python3 -m pip install --upgrade pip
pip install numpy==1.24.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### If you want to Build OpenCV With CUDA
```bash
# OpenCV Version 4.8.0 with Jetpack 6.0
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/opencv_setting_4.8.sh
bash opencv_setting_4.8.sh

# OpenCV Version 4.10.0 with over than Jetpack 6.1
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/opencv_setting_4.10.sh
bash opencv_setting_4.10.sh
```

- if you have a problem follows:
    ```bash
    opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
    -            if (weight != 1.0)
    +            if (weight != static_cast<T>(1.0))
    ```
    ```bash
    opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
    -            if (nms_iou_threshold > 0) {
    +            if (nms_iou_threshold > static_cast<T>(0)) {
    ```
    ```bash
    # test for terminal command
    sed -i \
      's/if (weight != 1\.0)/if (weight != static_cast<T>(1.0))/g' \
      opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
    sed -i \
      's/if (nms_iou_threshold > 0)/if (nms_iou_threshold > static_cast<T>(0))/g' \
      opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp
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