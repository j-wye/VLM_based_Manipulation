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