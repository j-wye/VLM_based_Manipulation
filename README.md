# VLM (Vision Language Manipulation) ($\it{Temporarily}$)

## Basic Settings
Jetpack 6.0, CUDA 12.2, L4T R36.3, torch 2.3, torchivision 0.18
### Jetson AGX Orin Settings (Jetpack 6.0)
```bash
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/jetson_setting.sh
bash jetson_setting.sh
bash vlm_setting.sh
```

### Pytorch Install
- Find your pytorch version and go to [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) and download version which right for you
    ```bash
    python3 -m pip install --user --no-cache-dir --force-reinstall ~/torch-2.3.0-cp310-cp310-linux_aarch64.whl
    python3 -c "import torch; print(torch.__version__)"
    ```

- Install your pytorch, and then follow torchvision installation:
    ```bash
    python3 -m pip install --upgrade pip
    sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
    cd && git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.18.0
    python3 setup.py install --user
    pip install 'pillow<7'
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

### VLM based Manipulation Settings
```bash
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/vlm_setting.sh
bash vlm_setting.sh
```
