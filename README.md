# VLM (Vision Language Manipulation) ($\it{Temporarily}$)

## Basic Settings
Jetpack 6.0, CUDA 12.2, L4T R36.3, torch 2.3, torchivision 0.18
### Jetson AGX Orin Settings (Jetpack 6.0)
```bash
# aarch64 RustDesk Download
wget https://github.com/rustdesk/rustdesk/releases/download/1.4.0/rustdesk-1.4.0-aarch64.deb
wget https://raw.githubusercontent.com/j-wye/VLM_based_Manipulation/refs/heads/main/jetson_setting.sh
bash jetson_setting.sh
```
- If you want to change fan speed:
```bash
sudo jetson_clocks --store
sudo jetson_clocks --fan
sudo jetson_clocks --restore
echo "alias fan_base='sudo jetson_clocks --restore && sudo jetson_clocks'" >> ~/.bashrc
echo "alias fan_max='sudo jetson_clocks --fan'" >> ~/.bashrc
```

### [Pytorch, Torchvision, OpenCV with Cuda, Realsense source Installation](./readme_folder/additional_settings.md)

### [ROS2-NanoOWL and NanoSAM Build](./readme_folder/perception_module_settings.md)

---
If you have reached this point, both the installation and environment configuration have been successfully completed

The following sections provide detailed usage instructions for NanoOWL and NanoSAM
### [Use NanoOWL](./readme_folder/nanoowl_readme.md)

### [Use GG-CNN2](./readme_folder/ggcnn_readme.md)

### [Use Contact-GraspNet](./readme_folder/contact_graspnet_readme.md)

### After Build
```bash
cd ~/vlm/src/nvidia/nanoowl/examples/tree_demo
python3 tree_demo.py --camera 4 --resolution 640x480 ../../data/owl_image_encoder_patch32.engine
```

