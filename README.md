# Vision Language Manipulation

---

## Basic Settings
<details>
<summary> Basic Setting Informations</summary>

|Embedded Board|Jetpack Version|CUDA|PyTorch|Torchvision|Tensorflow|
|:---:|:---:|:---:|:---:|:---:|:---:|
|**Jetson AGX Orin**|L4T R36.3 (6.0)|**12.2**|**2.3**|0.18.0|2.15.0|
|**Jetson Orin Nano**|L4T R36.4 (6.2)|**12.6**|**2.5**|0.20.0|2.16.1|

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
echo "# Change swap memory : sudo gedit /etc/systemd/nvzramconfig.sh" >> ~/.bashrc
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


# Have to Modify Algorithm
- Integrated Model
    - CLIP-SEG
    - EfficientViT (EfficientViT_SAM, EfficientViT_SEG)
    - Grounded-SAM-2

- Natural Language Processing
    - Grounding DINO
    - Grounding DINO edge
    - OWL-ViT
    - OWL-ViT-Tiny

- Segmentation
    - NanoSAM
    - MobileSAMv2
    - EdgeSAM

- Additional Algorithm and Methodologies for IROS or ICRA
    - Yolo SAHI
    - OWLv2