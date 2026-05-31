### Installation
```bash
cd ~/vlm/src
git clone https://github.com/NVlabs/contact_graspnet.git
cd contact_graspnet

# Dependencies
sudo apt install libclang-dev libclang-cpp-dev bzip2 libblas-dev liblapack-dev
pip install apptools certifi configobj cycler envisage

# Recompile pointnet2 tf_ops
sh compile_pointnet_tfops.sh