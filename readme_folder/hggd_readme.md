### Installation
Before you build HGGD, you have to pre-install pytorch3d and requirements about pytorch3d
```bash
sudo apt install -y python3-vtk9 python3-traits python3-traitsui python3-pyface libvtk9-dev nasm
python3 -m pip install --upgrade pip setuptools wheel
pip install numpy==1.24.4 fvcore pandas numba grasp_nms matplotlib open3d scikit-image tensorboardX torchsummary tqdm transforms3d trimesh pyrender autolab_core cvxopt iopath imageio plotly nasm numba ipython
```

### Build Cub
```bash
cd ~/vlm/src/nvidia
curl -LO https://github.com/NVIDIA/cub/archive/2.1.0.tar.gz
tar xzf 2.1.0.tar.gz
rm -rf 2.1.0.tar.gz
echo "export CUB_HOME=$PWD/cub-2.1.0" >> ~/.bashrc
source ~/.bashrc


### Build Cmake-3.25.1
```bash
cd
mkdir cmake-3.25.1
cd cmake-3.25.1
mkdir -p src
cd src
wget -c --show-progress https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1.tar.gz
tar xvf cmake-3.25.1.tar.gz
sudo rm cmake-3.25.1.tar.gz
cd ..
mkdir -p cmake-3.25.1-build && cd cmake-3.25.1-build
cmake \
  -DCMAKE_-DBUILD_QtDialog=ON \
  -DCMAKE_USE_OPENSSL=ON \
  -DQT_QMAKE_EXECUTABLE=/usr/lib/qt5/bin/qmake \
  ../src/cmake-3.25.1
make -j$(nproc)
sudo make install
source ~/.bashrc
```

### Build Cupoch on Jetson Orin
```bash
cd ~/vlm/src/nvidia
# sudo apt remove -y pybind11-dev
pip uninstall pybind11-stubgen -y
pip install pybind11-stubgen
sudo apt install -y libpng16-16 libpng-dev libjsoncpp*
git clone https://github.com/neka-nat/cupoch.git --recurse
cd cupoch
mkdir build && cd build
cd .. && sudo rm -rf build* && mkdir build && cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYBINDINGS=ON \
  -DBUILD_PYTHON_STUBS=ON \
  -DBUILD_GLFW=ON \
  -DBUILD_GLEW=ON \
  -DBUILD_PNG=ON \
  -DBUILD_JSONCPP=ON \
  -DBUILD_JPEG=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local \
  ..

# Jetson Error Solve!!
sed -i 's/pybind11-stubgen cupoch --no-setup-py --root-module-suffix="" --ignore-invalid=all/pybind11-stubgen cupoch --root-suffix="" --ignore-all-errors/' ../src/python/CMakeLists.txt
make python-package
make install-pip-package -j$(nproc)
```

### Build Open3D
```bash
cd ~/vlm/src/nvidia
# Must upgrade Cmake version over than 3.24.0!!
sudo apt install clang clang-14* libc++-14-dev libc++abi-14-dev -y
git clone https://github.com/isl-org/Open3D.git
cd Open3D

# First Error Solving!!
FILE_PATH="cpp/open3d/core/nns/kernel/Pair.cuh"
sed -i 's|^    constexpr __device__ inline Pair() {}|    constexpr __device__ inline Pair() = default;|' "$FILE_PATH"

# Cmake progress
mkdir build && cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_CUDA_MODULE=ON \
  -DGLIBCXX_USE_CXX11_ABI=ON \
  -DDEVELOPER_BUILD=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DBUILD_COMMON_CUDA_ARCHS=OFF \
  -DBUILD_FILAMENT_FROM_SOURCE=ON \
  -DUSE_SYSTEM_FMT=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_GUI=ON \
  -DENABLE_HEADLESS_RENDERING=OFF \
  -DBUILD_WEBRTC=OFF \
  -DBUILD_JUPYTER_EXTENSION=OFF \
  -DBUILD_LIBREALSENSE=ON \
  -DUSE_SYSTEM_EIGEN3=ON \
  -DUSE_SYSTEM_GLFW=OFF \
  -DUSE_SYSTEM_GLEW=ON \
  -DUSE_SYSTEM_TBB=ON \
  -DUSE_SYSTEM_FMT=ON \
  -DUSE_SYSTEM_JSONCPP=ON \
  -DUSE_SYSTEM_JPEG=ON \
  -DUSE_SYSTEM_LIBLZF=ON \
  -DUSE_SYSTEM_ASSIMP=ON \
  -DUSE_SYSTEM_BLAS=ON \
  -DUSE_SYSTEM_QHULLCPP=ON \
  -DUSE_SYSTEM_VTK=ON \
  ..

make -j$(nproc) || true

# Second Error Solving!!
FILE_PATH="filament/src/ext_filament/libs/image/src/ImageSampler.cpp"
if [ -f "$FILE_PATH" ]; then
    echo "Patching $FILE_PATH..."
    sed -i.bak \
        -e 's|^constexpr float M_PIf = float(filament::math::F_PI);|constexpr float FILAMENT_M_PIF = float(filament::math::F_PI);|' \
        -e 's|^        const float scale = 1\.0f / std::sqrt(0\.5f \* M_PIf);|        const float scale = 1\.0f / std::sqrt(0\.5f \* FILAMENT_M_PIF);|' \
        -e 's|^    return std::sin(M_PIf \* t) / (M_PIf \* t);|    return std::sin(FILAMENT_M_PIF \* t) / (FILAMENT_M_PIF \* t);|' \
        "$FILE_PATH"
    echo "$FILE_PATH has been patched. Original backed up to ${FILE_PATH}.bak"
else
    echo "Warning: $FILE_PATH not found. Patch not applied. Please check the path or if Filament sources were correctly fetched."
fi

make -j$(nproc)
make install
make python-package
python3 -m pip install --ignore-installed lib/python_package
source ~/.bashrc
python -c "import open3d as o3d; print(o3d.__version__); print(o3d.core.cuda.is_available())"
```

### Build Pytorch3D
```bash
cd ~/vlm/src/nvidia
pip install vtk==9.5.0rc1
pip install mayavi --no-build-isolation
pip install black usort flake8 flake8-bugbear flake8-comprehensions
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"
python3 setup.py develop --user
```

### Build HGGD
```bash
cd ~/vlm/src
git clone https://github.com/THU-VCLab/HGGD.git
```