sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
cd
git clone --branch v0.18.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.18.0
python3 setup.py install --user
cd ../
pip install 'pillow<7'