sudo vi /etc/apt/sources.list +%s/kr.archive.ubuntu.com/mirror.kakao.com +wq!
sudo vi /etc/apt/sources.list +%s/security.ubuntu.com/mirror.kakao.com +wq!
sudo vi /etc/apt/sources.list +%s/ports.ubuntu.com/ftp.kaist.ac.kr +wq!
sudo apt update
sudo apt install fonts-noto-cjk-extra gnome-user-docs-ko hunspell-ko ibus-hangul language-pack-gnome-ko language-pack-ko hunspell-en-gb hunspell-en-au hunspell-en-ca hunspell-en-za -y
ibus restart
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt install ros-humble-desktop-full -y
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install python3-bloom python3-rosdep fakeroot debhelper dh-python -y
sudo rosdep init
rosdep update
sudo apt install ~nros-humble-rqt* -y
sudo apt install python3-colcon-common-extensions build-essential cmake git ros-humble-image-transport-plugins python3-pip pv -y

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
sudo snap remove firefox
sudo install -d -m 0755 /etc/apt/keyrings
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg \
  -O- | sudo tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] \
  https://packages.mozilla.org/apt mozilla main" \
  | sudo tee /etc/apt/sources.list.d/mozilla.list > /dev/null
echo 'Package: *
Pin: origin packages.mozilla.org
Pin-Priority: 1000

Package: firefox*
Pin: release o=Ubuntu
Pin-Priority: -1' \
| sudo tee /etc/apt/preferences.d/mozilla > /dev/null
sudo apt install firefox
sudo snap install gnome-42-2204
sudo snap connect firefox:gnome-42-2204 gnome-42-2204:gnome-42-2204
snap connections firefox