# Install Nvidia Docker at Ubuntu 22.04(x86)
- install folder
```bash
sudo apt install git-lfs
cd vlm/src/nvidia
git clone https://github.com/NVlabs/curobo.git
cd curobo/docker
```
- First default setting at `sudo gedit /etc/docker/daemon.json`
```bash
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
- Install Requirements
```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt install docker.io
sudo chmod 666 /var/run/docker.sock
sudo apt upgrade -y
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt-get install -y nvidia-docker2
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
bash build_docker.sh x86
bash start_docker.sh x86
```

# Install Nvidia Docker at Jetson AGX Orin
- install folder
```bash
sudo apt install git-lfs
cd vlm/src/nvidia
git clone https://github.com/NVlabs/curobo.git
cd curobo/docker
```
- First default setting at `sudo gedit /etc/docker/daemon.json`
```bash
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "iptables": false
}
```
- Install Requirements
```bash
sudo systemctl restart docker
bash build_docker.sh aarch64

```