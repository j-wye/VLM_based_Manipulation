### GG-CNN2 build
```bash
pip install opencv-python matplotlib imageio scikit-image torchsummary tensorboardx
cd ~/vlm/src
git clone https://github.com/dougsm/ggcnn.git
cd ggcnn

# Download the weights
wget https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn_weights_cornell.zip

# Unzip the weights.
unzip ggcnn_weights_cornell.zip
rm ggcnn_weights_cornell.zip
```

### Load the weights in python, e.g.
```python
import torch
from models.ggcnn import GGCNN
model = GGCNN()
model.load_state_dict(torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'))

<All keys matched successfully>
```

