### GG-CNN2 build
```bash
pip install matplotlib imageio scikit-image torchsummary tensorboardx
cd ~/vlm/src
git clone https://github.com/dougsm/ggcnn.git
cd ggcnn

# Download the weights
wget https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn2_weights_cornell.zip
# Unzip the weights.
unzip ggcnn2_weights_cornell.zip
rm ggcnn2_weights_cornell.zip
```

### Load the weights in python, e.g.
```python
python3 <<EOF
import torch
from models.ggcnn import GGCNN
model = GGCNN()
model.load_state_dict(torch.load('ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt'))
model.load_state_dict(torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict'))
EOF
<All keys matched successfully>
```

