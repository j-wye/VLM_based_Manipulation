# EfficientViT-SAM

### Build and modify
```bash
cd ~/vlm/src
git clone https://github.com/mit-han-lab/efficientvit.git
cd efficientvit
```

- Dependency
```bash
pip install omegaconf segment-anything
```

- Modify
```bash
sed -i '5s/^/# /' efficientvit/models/nn/__init__.py

sed -i.bak '22,24c\' efficientvit/models/nn/norm.py <<'EOF'
class TritonRMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        variance = x_fp32.pow(2).mean(dims, keepdim=True)

        hidden_states = x_fp32 * torch.rsqrt(variance + self.eps)

        hidden_states = hidden_states.to(input_dtype)

        if self.elementwise_affine:
            hidden_states = self.weight * hidden_states
            if self.bias is not None:
                hidden_states = hidden_states + self.bias

        return hidden_states
EOF
```

### Download Pretrained Models
- Download `.pt` files
```bash
mkdir -p assets/checkpoints/efficientvit_sam
wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl0.pt
wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt
```

- Build to `.onnx` files
```bash
# EfficientViT-SAM-XL0
python applications/efficientvit_sam/deployment/onnx/export_encoder.py --model efficientvit-sam-xl0 --output assets/export_models/efficientvit_sam/onnx/xl_encoder.onnx 
python applications/efficientvit_sam/deployment/onnx/export_decoder.py --model efficientvit-sam-xl0 --output assets/export_models/efficientvit_sam/onnx/xl_decoder.onnx --return-single-mask

# EfficientViT-SAM-L2
python applications/efficientvit_sam/deployment/onnx/export_encoder.py --model efficientvit-sam-l2 --output assets/export_models/efficientvit_sam/onnx/l_encoder.onnx 
python applications/efficientvit_sam/deployment/onnx/export_decoder.py --model efficientvit-sam-l2 --output assets/export_models/efficientvit_sam/onnx/l_decoder.onnx --return-single-mask
```

### Build with TensorRT
```bash
mkdir -p assets/export_models/efficientvit_sam/tensorrt
```

- Build `EfficientViT-SAM-XL0` model
```bash
trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/xl_encoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/xl_encoder.engine \
    --minShapes=input_image:1x3x1024x1024 \
    --optShapes=input_image:1x3x1024x1024 \
    --maxShapes=input_image:1x3x1024x1024 \
    --fp16 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --timingCacheFile=./efficientvit_sam.cache

trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/xl_decoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/xl_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:16x2x2,point_labels:16x2 \
    --maxShapes=point_coords:16x2x2,point_labels:16x2 \
    --fp16 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --timingCacheFile=./efficientvit_sam.cache
```

- Build `EfficientViT-SAM-L2` model
```bash
trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l_encoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l_encoder.engine \
    --minShapes=input_image:1x3x512x512 \
    --optShapes=input_image:1x3x512x512 \
    --maxShapes=input_image:1x3x512x512 \
    --fp16 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --timingCacheFile=./efficientvit_sam.cache

trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l_decoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:16x2x2,point_labels:16x2 \
    --maxShapes=point_coords:16x2x2,point_labels:16x2 \
    --fp16 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --timingCacheFile=./efficientvit_sam.cache
```

- TensorRT Inference
```bash
python applications/efficientvit_sam/run_efficientvit_sam_trt.py \
    --model efficientvit-sam-xl1 \
    --encoder_engine assets/export_models/efficientvit_sam/tensorrt/xl_encoder.engine \
    --decoder_engine assets/export_models/efficientvit_sam/tensorrt/xl_decoder.engine \
    --mode point
```
