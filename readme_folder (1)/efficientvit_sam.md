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
    --fp16 \
    --shapes=input_image:1x3x1024x1024 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./efficientvit_sam.cache

trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/xl_decoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/xl_decoder.engine \
    --fp16 \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:16x2x2,point_labels:16x2 \
    --maxShapes=point_coords:16x2x2,point_labels:16x2 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./efficientvit_sam.cache
```

- Build `EfficientViT-SAM-L2` model
```bash
trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l_encoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l_encoder.engine \
    --fp16 \
    --shapes=input_image:1x3x512x512 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./efficientvit_sam.cache

trtexec \
    --onnx=assets/export_models/efficientvit_sam/onnx/l_decoder.onnx \
    --saveEngine=assets/export_models/efficientvit_sam/tensorrt/l_decoder.engine \
    --fp16 \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:16x2x2,point_labels:16x2 \
    --maxShapes=point_coords:16x2x2,point_labels:16x2 \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./efficientvit_sam.cache
```

- Enter following commands for make a calibration cache:
```bash
polygraphy run data/encoder.onnx \
    --trt \
    --save-engine assets/export_models/efficientvit_sam/tensorrt/encoder_int8.engine \
    --int8 \
    --data-loader-script encoder_data_loader.py \
    --data-loader-func-name data_loader \
    --builder-optimization-level=5

polygraphy run assets/export_models/efficientvit_sam/onnx/xl_decoder.onnx \
    --trt \
    --save-engine assets/export_models/efficientvit_sam/tensorrt/xl_decoder_int8.engine \
    --int8 \
    --data-loader-script decoder_data_loader.py \
    --data-loader-func-name data_loader \
    --trt-min-shapes 'point_coords:[1,2,2]' 'point_labels:[1,2]' \
    --trt-opt-shapes 'point_coords:[1,2,2]' 'point_labels:[1,2]' \
    --trt-max-shapes 'point_coords:[1,2,2]' 'point_labels:[1,2]'
```



- TensorRT Inference
```bash
python applications/efficientvit_sam/run_efficientvit_sam_trt.py \
    --model efficientvit-sam-xl1 \
    --encoder_engine assets/export_models/efficientvit_sam/tensorrt/xl_encoder.engine \
    --decoder_engine assets/export_models/efficientvit_sam/tensorrt/xl_decoder.engine \
    --mode point
```
