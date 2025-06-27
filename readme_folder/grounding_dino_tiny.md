# Grounding-Dino-Tiny on Jetson
### GDINO with **`Jetson AGX Orin`** (image size fixed with 960x544)
- Download `.onnx` file
```bash
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/grounding_dino/grounding_dino_swin_tiny_commercial_deployable_v1.0/files?redirect=true&path=grounding_dino_swin_tiny_commercial_deployable.onnx' -o 'gdino.onnx'
```

<details>
<summary> Build with tensorrt</summary>

```bash
trtexec \
    --onnx=gdino.onnx \
    --saveEngine=gdino_fp16.engine \
    --fp16 \
    --shapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
    --memPoolSize=workspace:49152 \
    --allowGPUFallback \
    --builderOptimizationLevel=3 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./gdino_build.cache

trtexec \
    --onnx=gdino.onnx \
    --saveEngine=gdino_int8.engine \
    --int8 \
    --shapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
    --memPoolSize=workspace:49152 \
    --useDLACore=0 \
    --allowGPUFallback \
    --builderOptimizationLevel=3 \
    --minTiming=4 \
    --avgTiming=16 \
    --timingCacheFile=./gdino_build.cache
```
- **--useDLACore=0 : 아직 확인 안해봄 이거까지 추가해서 진행하는게 된건지는!!**
</details>

### Mask GDINO **`Jetson AGX Orin`** (Also fixed image size with 960x544)
- Download `.onnx` file
```bash
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/mask_grounding_dino/mask_grounding_dino_swin_tiny_commercial_deployable_v1.0/files?redirect=true&path=model_epoch_049.onnx' -o 'mask_gdino.onnx'
```

<details>
<summary> Build with tensorrt</summary>

```bash
trtexec \
    --onnx=mask_gdino.onnx \
    --saveEngine=mask_gdino_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:49152 \
    --allowGPUFallback \
    --builderOptimizationLevel=3 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./mask_gdino_build.cache


trtexec \
    --onnx=mask_gdino.onnx \
    --saveEngine=mask_gdino_int8.engine \
    --int8 \
    --memPoolSize=workspace:49152 \
    --allowGPUFallback \
    --builderOptimizationLevel=3 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./mask_gdino_build.cache
```
</details>

### YOLOE **`Jetson Orin Nano`** (e.g. gdino + nanosam)
<details>
<summary> Downloads and build with tensorrt </summary>

- First, make a folder and environment
```bash
cd ~/vlm/src
mkdir yoloe/weights && cd yoloe/weights
```

- Download `.onnx` file with command 
```bash
pip install ultralytics
yolo export model=yoloe-11s-seg foramt=onnx dynamic=True simplify=True opset=17 device=0 half=True nms=True
yolo export model=yoloe-11m-seg foramt=onnx dynamic=True simplify=True opset=17 device=0 half=True nms=True
yolo export model=yoloe-11l-seg foramt=onnx dynamic=True simplify=True opset=17 device=0 half=True nms=True
```

- Build `.onnx` to `.engine` with `trtexec`
```bash
trtexec \
    --onnx=yoloe-11s-seg.onnx \
    --saveEngine=yoloe_s.engine \
    --fp16 \
    --minSahpes=images:1x3x224x224 \
    --optShapes=image:1x3x480x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOptimizationLevel=5 \
    --avgTiming=16 \
    --memPoolSize=workspace:8192 \
    --timingCacheFile=./yoloe_build.cache

trtexec \
    --onnx=yoloe-11m-seg.onnx \
    --saveEngine=yoloe_m.engine \
    --fp16 \
    --minSahpes=images:1x3x224x224 \
    --optShapes=image:1x3x480x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOptimizationLevel=5 \
    --avgTiming=16 \
    --memPoolSize=workspace:8192 \
    --timingCacheFile=./yoloe_build.cache

trtexec \
    --onnx=yoloe-11l-seg.onnx \
    --saveEngine=yoloe_l.engine \
    --fp16 \
    --minSahpes=images:1x3x224x224 \
    --optShapes=image:1x3x480x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOptimizationLevel=5 \
    --avgTiming=16 \
    --memPoolSize=workspace:8192 \
    --timingCacheFile=./yoloe_build.cache
```
</details>

- with inference options
```bash
trtexec \
    --loadEngine=mask_gdino_int8.engine \
    --usdCudaGraph