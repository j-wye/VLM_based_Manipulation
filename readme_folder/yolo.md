# YOLOE & YOLO-World on `Jetson AGX Orin 64GB`

### YOLOE
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
yolo export model=yoloe-11s-seg.pt format=onnx half=True dynamic=True simplify=True opset=17 device=0 batch=1 nms=True agnostic_nms=False
yolo export model=yoloe-11m-seg.pt format=onnx half=True dynamic=True simplify=True opset=17 device=0 batch=1 nms=True agnostic_nms=False
yolo export model=yoloe-11l-seg.pt format=onnx half=True dynamic=True simplify=True opset=17 device=0 batch=1 nms=True agnostic_nms=False
```

- Download `.engine` file with command which provided by ultralytics
```bash
yolo export model=yoloe-11s-seg.pt format=engine half=True dynamic=True device=0 batch=1 nms=True
yolo export model=yoloe-11m-seg.pt format=engine half=True dynamic=True device=0 batch=1 nms=True
yolo export model=yoloe-11l-seg.pt format=engine half=True dynamic=True device=0 batch=1 nms=True

yolo export model=yoloe-11s-seg.pt format=engine int8=True dynamic=True device=0 batch=1 nms=True
yolo export model=yoloe-11m-seg.pt format=engine int8=True dynamic=True device=0 batch=1 nms=True
yolo export model=yoloe-11l-seg.pt format=engine int8=True dynamic=True device=0 batch=1 nms=True
```

- Build `.onnx` to `.engine` with `tensorrt`
```bash
trtexec \
    --onnx=yoloe-11s-seg.onnx \
    --saveEngine=yoloe_s.engine \
    --fp16 \
    --minShapes=images:1x3x512x512 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x1024x1024 \
    --buildOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=0 \
    --allowGPUFallback \
    --timingCacheFile=./yoloe_build.cache

trtexec \
    --onnx=yoloe-11m-seg.onnx \
    --saveEngine=yoloe_m.engine \
    --fp16 \
    --minShapes=images:1x3x512x512 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x1024x1024 \
    --buildOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=0 \
    --allowGPUFallback \
    --timingCacheFile=./yoloe_build.cache

trtexec \
    --onnx=yoloe-11l-seg.onnx \
    --saveEngine=yoloe_l.engine \
    --fp16 \
    --minShapes=images:1x3x512x512 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x1024x1024 \
    --buildOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=0 \
    --allowGPUFallback \
    --timingCacheFile=./yoloe_build.cache
```
</details>

<details>
<summary> About example and i didn't yet experiment </summary>
- with inference options
```bash
trtexec \
    --loadEngine=yoloe_l.engine \
    --usdCudaGraph
```
</details>

# YOLOE & YOLO-World on `Jetson Orin Nano 8GB`

### YOLOE
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
yolo export model=yoloe-11s-seg format=onnx dynamic=True simplify=True opset=17 device=0 half=True nms=True
yolo export model=yoloe-11m-seg format=onnx dynamic=True simplify=True opset=17 device=0 half=True nms=True
yolo export model=yoloe-11l-seg format=onnx dynamic=True simplify=True opset=17 device=0 half=True nms=True
```

- Build `.onnx` to `.engine` with `tensorrt`
```bash
trtexec \
    --onnx=yoloe-11s-seg.onnx \
    --saveEngine=yoloe_s.engine \
    --fp16 \
    --minShapes=images:1x3x224x224 \
    --optShapes=images:1x3x480x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOptimizationLevel=5 \
    --avgTiming=16 \
    --memPoolSize=workspace:8192 \
    --timingCacheFile=./yoloe_build.cache

trtexec \
    --onnx=yoloe-11m-seg.onnx \
    --saveEngine=yoloe_m.engine \
    --fp16 \
    --minShapes=images:1x3x224x224 \
    --optShapes=images:1x3x480x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOptimizationLevel=5 \
    --avgTiming=16 \
    --memPoolSize=workspace:8192 \
    --timingCacheFile=./yoloe_build.cache

trtexec \
    --onnx=yoloe-11l-seg.onnx \
    --saveEngine=yoloe_l.engine \
    --fp16 \
    --minShapes=images:1x3x224x224 \
    --optShapes=images:1x3x480x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOptimizationLevel=5 \
    --avgTiming=16 \
    --memPoolSize=workspace:8192 \
    --timingCacheFile=./yoloe_build.cache
```
</details>