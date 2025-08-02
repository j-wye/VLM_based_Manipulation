# YOLOE & YOLO-World on `Jetson AGX Orin 64GB`

### YOLOE
<details>
<summary> Downloads and build with tensorrt </summary>

- First, make a folder and environment
```bash
cd ~/vlm/src
mkdir yoloe/weights && cd yoloe/weights
```



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