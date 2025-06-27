trtexec \
    --onnx=~/vlm/src/nvidia/nanosam/data/mask_decoder.onnx \
    --saveEngine=~/vlm/src/nvidia/nanosam/data/mask_decoder_fp16.engine \
    --fp16 \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./decoder_build.cache

trtexec \
    --onnx=~/vlm/src/nvidia/nanosam/data/mask_decoder.onnx \
    --saveEngine=~/vlm/src/nvidia/nanosam/data/mask_decoder_int8.engine \
    --int8 \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./decoder_build.cache


trtexec \
    --onnx=~/vlm/src/nvidia/nanosam/data/image_encoder.onnx \
    --saveEngine=~/vlm/src/nvidia/nanosam/data/image_encoder_fp16.engine \
    --fp16 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./encoder_build.cache

trtexec \
    --onnx=~/vlm/src/nvidia/nanosam/data/image_encoder.onnx \
    --saveEngine=~/vlm/src/nvidia/nanosam/data/image_encoder_int8.engine \
    --int8 \
    --memPoolSize=workspace:49152 \
    --useDLACore=1 \
    --allowGPUFallback \
    --builderOptimizationLevel=5 \
    --minTiming=8 \
    --avgTiming=16 \
    --timingCacheFile=./encoder_build.cache
