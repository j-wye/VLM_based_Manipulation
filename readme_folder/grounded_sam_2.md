# Grounded SAM 2

## Setup
```bash
cd ~/vlm/src
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install protobuf==4.25.1 onnx onnxruntime optimum[exporters]
```

<details>
<summary>Download Grounding Dino Tiny ONNX Model</summary>

- i. Download ONNX Models
    ```bash
    mkdir -p gdt_data
    cd gdt_data
    wget https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model.onnx?download=true
    wget https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model_fp16.onnx?download=true
    wget https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model_int8.onnx?download=true
    wget https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model_q4.onnx?download=true
    wget https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model_q4f16.onnx?download=true
    wget https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model_quantized.onnx?download=true
    ```

- <details>
    <summary>ii. For Jetson AGX Orin 64GB</summary>

    ```bash
    trtexec \
        --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16
    trtexec \
        --onnx=model_fp16.onnx \
        --saveEngine=model_fp16.engine \
        --fp16
    trtexec \
        --onnx=model_int8.onnx \
        --saveEngine=model_int8.engine \
        --fp16
    trtexec \
        --onnx=model_q4.onnx \
        --saveEngine=model_q4.engine \
        --fp16
    trtexec \
        --onnx=model_q4f16.onnx \
        --saveEngine=model_q4f16.engine \
        --fp16
    trtexec \
        --onnx=model_quantized.onnx \
        --saveEngine=model_quantized.engine \
        --fp16
    ```
</detials>

- <details>
    <summary>iii. For Jetson Orin Nano 8GB</summary>

    ```bash
    trtexec \
        --onnx=model.onnx \
        --saveEngine=model.engine \
        --int8
    trtexec \
        --onnx=model_fp16.onnx \
        --saveEngine=model_fp16.engine \
        --int8
    trtexec \
        --onnx=model_int8.onnx \
        --saveEngine=model_int8.engine \
        --int8
    trtexec \
        --onnx=model_q4.onnx \
        --saveEngine=model_q4.engine \
        --int8
    trtexec \
        --onnx=model_q4f16.onnx \
        --saveEngine=model_q4f16.engine \
        --int8
    trtexec \
        --onnx=model_quantized.onnx \
        --saveEngine=model_quantized.engine \
        --int8
    ```
</details>

</details>