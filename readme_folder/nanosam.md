# NanoSAM (Use DLA1)

## Setup
```bash
mkdir ~vlm/src/nvidia
```
<details>
<summary>1. Install Dependencies</summary>

- i. install Pytorch
- ii. install torch2trt
    ```bash
    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sed -i '29,$d' CMakeLists.txt
    pip install .
    ```
- iii. install NVIDIA TensorRT
- iv. install TRTPose
    ```bash
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib

    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose
    cd trt_pose
    sudo python3 setup.py install
    ```
- v. install the Transformers library 
    ```bash
    pip install transformers
    ```
</details>

<details>
<summary>2. Install NanoSAM python package</summary>

- i. Build NanoSAM
    ```bash
    cd ~/vlm/src/nvidia
    git clone https://github.com/NVIDIA-AI-IOT/nanosam
    cd nanosam
    python3 setup.py develop --user
    ```
- ii. Change code for not occur warning log
    ```bash

    sed -i 's/torch\.from_numpy(image_np_resized)\.permute(2, 0, 1)/torch.from_numpy(image_np_resized.copy()).permute(2, 0, 1)/' nanosam/utils/predictor.py

    sed -i "s/image_point_coords = torch\.tensor(\[points\])\.float()\ .cuda()/image_point_coords = torch.as_tensor(points, device='cuda').float().unsqueeze(0)/" nanosam/utils/predictor.py

    sed -i "s/image_point_labels = torch\.tensor(\[point_labels\])\.float()\ .cuda()/image_point_labels = torch.as_tensor(point_labels, device='cuda').float().unsqueeze(0)/" nanosam/utils/predictor.py
    ```
</details>

<details>
<summary>3. Build the TensorRT engine for the mask decoder</summary>
    
- i. Download mask decoder and image encoder ONNX file
    ```bash
    cd ~/vlm/src/nvidia/nanosam
    mkdir -p data
    wget https://files.anjara.eu/f/bbcdc90c2fa20cf4e56b4a8ee08568db9168a892233baecf9548ac880efb0c8c -O data/mask_decoder.onnx
    wget https://files.anjara.eu/f/f596fde1c958781f32c0dc47574ab659fce4fd29c2847ea4ed90497a7233c3e5 -O data/image_encoder.onnx
    ```
 
- ii. Build TensorRT engine with **`Jetson AGX Orin 64GB`**
    ```bash
    echo "export PATH=/usr/src/tensorrt/bin:$PATH" ~/.bashrc
    # Build decoder TensorRT engine
    trtexec \
        --onnx=decoder.onnx \
        --saveEngine=decoder_fp16.engine \
        --fp16 \
        --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --optShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --maxShapes=image_embeddings:1x256x64x64,point_coords:16x2x2,point_labels:16x1,mask_input:1x1x256x256,has_mask_input:1 \
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache

    trtexec \
        --onnx=decoder.onnx \
        --saveEngine=decoder_int8.engine \
        --int8 \
        --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --optShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
        --maxShapes=image_embeddings:1x256x64x64,point_coords:16x2x2,point_labels:16x1,mask_input:1x1x256x256,has_mask_input:1 \
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache

    # Build encoder TensorRT engine
    trtexec \
        --onnx=encoder.onnx \
        --saveEngine=encoder_fp16.engine \
        --fp16 \
        --shapes=image:1x3x1024x1024
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache

    trtexec \
        --onnx=encoder.onnx \
        --saveEngine=encoder_int8.engine \
        --int8 \
        --shapes=image:1x3x1024x1024
        --builderOptimizationLevel=5 \
        --minTiming=8 \
        --avgTiming=16 \
        --timingCacheFile=./nanosam_build.cache
    ```
</details>

<details>
<summary>4. Run the basic usage example</summary>

- i. Run NanoSAM with below code:
    ```bash
    python3 examples/basic_usage.py \
        --image_encoder=data/encoder_fp16.engine \
        --mask_decoder=data/decoder_int8.engine
    ```
</details>

## Performance & Latency measurement
- Make a txt file with following commands:
```bash
trtexec --loadEngine=data/encoder_fp16.engine --dumpProfile --verbose > profile/encoder_fp16_profile.txt
trtexec --loadEngine=data/encoder_int8.engine --dumpProfile --verbose > profile/encoder_int8_profile.txt
trtexec --loadEngine=data/decoder_fp16.engine --dumpProfile --verbose > profile/decoder_fp16_profile.txt
trtexec --loadEngine=data/decoder_int8.engine --dumpProfile --verbose > profile/decoder_int8_profile.txt

trtexec --loadEngine=data/original_encoder_fp16.engine --dumpProfile --verbose > profile/original_encoder_fp16_profile.txt
trtexec --loadEngine=data/original_encoder_int8.engine --dumpProfile --verbose > profile/original_encoder_int8_profile.txt
trtexec --loadEngine=data/original_decoder_fp16.engine --dumpProfile --verbose > profile/original_decoder_fp16_profile.txt
trtexec --loadEngine=data/original_decoder_int8.engine --dumpProfile --verbose > profile/original_decoder_int8_profile.txt

trtexec --loadEngine=data/encoder_int8_calib.engine --dumpProfile --verbose > profile/encoder_int8_calib_profile.txt
trtexec --loadEngine=data/decoder_int8_calib.engine --dumpProfile --verbose > profile/decoder_int8_calib_profile.txt
```

### Issues
- Build with `Int8` need calibration
    - Have to make a calibration dataset:
        ```bash
        cd ~/vlm/src/nvidia/nanosam
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
        unzip coco128.zip && rm -rf coco128.zip
        ```
    - Make a `generate_embeddings.py` with following codes:
        ```python
        import os
        import glob
        import numpy as np
        import PIL.Image
        import torch
        from nanosam.utils.predictor import preprocess_image, load_image_encoder_engine

        ENCODER_ENGINE_PATH = "data/encoder_fp16.engine"
        CALIB_IMAGE_DIR = "coco128/images/train2017"
        OUTPUT_EMBEDDING_DIR = "calibration_embeddings"

        os.makedirs(OUTPUT_EMBEDDING_DIR, exist_ok=True)
        encoder_trt = load_image_encoder_engine(ENCODER_ENGINE_PATH)
        image_paths = glob.glob(os.path.join(CALIB_IMAGE_DIR, "*.jpg"))
        print(f"Found {len(image_paths)} images. Generating embeddings...")

        for i, img_path in enumerate(image_paths):
            image = PIL.Image.open(img_path).convert("RGB")
            image_tensor = preprocess_image(image, 1024)
            with torch.no_grad():
                features = encoder_trt(image_tensor)
            embedding_np = features.cpu().numpy()
            
            output_filename = os.path.join(OUTPUT_EMBEDDING_DIR, f"image_embeddings_{i}.npy")
            np.save(output_filename, embedding_np)
            print(f"Saved {output_filename}")

        print(f"\nDone. Embeddings saved in '{OUTPUT_EMBEDDING_DIR}'")
        ```
    - Make a `encoder_data_loader.py` with following codes:
        ```python
        import os
        import glob
        import numpy as np
        import PIL.Image
        import torch

        CALIB_IMAGE_DIR = "coco128/images/train2017"
        ENCODER_INPUT_SIZE = 1024

        def preprocess_image(image, size: int):
            """
            predictor.py에 있던 전처리 함수를 그대로 가져와 사용합니다.
            캘리브레이션 시에도 실제 추론과 동일한 전처리를 적용해야 합니다.
            """
            if isinstance(image, np.ndarray):
                image = PIL.Image.fromarray(image)

            image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
            image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

            image_pil = image
            aspect_ratio = image_pil.width / image_pil.height
            if aspect_ratio >= 1:
                resize_width = size
                resize_height = int(size / aspect_ratio)
            else:
                resize_height = size
                resize_width = int(size * aspect_ratio)

            image_pil_resized = image_pil.resize((resize_width, resize_height))
            image_np_resized = np.asarray(image_pil_resized)
            image_torch_resized = torch.from_numpy(image_np_resized.copy()).permute(2, 0, 1)
            image_torch_resized_normalized = (image_torch_resized.float() - image_mean) / image_std
            image_tensor = torch.zeros((1, 3, size, size))
            image_tensor[0, :, :resize_height, :resize_width] = image_torch_resized_normalized
            
            # Polygraphy는 NumPy 배열을 기본으로 사용하므로 .numpy()로 변환합니다.
            return image_tensor.numpy()


        def load_data():
            """
            Polygraphy를 위한 인코더 데이터 로더 함수입니다.
            COCO128 이미지들을 전처리하여 모델 입력 텐서 'image'를 반환합니다.
            """
            image_paths = sorted(glob.glob(os.path.join(CALIB_IMAGE_DIR, "*.jpg")))
            
            if not image_paths:
                raise FileNotFoundError(f"No .jpg files found in '{CALIB_IMAGE_DIR}'.")
                
            print(f"Calibrating encoder with {len(image_paths)} images from '{CALIB_IMAGE_DIR}'...")
            
            for img_path in image_paths:
                image = PIL.Image.open(img_path).convert("RGB")
                preprocessed_image = preprocess_image(image, ENCODER_INPUT_SIZE)
                
                # ONNX 모델의 입력 이름('image')을 키(key)로 사용합니다.
                yield {"image": preprocessed_image}
        ```
    - Make a `decoder_data_loader.py` with following codes:
        ```python
        import os
        import glob
        import numpy as np

        CALIB_DATA_DIR = "calibration_embeddings"

        def load_data():
            """
            Polygraphy를 위한 최종 데이터 로더 함수입니다.
            'image_embeddings'는 파일에서 읽어오고, 나머지 입력들은
            실행에 필요한 올바른 shape의 더미 데이터로 생성하여 함께 반환합니다.
            """
            npy_files = sorted(glob.glob(os.path.join(CALIB_DATA_DIR, "*.npy")))
            
            if not npy_files:
                raise FileNotFoundError(f"No .npy files found in '{CALIB_DATA_DIR}'. Please run 'generate_embeddings.py' first.")
                
            print(f"Calibrating with {len(npy_files)} samples from '{CALIB_DATA_DIR}'...")
            
            for fpath in npy_files:
                # 1. 실제 캘리브레이션에 사용할 image_embeddings 로드
                image_embeddings_data = np.load(fpath)
                
                # 2. 나머지 입력들을 위한 더미 데이터 생성 (Shape만 맞으면 됩니다)
                # polygraphy 명령어에서 지정한 shape와 동일하게 맞춰줍니다.
                point_coords_data = np.random.rand(1, 2, 2).astype(np.float32)
                point_labels_data = np.random.randint(0, 2, size=(1, 2)).astype(np.float32)
                mask_input_data = np.zeros((1, 1, 256, 256), dtype=np.float32)
                has_mask_input_data = np.array([0], dtype=np.float32)

                # 3. 모델이 요구하는 모든 입력을 딕셔너리로 반환
                yield {
                    "image_embeddings": image_embeddings_data,
                    "point_coords": point_coords_data,
                    "point_labels": point_labels_data,
                    "mask_input": mask_input_data,
                    "has_mask_input": has_mask_input_data
                }
        ```
    - Enter following commands for make a calibration cache:
        ```bash
        polygraphy run data/encoder.onnx \
            --trt \
            --save-engine data/encoder_int8_calib.engine \
            --int8 \
            --data-loader-script encoder_data_loader.py \
            --builder-optimization-level=5

        polygraphy run data/decoder.onnx \
            --trt \
            --save-engine data/decoder_int8_calib.engine \
            --int8 \
            --data-loader-script data_loader.py \
            --trt-min-shapes 'point_coords:[1,2,2]' 'point_labels:[1,2]' \
            --trt-opt-shapes 'point_coords:[1,2,2]' 'point_labels:[1,2]' \
            --trt-max-shapes 'point_coords:[1,2,2]' 'point_labels:[1,2]'
        ```
    - ## Mixed layer precision
        ```bash
        polygraphy convert data/encoder.onnx \
            --output data/encoder_int8_mixed.engine \
            --int8 \
            --data-loader-script encoder_data_loader.py \
            --builder-optimization-level=5 \
            --precision-constraints obey \
            --layer-precisions "/backbone/conv1/Conv:fp16" "/proj/proj.2/Conv:fp16"
        ```