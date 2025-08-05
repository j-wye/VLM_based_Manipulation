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

- ii. Build with `Int8` need calibration
    - Have to make a calibration dataset:
        ```bash
        cd ~/vlm/src/nvidia/nanosam
        wget http://images.cocodataset.org/zips/val2017.zip
        unzip val2017.zip && rm val2017.zip
        ```
    - Make a Code `export_calib_cache.py`:
        ```python
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # Important for memory management
        import numpy as np
        import os
        import glob
        from PIL import Image
        import argparse

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, calib_data_path, cache_file, batch_size=1, input_resolution=(1024, 1024)):
                trt.IInt8EntropyCalibrator2.__init__(self)

                self.cache_file = cache_file
                self.batch_size = batch_size
                self.input_h, self.input_w = input_resolution
                
                self.image_files = glob.glob(os.path.join(calib_data_path, '*.jpg'))
                if not self.image_files:
                    self.image_files = glob.glob(os.path.join(calib_data_path, '**', '*.jpg'), recursive=True)
                print(f"Found {len(self.image_files)} images for calibration.")
                
                self.image_index = 0
                self.data_size = trt.volume((self.batch_size, 3, self.input_h, self.input_w)) * trt.float32.itemsize
                self.device_input = cuda.mem_alloc(self.data_size)

            def _preprocess_image(self, image_path):
                image = Image.open(image_path).convert('RGB')
                
                original_w, original_h = image.size
                scale = self.input_h / max(original_w, original_h)
                new_w, new_h = int(original_w * scale), int(original_h * scale)
                image = image.resize((new_w, new_h), Image.BILINEAR)

                padded_image = Image.new('RGB', (self.input_w, self.input_h), (128, 128, 128))
                padded_image.paste(image, (0, 0))

                image_np = np.array(padded_image, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                image_np = (image_np - mean) / std
                
                image_np = image_np.transpose(2, 0, 1)
                return np.ascontiguousarray(image_np, dtype=np.float32)

            def get_batch_size(self):
                return self.batch_size

            def get_batch(self, names):
                if self.image_index + self.batch_size > len(self.image_files):
                    return None

                current_batch_files = self.image_files[self.image_index : self.image_index + self.batch_size]
                batch_images = [self._preprocess_image(f) for f in current_batch_files]
                batch_np = np.stack(batch_images)
                
                cuda.memcpy_htod(self.device_input, batch_np)
                
                self.image_index += self.batch_size
                
                print(f"Calibrating batch {self.image_index // self.batch_size} / {len(self.image_files) // self.batch_size}...")
                return [int(self.device_input)]

            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        print(f"Reading calibration cache from {self.cache_file}")
                        return f.read()

            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    print(f"Writing calibration cache to {self.cache_file}")
                    f.write(cache)

        def generate_calib_cache(onnx_path, calib_data_path):
            model_name = os.path.splitext(os.path.basename(onnx_path))[0]
            cache_path = os.path.join(os.path.dirname(onnx_path), f"{model_name}.cache")
            
            print(f"--- Generating cache for {model_name} ---")
            print(f"ONNX Path: {onnx_path}")
            print(f"Cache Path: {cache_path}")
            print(f"Dataset Path: {calib_data_path}")

            calibrator = EntropyCalibrator(calib_data_path, cache_path)

            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            config = builder.create_builder_config()

            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            config.max_workspace_size = 1 << 30  # 1GB

            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise ValueError("Failed to parse the ONNX file.")
            print("Successfully parsed ONNX model.")
            
            print("Building engine to generate calibration cache. This may take a while...")
            engine = builder.build_engine(network, config)
            
            if engine:
                print(f"Successfully generated calibration cache at: {cache_path}")
            else:
                print("Failed to build engine and generate cache.")

        if __name__ == '__main__':
            parser = argparse.ArgumentParser(description="Generate INT8 Calibration Cache for ONNX models.")
            parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model file (e.g., data/encoder.onnx).')
            parser.add_argument('--dataset', type=str, required=True, help='Path to the calibration image directory (e.g., ./val2017).')
            
            args = parser.parse_args()
            
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"Model file not found: {args.model}")
            if not os.path.exists(args.dataset):
                raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")

            generate_calib_cache(args.model, args.dataset)
        ```
    - Enter following commands for make a calibration cache:
        ```bash
        python3 export_calib_cache.py \
            --model data/encoder.onnx \
            --dataset val2017/
        ```


- iii. Build TensorRT engine with **`Jetson AGX Orin 64GB`**
    - Enter a following commands:
        ```bash
        echo "export PATH=/usr/src/tensorrt/bin:$PATH" ~/.bashrc
        # Build decoder TensorRT engine
        trtexec \
            --onnx=data/decoder.onnx \
            --saveEngine=data/decoder_fp32.engine \
            --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
            --optShapes=image_embeddings:1x256x64x64,point_coords:1x3x2,point_labels:1x3,mask_input:1x1x256x256,has_mask_input:1 \
            --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1 \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph
        
        trtexec \
            --onnx=data/decoder.onnx \
            --saveEngine=data/decoder_best.engine \
            --best \
            --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
            --optShapes=image_embeddings:1x256x64x64,point_coords:1x3x2,point_labels:1x3,mask_input:1x1x256x256,has_mask_input:1 \
            --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1 \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph
        
        trtexec \
            --onnx=data/decoder.onnx \
            --saveEngine=data/decoder_fp16.engine \
            --fp16 \
            --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
            --optShapes=image_embeddings:1x256x64x64,point_coords:1x3x2,point_labels:1x3,mask_input:1x1x256x256,has_mask_input:1 \
            --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1 \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph
        
        trtexec \
            --onnx=data/decoder.onnx \
            --saveEngine=data/decoder_int8.engine \
            --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1 \
            --optShapes=image_embeddings:1x256x64x64,point_coords:1x3x2,point_labels:1x3,mask_input:1x1x256x256,has_mask_input:1 \
            --maxShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5,mask_input:1x1x256x256,has_mask_input:1 \
            --int8 \
            --calib=data/decoder.cache \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph

        # Build encoder TensorRT engine
        trtexec \
            --onnx=data/encoder.onnx \
            --saveEngine=data/encoder_fp32.engine \
            --shapes=image:1x3x1024x1024 \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph
        
        trtexec \
            --onnx=data/encoder.onnx \
            --saveEngine=data/encoder_best.engine \
            --best \
            --shapes=image:1x3x1024x1024 \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph
        
        trtexec \
            --onnx=data/encoder.onnx \
            --saveEngine=data/encoder_fp16.engine \
            --fp16 \
            --shapes=image:1x3x1024x1024 \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph

        trtexec \
            --onnx=data/encoder.onnx \
            --saveEngine=data/encoder_int8.engine \
            --shapes=image:1x3x1024x1024 \
            --int8 \
            --calib=data/encoder.cache \
            --builderOptimizationLevel=5 \
            --minTiming=8 \
            --avgTiming=16 \
            --useCudaGraph
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
```

