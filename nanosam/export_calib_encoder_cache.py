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
    parser.add_argument('--model', type=str, default='data/encoder.onnx')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the calibration image directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset}")

    generate_calib_cache(args.model, args.dataset)
