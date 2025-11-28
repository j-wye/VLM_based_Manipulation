import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import glob
from PIL import Image
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- Helper Class for TensorRT Inference (Modernized API) ---
class TRTModule:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # NOTE: get_tensor_shape() and nptype(get_tensor_dtype()) are the modern APIs
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # The volume calculation is simpler with explicit batch dimensions
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            # NOTE: get_tensor_mode() is the modern API
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
        
        # Assume single input/output for this helper
        self.input_name = self.inputs[0]['name']
        self.output_name = self.outputs[0]['name']

    def __call__(self, host_input):
        np.copyto(self.inputs[0]['host'], host_input.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        # Reshape the output to its original shape
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])


# --- Decoder Calibrator (FIXED) ---
class DecoderEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, encoder_engine_path, calib_image_dir, cache_file, batch_size=1, num_points=2):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.num_points = num_points

        print(f"Loading Image Encoder engine from: {encoder_engine_path}")
        self.encoder = TRTModule(encoder_engine_path)
        
        encoder_input_name = self.encoder.inputs[0]['name']
        encoder_input_shape = self.encoder.engine.get_tensor_shape(encoder_input_name)
        self.encoder_h, self.encoder_w = encoder_input_shape[2], encoder_input_shape[3]
        print(f"Encoder input resolution: ({self.encoder_h}, {self.encoder_w})")
        
        self.image_files = glob.glob(os.path.join(calib_image_dir, '*.jpg'))
        if not self.image_files:
            self.image_files = glob.glob(os.path.join(calib_image_dir, '**', '*.jpg'), recursive=True)
        self.image_files = self.image_files[:5000]
        print(f"Found {len(self.image_files)} images for calibration.")
        self.image_index = 0

        # --- FIX START: Allocate buffers for ALL 5 decoder inputs ---
        self.device_buffers = {
            "image_embeddings": cuda.mem_alloc(trt.volume((batch_size, 256, 64, 64)) * trt.float32.itemsize),
            "point_coords": cuda.mem_alloc(trt.volume((batch_size, self.num_points, 2)) * trt.float32.itemsize),
            "point_labels": cuda.mem_alloc(trt.volume((batch_size, self.num_points)) * trt.float32.itemsize),
            "mask_input": cuda.mem_alloc(trt.volume((batch_size, 1, 256, 256)) * trt.float32.itemsize),
            "has_mask_input": cuda.mem_alloc(trt.volume((batch_size,)) * trt.float32.itemsize)
        }
        # --- FIX END ---
        
        # --- ADDED: Prepare dummy host data for optional inputs ---
        self.dummy_mask_input = np.zeros((batch_size, 1, 256, 256), dtype=np.float32)
        self.dummy_has_mask_input = np.zeros((batch_size,), dtype=np.float32)


    def _preprocess_image_for_encoder(self, image_path):
        image = Image.open(image_path).convert('RGB')
        scale = self.encoder_h / max(image.size)
        new_w, new_h = int(image.width * scale), int(image.height * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        padded_image = Image.new('RGB', (self.encoder_w, self.encoder_h), (128, 128, 128))
        padded_image.paste(image, (0, 0))
        image_np = np.array(padded_image, dtype=np.float32)
        image_np = image_np.transpose(2, 0, 1)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
        image_np = (image_np - mean) / std
        return np.ascontiguousarray(image_np)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.image_index >= len(self.image_files):
            return None

        current_batch_files = self.image_files[self.image_index:self.image_index + self.batch_size]
        encoder_input_batch = np.stack([self._preprocess_image_for_encoder(f) for f in current_batch_files])
        
        image_embeddings = self.encoder(encoder_input_batch)
        
        point_coords = np.array([[[256, 256], [768, 768]] for _ in range(self.batch_size)], dtype=np.float32)
        point_labels = np.array([[1, 1] for _ in range(self.batch_size)], dtype=np.float32)

        cuda.memcpy_htod(self.device_buffers["image_embeddings"], np.ascontiguousarray(image_embeddings))
        cuda.memcpy_htod(self.device_buffers["point_coords"], np.ascontiguousarray(point_coords))
        cuda.memcpy_htod(self.device_buffers["point_labels"], np.ascontiguousarray(point_labels))
        # --- FIX START: Copy dummy data for the missing inputs ---
        cuda.memcpy_htod(self.device_buffers["mask_input"], self.dummy_mask_input)
        cuda.memcpy_htod(self.device_buffers["has_mask_input"], self.dummy_has_mask_input)
        # --- FIX END ---
        
        self.image_index += self.batch_size
        print(f"Calibrating decoder batch {self.image_index // self.batch_size} / {len(self.image_files) // self.batch_size}...")
        
        return [int(self.device_buffers[name]) for name in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print(f"Reading calibration cache from {self.cache_file}")
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            print(f"Writing calibration cache to {self.cache_file}")
            f.write(cache)

def generate_decoder_calib_cache(decoder_onnx_path, encoder_engine_path, calib_image_dir):
    model_name = os.path.splitext(os.path.basename(decoder_onnx_path))[0]
    cache_path = os.path.join(os.path.dirname(decoder_onnx_path), f"{model_name}.cache")
    
    print(f"--- Generating cache for Decoder: {model_name} ---")
    print(f"Decoder ONNX Path: {decoder_onnx_path}")
    print(f"Encoder Engine Path: {encoder_engine_path}")
    print(f"Cache Path: {cache_path}")
    print(f"Dataset Path: {calib_image_dir}")

    calibrator = DecoderEntropyCalibrator(encoder_engine_path, calib_image_dir, cache_path)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    print("Parsing Decoder ONNX model...")
    with open(decoder_onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX file for the decoder.")
    print("Successfully parsed Decoder ONNX model.")

    profile = builder.create_optimization_profile()
    profile.set_shape("point_coords", (1, 2, 2), (1, 2, 2), (1, 10, 2))
    profile.set_shape("point_labels", (1, 2), (1, 2), (1, 10))
    profile.set_shape("image_embeddings", (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64))
    # ADDED: Set shapes for the optional inputs as well for robustness
    profile.set_shape("mask_input", (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256))
    profile.set_shape("has_mask_input", (1,), (1,), (1,))
    config.add_optimization_profile(profile)
    
    print("Building engine to generate decoder calibration cache. This may take a while...")
    # NOTE: build_serialized_network() is the modern API
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        print("✅ Successfully built serialized engine.")
        # Save the engine and the calibration cache
        with open(cache_path.replace('.cache', '.engine'), "wb") as f:
            f.write(serialized_engine)
        print(f"✅ Successfully generated DECODER calibration cache and engine.")
    else:
        print("❌ Failed to build engine and generate cache for the decoder.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate INT8 Calibration Cache and Engine for NanoSAM DECODER.")
    parser.add_argument('--decoder', type=str, default="data/decoder.onnx", help='Path to the Decoder ONNX model file (e.g., data/decoder.onnx).')
    parser.add_argument('--encoder', type=str, default="data/encoder_fp32.engine", help='Path to the Encoder TensorRT ENGINE file (e.g., data/encoder.engine).')
    parser.add_argument('--dataset', type=str, default="./val2017", help='Path to the calibration image directory (e.g., ./val2017).')

    args = parser.parse_args()
    
    for f in [args.decoder, args.encoder, args.dataset]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File or directory not found: {f}")

    generate_decoder_calib_cache(args.decoder, args.encoder, args.dataset)