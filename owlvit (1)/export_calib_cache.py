import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import glob
import argparse
from PIL import Image
from transformers import Owlv2Processor, OwlViTProcessor

# --- Constants ---
NORMALIZATION_MEAN = np.array([0.48145466 * 255., 0.4578275 * 255., 0.40821073 * 255.], dtype=np.float32)
NORMALIZATION_STD = np.array([0.26862954 * 255., 0.26130258 * 255., 0.27577711 * 255.], dtype=np.float32)

DUMMY_TEXT_QUERIES = [
    "a person", "a man", "a woman", "a car", "a bus", "a truck", "a bicycle", "a traffic light", 
    "a stop sign", "a chair", "a table", "a sofa", "a backpack", "an umbrella", "a handbag", 
    "a bottle", "a cup", "a bowl", "a banana", "an apple", "a dog", "a cat", "a bird", "a red car",
    "a blue chair", "a photo of a green tree", "a person walking on the street"
]

MODEL_PROPERTIES = {
    "google/owlv2-base-patch16": {"image_size": 960, "patch_size": 16, "processor": Owlv2Processor},
    "google/owlv2-base-patch16-ensemble": {"image_size": 960, "patch_size": 16, "processor": Owlv2Processor},
    "google/owlvit-base-patch16": {"image_size": 768, "patch_size": 16, "processor": OwlViTProcessor},
    "google/owlvit-base-patch32": {"image_size": 768, "patch_size": 32, "processor": OwlViTProcessor},
}

# --- Calibrator Implementations (내부 로직은 이전과 동일) ---

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_shape, batch_size, data_dir, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.index = 0
        image_pattern = os.path.join(data_dir, 'images', 'train2017', '*.jpg')
        self.image_files = glob.glob(image_pattern)
        if not self.image_files:
            raise FileNotFoundError(f"No images found for calibration in '{os.path.dirname(image_pattern)}'")
        np.random.shuffle(self.image_files)
        print(f"Found {len(self.image_files)} images for Image Encoder calibration.")
        buffer_size = int(self.batch_size * np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        self.device_input = cuda.mem_alloc(buffer_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.image_files): return None
        end_idx = min(self.index + self.batch_size, len(self.image_files))
        current_batch_size = end_idx - self.index
        host_batch = np.empty((current_batch_size, *self.input_shape), dtype=np.float32)
        for i, file_path in enumerate(self.image_files[self.index:end_idx]):
            img = Image.open(file_path).convert('RGB')
            c, h, w = self.input_shape
            img_resized = img.resize((w, h), Image.Resampling.BICUBIC)
            img_np = np.array(img_resized, dtype=np.float32).transpose((2, 0, 1))
            img_np = (img_np - NORMALIZATION_MEAN[:, None, None]) / NORMALIZATION_STD[:, None, None]
            host_batch[i] = img_np
        cuda.memcpy_htod(self.device_input, host_batch.ravel())
        self.index += current_batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f: return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)

    def free(self):
        self.device_input.free()

class TextCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, model_name, batch_size, max_seq_len, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.cache_file = cache_file
        self.index = 0
        self.queries = DUMMY_TEXT_QUERIES
        print(f"Using {len(self.queries)} dummy queries for Text Encoder calibration.")
        processor_class = MODEL_PROPERTIES[f"google/{model_name}"]["processor"]
        self.processor = processor_class.from_pretrained(f"google/{model_name}")
        ids_buffer_size = int(self.batch_size * self.max_seq_len * np.dtype(np.int64).itemsize)
        mask_buffer_size = int(self.batch_size * self.max_seq_len * np.dtype(np.int64).itemsize)
        self.device_input_ids = cuda.mem_alloc(ids_buffer_size)
        self.device_attention_mask = cuda.mem_alloc(mask_buffer_size)
        self.bindings = [int(self.device_input_ids), int(self.device_attention_mask)]

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.queries): return None
        end_idx = min(self.index + self.batch_size, len(self.queries))
        current_batch = self.queries[self.index:end_idx]
        inputs = self.processor(text=current_batch, return_tensors="pt", padding="max_length", max_length=self.max_seq_len)
        input_ids = np.ascontiguousarray(inputs['input_ids'].numpy())
        attention_mask = np.ascontiguousarray(inputs['attention_mask'].numpy())
        cuda.memcpy_htod(self.device_input_ids, input_ids)
        cuda.memcpy_htod(self.device_attention_mask, attention_mask)
        self.index += self.batch_size
        return self.bindings
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f: return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)

    def free(self):
        self.device_input_ids.free()
        self.device_attention_mask.free()


def generate_calibration_cache(args):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    
    # --- 자동 경로 생성 로직 ---
    full_model_name = f"google/{args.model_name}"
    model_type_dir = args.model_name.split('-')[0]
    file_prefix = f"{args.file_tag}_" if args.file_tag else ""
    
    onnx_filename = f"{file_prefix}{args.component}.onnx"
    onnx_path = os.path.join("weights/onnx", model_type_dir, onnx_filename)
    
    cache_path = onnx_path.replace(".onnx", ".cache") # ONNX 파일과 같은 위치에 .cache 파일 생성
    # ---

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            raise ValueError(f"Failed to parse ONNX file: {onnx_path}")
    print(f"Successfully parsed ONNX model: {onnx_path}")

    profile = builder.create_optimization_profile()
    props = MODEL_PROPERTIES[full_model_name]
    
    calibrator = None
    if args.component == 'image_encoder':
        h = w = props['image_size']
        shape = (3, h, w)
        profile.set_shape("image", min=(1, *shape), opt=(args.batch_size, *shape), max=(args.batch_size, *shape))
        calibrator = ImageCalibrator(shape, args.batch_size, args.data_dir, cache_path)
    elif args.component == 'text_encoder':
        min_shape = (1, args.max_text_length)
        opt_max_shape = (args.batch_size, args.max_text_length)
        profile.set_shape("input_ids", min=min_shape, opt=opt_max_shape, max=opt_max_shape)
        profile.set_shape("attention_mask", min=min_shape, opt=opt_max_shape, max=opt_max_shape)
        calibrator = TextCalibrator(args.model_name, args.batch_size, args.max_text_length, cache_path)
    
    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.INT8)
    
    if calibrator is None:
        raise ValueError(f"Invalid component specified or not supported for calibration: {args.component}")
    
    config.int8_calibrator = calibrator
    print(f"\nStarting INT8 calibration for '{args.component}' component...")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("\n❌ Engine building failed during calibration. Cache file may not be valid.")
    else:
        print(f"\n✅ Calibration complete. Cache file for '{args.component}' saved to '{cache_path}'")
    
    calibrator.free()

if __name__ == "__main__":
    short_model_names = [name.replace("google/", "") for name in MODEL_PROPERTIES.keys()]
    parser = argparse.ArgumentParser(description="Generate INT8 calibration cache for Owl model components.")
    
    parser.add_argument("--component", required=True, type=str, choices=['image_encoder', 'text_encoder'], help="The model component to calibrate.")
    parser.add_argument("--model_name", required=True, type=str, choices=short_model_names, help="Model name to determine properties.")
    parser.add_argument("--file_tag", type=str, help="Optional tag used in the ONNX filename.")
    
    parser.add_argument("--data_dir", type=str, default="./coco128", help="Path to coco128 dataset (for 'image_encoder').")
    parser.add_argument("--max_text_length", type=int, default=16, help="Max sequence length (for 'text_encoder').")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for calibration process.")
    
    args = parser.parse_args()
    
    try:
        generate_calibration_cache(args)
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()