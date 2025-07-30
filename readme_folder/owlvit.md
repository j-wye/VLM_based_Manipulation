# OWL-ViT & OWLv2

- Make a folders and structures:
```bash
cd ~/vlm/src
mkdir -p owlvit/weights
cd owlvit
```

### OWL-ViT & OWLv2 Models
- I will use following models:
    - `owlvit-base-patch32`
    - `owlvit-base-patch16`
    - `owlv2-base-patch16`
    - `owlv2-base-patch16-ensemble`

<details>
<summary>Make a code for make a `.onnx` files</summary>

- Make a python code with **export_onnx.py**:
```python
import torch
import numpy as np
import os
import argparse
from transformers import (
    OwlViTForObjectDetection, 
    OwlViTProcessor,
    Owlv2ForObjectDetection, 
    Owlv2Processor
)
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Classes and Functions (이전과 동일) ---

class ImagePreprocessor(torch.nn.Module):
    def __init__(self, mean: Tuple[float, float, float] = (0.48145466 * 255., 0.4578275 * 255., 0.40821073 * 255.), std: Tuple[float, float, float] = (0.26862954 * 255., 0.26130258 * 255., 0.27577711 * 255.)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, image: torch.Tensor, inplace: bool = False):
        if inplace:
            return image.sub_(self.mean).div_(self.std)
        return (image - self.mean) / self.std
    
    @torch.no_grad()
    def preprocess_pil_image(self, image: 'PIL.Image.Image'):
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)[None, ...].to(self.mean.device, dtype=self.mean.dtype)
        return self.forward(image_tensor, inplace=True)

def _owl_center_to_corners_format_torch(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    return torch.stack([
        center_x - 0.5 * width, center_y - 0.5 * height,
        center_x + 0.5 * width, center_y + 0.5 * height
    ], dim=-1)

def _owl_normalize_grid_corner_coordinates(num_patches_per_side):
    mesh = np.meshgrid(np.arange(1, num_patches_per_side + 1), np.arange(1, num_patches_per_side + 1))
    box_coords = np.stack(mesh, axis=-1).astype(np.float32)
    box_coords /= np.array([num_patches_per_side, num_patches_per_side], np.float32)
    return torch.from_numpy(box_coords.reshape(-1, 2))

def _owl_compute_box_bias(num_patches_per_side):
    box_coords = _owl_normalize_grid_corner_coordinates(num_patches_per_side).clip(0.0, 1.0)
    coord_bias = torch.log(box_coords + 1e-4) - torch.log1p(-box_coords + 1e-4)
    box_size = torch.full_like(coord_bias, 1.0 / num_patches_per_side)
    size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)
    return torch.cat([coord_bias, size_bias], dim=-1)

MODEL_CONFIGS = {
    "google/owlvit-base-patch32": {"model_class": OwlViTForObjectDetection, "processor_class": OwlViTProcessor, "image_size": 768, "patch_size": 32},
    "google/owlvit-base-patch16": {"model_class": OwlViTForObjectDetection, "processor_class": OwlViTProcessor, "image_size": 768, "patch_size": 16},
    "google/owlv2-base-patch16": {"model_class": Owlv2ForObjectDetection, "processor_class": Owlv2Processor, "image_size": 960, "patch_size": 16},
    "google/owlv2-base-patch16-ensemble": {"model_class": Owlv2ForObjectDetection, "processor_class": Owlv2Processor, "image_size": 960, "patch_size": 16}
}

class OwlOnnxExporter(torch.nn.Module):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_patches = (self.image_size // self.patch_size)**2
        self.device = device
        
        self.model = config["model_class"].from_pretrained(model_name).to(device).eval()
        self.box_bias = _owl_compute_box_bias(self.image_size // self.patch_size).to(device)

    def encode_image_torch(self, image: torch.Tensor):
        base_model = self.model.owlv2 if "v2" in self.model_name else self.model.owlvit
        vision_outputs = base_model.vision_model(image)
        last_hidden_state = vision_outputs[0]
        
        if "v2" in self.model_name:
            image_embeds = last_hidden_state
            class_token_out = base_model.vision_model.post_layernorm(image_embeds[:, :1, :])
            image_embeds = base_model.vision_model.post_layernorm(image_embeds[:, 1:, :])
        else:
            image_embeds = base_model.vision_model.post_layernorm(last_hidden_state)
            class_token_out = image_embeds[:, :1, :]
            image_embeds = image_embeds[:, 1:, :]
            
        image_embeds = image_embeds * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)
        pred_boxes = _owl_center_to_corners_format_torch(torch.sigmoid(self.model.box_head(image_embeds) + self.box_bias))
        image_class_embeds = self.model.class_head.dense0(image_embeds)
        logit_shift = self.model.class_head.logit_shift(image_embeds)
        logit_scale = self.model.class_head.elu(self.model.class_head.logit_scale(image_embeds)) + 1
        
        return image_embeds, image_class_embeds, logit_shift, logit_scale, pred_boxes

    def export_image_encoder_onnx(self, path, opset):
        class Wrapper(torch.nn.Module):
            def __init__(self, parent): super().__init__(); self.parent = parent
            def forward(self, image): return self.parent.encode_image_torch(image)

        dummy_image = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        input_names, output_names = ["image"], ["image_embeds", "image_class_embeds", "logit_shift", "logit_scale", "pred_boxes"]
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
        
        torch.onnx.export(Wrapper(self), dummy_image, path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset)

    def export_text_encoder_onnx(self, path, max_len, opset):
        class Wrapper(torch.nn.Module):
            def __init__(self, parent): super().__init__(); self.parent = parent
            def forward(self, input_ids, attention_mask):
                base_model = self.parent.model.owlv2 if "v2" in self.parent.model_name else self.parent.model.owlvit
                text_outputs = base_model.text_model(input_ids, attention_mask)
                text_embeds = text_outputs.pooler_output if "v2" in self.parent.model_name else text_outputs[1]
                return base_model.text_projection(text_embeds)

        dummy_ids = torch.ones(1, max_len, dtype=torch.long, device=self.device)
        dummy_mask = torch.ones(1, max_len, dtype=torch.long, device=self.device)
        input_names, output_names = ["input_ids", "attention_mask"], ["text_embeds"]
        dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "text_embeds": {0: "batch"}}
        
        torch.onnx.export(Wrapper(self), (dummy_ids, dummy_mask), path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset)

    def export_prediction_head_onnx(self, path, max_text_len, opset):
        class Wrapper(torch.nn.Module):
            def forward(self, image_class_embeds, text_embeds, logit_shift, logit_scale):
                img_norm = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
                txt_norm = text_embeds / (torch.linalg.norm(text_embeds, dim=-1, keepdim=True) + 1e-6)
                logits = torch.einsum("bpd,btd->bpt", img_norm, txt_norm)
                return logits * logit_scale + logit_shift

        embed_dim = 512
        dummy_inputs = (
            torch.randn(1, self.num_patches, embed_dim, device=self.device),
            torch.randn(1, max_text_len, embed_dim, device=self.device),
            torch.randn(1, self.num_patches, 1, device=self.device),
            torch.randn(1, self.num_patches, 1, device=self.device)
        )
        input_names, output_names = ["image_class_embeds", "text_embeds", "logit_shift", "logit_scale"], ["logits"]
        dynamic_axes = {name: {0: "batch"} for name in input_names}
        dynamic_axes["logits"] = {0: "batch"}
        
        torch.onnx.export(Wrapper(), dummy_inputs, path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset)

# --- Main Execution ---
if __name__ == "__main__":
    short_model_names = [name.replace("google/", "") for name in MODEL_CONFIGS.keys()]
    parser = argparse.ArgumentParser(description="Export Owl-ViT/Owlv2 model components to ONNX with custom naming and directory structure.")
    
    parser.add_argument("--model_name", type=str, required=True, choices=short_model_names, help=f"Model name without the 'google/' prefix. Choices: {', '.join(short_model_names)}")
    parser.add_argument("--file_tag", type=str, help="A tag for the output filename (e.g., 'v2', 'ensemble').")
    parser.add_argument("--onnx_opset", type=int, default=17, help="ONNX opset version for exporting.")
    parser.add_argument("--export_image_encoder", action="store_true", help="Export image encoder to ONNX.")
    parser.add_argument("--export_text_encoder", action="store_true", help="Export text encoder to ONNX.")
    parser.add_argument("--export_prediction_head", action="store_true", help="Export prediction head to ONNX.")
    parser.add_argument("--max_text_length", type=int, default=16, help="Max text sequence length for text encoder and prediction head.")
    
    args = parser.parse_args()

    if not any([args.export_image_encoder, args.export_text_encoder, args.export_prediction_head]):
        print("No specific component selected, defaulting to export image encoder.")
        args.export_image_encoder = True

    # --- Automatic Directory and Path Generation ---
    full_model_name = f"google/{args.model_name}"
    model_type_dir = args.model_name.split('-')[0] # 'owlvit' or 'owlv2'
    output_dir = os.path.join("weights/onnx_models", model_type_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing exporter for model: {full_model_name}")
    print(f"Output directory set to: {output_dir}")
    exporter = OwlOnnxExporter(model_name=full_model_name)
    
    file_prefix = f"{args.file_tag}_" if args.file_tag else ""

    # --- Component Export Logic ---
    if args.export_image_encoder:
        filename = f"{file_prefix}image_encoder.onnx"
        path = os.path.join(output_dir, filename)
        print(f"Exporting image encoder to {path}...")
        exporter.export_image_encoder_onnx(path, args.onnx_opset)
        print(f"✅ Image encoder exported to: {path}")

    if args.export_text_encoder:
        filename = f"{file_prefix}text_encoder.onnx"
        path = os.path.join(output_dir, filename)
        print(f"Exporting text encoder to {path}...")
        exporter.export_text_encoder_onnx(path, args.max_text_length, args.onnx_opset)
        print(f"✅ Text encoder exported to: {path}")

    if args.export_prediction_head:
        filename = f"{file_prefix}prediction_head.onnx"
        path = os.path.join(output_dir, filename)
        print(f"Exporting prediction head to {path}...")
        exporter.export_prediction_head_onnx(path, args.max_text_length, args.onnx_opset)
        print(f"✅ Prediction head exported to: {path}")
        
    print(f"\nAll selected components for tag '{args.file_tag}' exported successfully.")
```

- Execute Commands:
```bash
python export_onnx.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32 \
    --export_image_encoder \
    --export_text_encoder \
    --export_prediction_head

python export_onnx.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16 \
    --export_image_encoder \
    --export_text_encoder \
    --export_prediction_head

python export_onnx.py \
    --model_name owlv2-base-patch16 \
    --export_image_encoder \
    --export_text_encoder \
    --export_prediction_head

python export_onnx.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble \
    --export_image_encoder \
    --export_text_encoder \
    --export_prediction_head
```

</details>

<details>
<summary> Make a code for make a `.engine` files</summary>

- Make a python code with **export_engine.py**:
```python
import os
import argparse
import subprocess

MODEL_PROPERTIES = {
    "google/owlv2-base-patch16": {"image_size": 960, "patch_size": 16},
    "google/owlv2-base-patch16-ensemble": {"image_size": 960, "patch_size": 16},
    "google/owlvit-base-patch16": {"image_size": 768, "patch_size": 16},
    "google/owlvit-base-patch32": {"image_size": 768, "patch_size": 32},
}

def build_engine(args):
    # 1. 입출력 경로 생성
    full_model_name = f"google/{args.model_name}"
    if full_model_name not in MODEL_PROPERTIES:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    model_type_dir = args.model_name.split('-')[0]  # 'owlvit' or 'owlv2'
    file_prefix = f"{args.file_tag}_" if args.file_tag else ""
    
    onnx_filename = f"{file_prefix}{args.component}.onnx"
    onnx_path = os.path.join("weights/onnx", model_type_dir, onnx_filename)

    engine_filename = f"{file_prefix}{args.component}.engine"
    output_dir = os.path.join("weights/tensorrt", model_type_dir)
    engine_path = os.path.join(output_dir, engine_filename)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Required ONNX file not found at: {onnx_path}\nPlease run export_onnx.py first.")

    # 2. trtexec 명령어 조립
    command = ["/usr/src/tensorrt/bin/trtexec"]
    command.append(f"--onnx={onnx_path}")
    command.append(f"--saveEngine={engine_path}")

    # 정밀도(Precision) 옵션 설정
    if args.precision == 'fp16':
        command.append("--fp16")
    elif args.precision == 'int8':
        command.append("--int8")
        if args.int8_cache_path:
            command.append(f"--calibrationCache={args.int8_cache_path}")
        else:
            # INT8 모드에서는 Calibration Cache가 필수적이므로 에러 처리
            raise ValueError("--precision INT8 requires --int8_cache_path to be set.")
    
    # 컴포넌트에 따른 Shape 정보 설정
    props = MODEL_PROPERTIES[full_model_name]
    if args.component == 'image_encoder':
        image_size = props['image_size']
        command.append(f"--shapes=image:1x3x{image_size}x{image_size}")
    elif args.component == 'text_encoder':
        max_len = args.max_text_length
        command.append(f"--shapes=input_ids:1x{max_len},attention_mask:1x{max_len}")
    elif args.component == 'prediction_head':
        num_patches = (props['image_size'] // props['patch_size']) ** 2
        max_len = args.max_text_length
        embed_dim = 512
        command.append(f"--shapes=image_class_embeds:1x{num_patches}x{embed_dim},text_embeds:1x{max_len}x{embed_dim},logit_shift:1x{num_patches}x1,logit_scale:1x{num_patches}x1")

    # (확장 가능) 추가 최적화 옵션
    if args.builder_optimization_level is not None:
        command.append(f"--builderOptimizationLevel={args.builder_optimization_level}")
    if args.min_timing is not None:
        command.append(f"--minTiming={args.min_timing}")
    if args.avg_timing is not None:
        command.append(f"--avgTiming={args.avg_timing}")

    # 3. 명령어 실행
    print("─" * 80)
    print(f"Building {args.component} for {args.model_name} with tag '{args.file_tag}'...")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output Engine: {engine_path}")
    print("Running command:")
    # 보기 쉽게 여러 줄로 출력
    print("  " + " \\\n    ".join(command))
    print("─" * 80)

    try:
        subprocess.run(command, check=True)
        print(f"\n✅ Successfully built engine: {engine_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to build engine. trtexec returned non-zero exit code: {e.returncode}")
    except FileNotFoundError:
        print("\n❌ Error: 'trtexec' not found. Please ensure TensorRT is installed correctly and /usr/src/tensorrt/bin is in your PATH or accessible.")


if __name__ == "__main__":
    short_model_names = [name.replace("google/", "") for name in MODEL_PROPERTIES.keys()]
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX files using a structured workflow.")
    
    # --- 필수 인자 ---
    parser.add_argument("--model_name", type=str, required=True, choices=short_model_names, help="Model name, used to determine input/output paths and properties.")
    parser.add_argument("--component", type=str, required=True, choices=['image_encoder', 'text_encoder', 'prediction_head'], help="The model component to build.")
    
    # --- 선택적 인자 ---
    parser.add_argument("--file_tag", type=str, help="Optional tag used in the ONNX filename (e.g., '32', '16', 'ensemble').")
    parser.add_argument("--precision", type=str, default='fp16', choices=['fp16', 'int8'], help="Precision for the engine build.")
    parser.add_argument("--int8_cache_path", type=str, help="Path to the INT8 calibration cache file (required if --precision is INT8).")
    parser.add_argument("--max_text_length", type=int, default=16, help="Max sequence length for text/prediction components.")
    
    # --- 향후 확장을 위한 최적화 인자 ---
    parser.add_argument("--builder_optimization_level", type=int, default=5, help="Set TensorRT builder optimization level (0-5).")
    parser.add_argument("--min_timing", type=int, default=8, help="Set the minimum number of timing iterations for trtexec.")
    parser.add_argument("--avg_timing", type=int, default=16, help="Set the number of averaging timing iterations for trtexec.")
    
    # --- 인자 파싱 및 엔진 빌드 ---
    args = parser.parse_args()
    
    build_engine(args)
```

- Execute Commands:
```bash
# OWL-ViT base patch32
python export_engine.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32 \
    --component image_encoder \
    --force_fp32_layers "*LayerNorm*"

python export_engine.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32 \
    --component text_encoder \
    --force_fp32_layers "*LayerNorm*"

python export_engine.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32 \
    --component prediction_head

# OWL-ViT base patch16
python export_engine.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16 \
    --component image_encoder \
    --force_fp32_layers "*LayerNorm*"

python export_engine.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16 \
    --component text_encoder

python export_engine.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16 \
    --component prediction_head

# OWLv2 base patch16
python export_engine.py \
    --model_name owlv2-base-patch16 \
    --component image_encoder \
    --force_fp32_layers "*LayerNorm*"

python export_engine.py \
    --model_name owlv2-base-patch16 \
    --component text_encoder

python export_engine.py \
    --model_name owlv2-base-patch16 \
    --component prediction_head

# OWLv2 base patch16 ensemble
python export_engine.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble \
    --component image_encoder \
    --force_fp32_layers "*LayerNorm*"

python export_engine.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble \
    --component text_encoder

python export_engine.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble \
    --component prediction_head
```
</details>

<details>
<summary>Make a code for make a `.cache` files with calibration dataset for `int8 calibration`</summary>

- First, have to download coco128 dataset
```bash
cd ~/vlm/src/owlvit
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip coco128.zip && rm coco128.zip
```

- Second, have to download annotations for calibration dummy text queries
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```

- Make a python code with **export_calib_cache.py**:
```python
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
```

- Execute Commands:
```bash
python3 export_calib_cache.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32 \
    --component image_encoder

python3 export_calib_cache.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32 \
    --component text_encoder

python3 export_calib_cache.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16 \
    --component image_encoder

python3 export_calib_cache.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16 \
    --component text_encoder

python3 export_calib_cache.py \
    --model_name owlv2-base-patch16 \
    --component image_encoder

python3 export_calib_cache.py \
    --model_name owlv2-base-patch16 \
    --component text_encoder

python3 export_calib_cache.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble \
    --component image_encoder

python3 export_calib_cache.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble \
    --component text_encoder
```
</details>

- Test with `onnx_latency.py`
```bash
python3 onnx_latency.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32

python3 onnx_latency.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16

python3 onnx_latency.py \
    --model_name owlv2-base-patch16 \

python3 onnx_latency.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble
```

- Test with `trt_latency.py`
```bash
python3 trt_latency.py \
    --model_name owlvit-base-patch32 \
    --file_tag 32

python3 trt_latency.py \
    --model_name owlvit-base-patch16 \
    --file_tag 16

python3 trt_latency.py \
    --model_name owlv2-base-patch16 \

python3 trt_latency.py \
    --model_name owlv2-base-patch16-ensemble \
    --file_tag ensemble
```





