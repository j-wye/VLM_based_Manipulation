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
from typing import List, Optional, Tuple

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
    output_dir = os.path.join("weights/onnx", model_type_dir)
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