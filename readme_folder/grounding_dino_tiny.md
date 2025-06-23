# Grounding-Dino-Tiny on Jetson

### gd_fix.onnx (image size fixed with 960x544)
- First make folder
```bash
cd ~/Downloads
mkdir -p fp16 && mkdir -p int8
```

- Download `.onnx` file
```bash
wget -O gd_fix.onnx 'https://xfiles.ngc.nvidia.com/org/nvidia/team/tao/models/grounding_dino/versions/grounding_dino_swin_tiny_commercial_deployable_v1.0/files/grounding_dino_swin_tiny_commercial_deployable.onnx?ssec-algo=AES256&versionId=3KWAS8ddBCoX3giKV31T5FrAfpKVI8kW&ssec-key=RRNg%2FDG7gN4dm50QMzLnT7K27B9KPlu4e%2FPxHpoq1N52R6i9lkwPl%2FvoHMoCekLswRu2R8X9TO8TUh%2FZqd2TU6ZbqEU9T4E9DOUbpHjuUblsWko0T%2FK09HnIPXxNaaNGcsa2u1OB8gcNl0hbXHa4QX0LdVX15g7qLUpHjJuQqP0e1AKkOr5kW0xutePbYTbWOJokPzvtUD5yzCRih%2BZ2QIKDDpe75NLuqSTf3l9erd6fv%2B0JLQ%2BLApKLb0Gf%2BpCEJWzn4lDvlvIMBdhk0e0b9Hay54AG2zmMyHB951fUVosy8HjLUKF3OgY8i%2BCyrjtDxs1jFlkuhIVmm5j%2FbS68o6nXqWLkNz1E7XfYx27824GOmPWK%2FX%2F9a4LsZaJ5PB3PU%2FOqDzRZZUICzEe5iIqr1yVHB3nvTMSOkTvKVXn2RvTAHSHz8GKSorJ9ziq2plBQrDVqiuCTlBM2rGhlzd%2FC8JFJhPln56I8Uue6WnIqCNj9SQWOgf8xPa2dn0U9Zli3&Signature=xV9NoGCCA~fGH0eL2QmRBQRIEZJ6wNwQJgJLJnOcL24BYw62NPFA8aew6WvWtN8DewWPouj848n2igESztu6OhoRzBRe3-a07Oc8V~~srWp8L5SkwKXM2GmE70dTqEP7ky6ZpBRij05mqTRD58zKwtxXft9E1C2mZBBmfGIszEMYXcUduUUO~XUiuy8JqXtXCTD-wgTRS45DvFT60fNQJauAn3Vc5~PxAYvCar4C~1DSRqrOVtk98aJ8RKU4Wze1OgBMw4Sg2UM4G6YrfyeT2WGKhdiqWSWxyk5BkxLB-7Pu~1Y3aGymub1a4Bt~LaE5ltgYQGAL8JpwuUo2mGr~rw__&kid=bXJrLWU3OGM1M2FhZjE4YzRiNmJiNjlkYmRhZjcxNjA3YWEw&Expires=1750502818&Key-Pair-Id=KCX06E8E9L60W&ssec-enabled=true'
```

- Build `.onnx` to `.engine` with `trtexec`
```bash
trtexec \
    --onnx=gd_fix.onnx \
    --fp16 \
    --saveEngine=fp16/gd_fix.engine \
    --minShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
    --optShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
    --maxShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256

trtexec \
    --onnx=gd_fix.onnx \
    --int8 \
    --saveEngine=int8/gd_fix.engine \
    --minShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
    --optShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256 \
    --maxShapes=inputs:1x3x544x960,input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256
```

### gd_dynamic.onnx (image size could change)
- Download `.pth` file
```bash
wget -O gd_dynamic.pth 'https://xfiles.ngc.nvidia.com/org/nvidia/team/tao/models/grounding_dino/versions/grounding_dino_swin_tiny_commercial_trainable_v1.0/files/grounding_dino_swin_tiny_commercial_trainable.pth?ssec-algo=AES256&versionId=X.3uhyEVPpl8Oc.ArXwvJVtezTNwFXyM&ssec-key=jf%2BqRl3E4kiQ2olHFjUoiCctrSY5cRTuGDnG2doiwl%2FcxHmUhGziselh%2FmSIvjGAtSszLwTmkbYQD9wtPD8ORWFgFh0syyNLbYPmKYGksF%2FS5yveFdScNjgPB8WvoY97W46qlw5mYS3enVqfp%2FUBip9vE%2B1zoGq%2BzZzI3NUslKA5fM1vSEdUoPYYXb4sN7sby3hDcCgHqIKnUnqptJa2Hu9w%2F3nC06k%2Ft4shk%2BVODrav%2F10pbhYty55P9aNk0Q5qTV7vdpYm2gzFtb9zdLXE8Z0E%2FXOXdBox2keReBiddvKzlmbrpvnJrItDMQU8Q6AteSUlj4IL1T0I8IRXvpXIuSly5DILBx68sjKFkWK%2F%2FQna9R9BF9iPxzJXoy8bh5fGu8nO95S5IKxsyeO35d0nsrnIaO%2FubHN85%2BzxOZYSYdYNaT3GWWv6fokbFSANSF5CvWfFs5xXz19ErSgPtVqlHn3wufe7yc7PemJFt%2FfaewXV3IFIHqoLgqn5esqLG8Re&Signature=FOt3Uy7HWd2CjiJSj426edf7OOM~39q7cg0Oxd5qLa1h3FV7f2DCdQsyq2Hh7slyUwbze43K4YbHcTtmwOZZOHTQ-rC4AF4PPGjdcclhbhO1p3HLwHecEXMgzo~ioItxFdfsuSt74fdpkGlXaJQzAk81IhvtyLjve4GNMusQxBBOCLttpO~WUq4MHGJVRuQ-Ir5WciWvcMK07jmYWZDNhlWGypEejEoM5gTODSD2c2pbJQxKZJnZslGIgQaB-43elpltpGKeQaMJwKKS9zPSf9tAId3KYjLmoM1sqiKJmt34aE-AXEL7-RvWQRTgMcktqXC6TMJ1l96-Ucw5qjMDdg__&kid=bXJrLWU3OGM1M2FhZjE4YzRiNmJiNjlkYmRhZjcxNjA3YWEw&Expires=1750502809&Key-Pair-Id=KCX06E8E9L60W&ssec-enabled=true'
```

- After Download, have to change .pth file with following code:
```python
import torch
from transformers import GroundingDinoForObjectDetection
import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse

def inspect_and_remap_weights(model, weights_path, debug=False):
    """
    Loads weights from a .pth file, inspects for key mismatches,
    and remaps them to the target model's state_dict.
    """
    print(f"--- Step 2.1: Loading weights from: {weights_path} ---")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found at {weights_path}")
        
    source_state_dict = torch.load(weights_path, map_location="cpu")
    # Handle checkpoints saved with a 'model' key
    if 'model' in source_state_dict:
        source_state_dict = source_state_dict['model']

    target_state_dict = model.state_dict()
    new_state_dict = target_state_dict.copy()

    # --- [개선 사항] 가중치 키 디버깅 기능 ---
    if debug:
        print("\n[DEBUG] Performing state_dict key inspection...")
        source_keys = set(source_state_dict.keys())
        target_keys = set(target_state_dict.keys())
        
        print(f"\nSource keys NOT in target model: {sorted(list(source_keys - target_keys))}")
        print(f"Target keys NOT in source pth: {sorted(list(target_keys - source_keys))}\n")
    # ------------------------------------

    print("--- Step 2.2: Remapping keys for compatibility... ---")
    remapped_keys = 0
    matched_keys = 0
    ignored_keys = 0

    for key, value in source_state_dict.items():
        # --- 핵심 리매핑 로직: Fused Multi-Head Attention 가중치 분리 ---
        if "in_proj_weight" in key:
            q_w, k_w, v_w = torch.chunk(value, 3, dim=0)
            new_state_dict[key.replace("in_proj_weight", "query.weight")] = q_w
            new_state_dict[key.replace("in_proj_weight", "key.weight")] = k_w
            new_state_dict[key.replace("in_proj_weight", "value.weight")] = v_w
            remapped_keys += 3
        elif "in_proj_bias" in key:
            q_b, k_b, v_b = torch.chunk(value, 3, dim=0)
            new_state_dict[key.replace("in_proj_bias", "query.bias")] = q_b
            new_state_dict[key.replace("in_proj_bias", "key.bias")] = k_b
            new_state_dict[key.replace("in_proj_bias", "value.bias")] = v_b
            remapped_keys += 3
        # -----------------------------------------------------------------
        elif key in new_state_dict:
            # 키가 정확히 일치하는 경우
            new_state_dict[key] = value
            matched_keys += 1
        else:
            ignored_keys += 1

    print(f"Remapping summary: {matched_keys} keys matched, {remapped_keys} keys remapped, {ignored_keys} keys ignored.")
    
    # strict=False를 사용하여 리매핑되지 않은 키(e.g., in_proj_weight)는 무시
    model.load_state_dict(new_state_dict, strict=False)
    print("[SUCCESS] Weights remapped and loaded into the model.")

def polish_onnx_for_trt(input_path, output_path):
    """
    Applies graph surgery to the ONNX model to ensure TensorRT compatibility.
    Specifically, it replaces EyeLike operators and casts int64 constants to int32.
    """
    print(f"\n--- Step 4: Polishing ONNX model for TensorRT: {input_path} ---")
    graph = gs.import_onnx(onnx.load(input_path))

    # Cast int64 constants to int32 for better TRT compatibility
    for tensor in graph.tensors().values():
        if isinstance(tensor, gs.Constant) and tensor.values.dtype == np.int64:
            tensor.values = tensor.values.astype(np.int32)

    # Replace EyeLike op with a sequence of basic ops
    for node in [n for n in graph.nodes if n.op == "EyeLike"]:
        # ... (이전과 동일한 EyeLike 대체 로직) ...
        shape_defining_tensor = node.inputs[0]
        final_output_tensor = node.outputs[0]
        output_dtype_proto = node.attrs.get("dtype", onnx.TensorProto.FLOAT)
        shape_out = gs.Variable(name=f"{node.name}_shape_out", dtype=np.int64)
        shape_node = gs.Node(op="Shape", inputs=[shape_defining_tensor], outputs=[shape_out])
        gather_indices = gs.Constant(name=f"{node.name}_gather_indices", values=np.array(0, dtype=np.int64))
        dim_out = gs.Variable(name=f"{node.name}_dim_out", dtype=np.int64)
        gather_node = gs.Node(op="Gather", attrs={"axis": 0}, inputs=[shape_out, gather_indices], outputs=[dim_out])
        range_start = gs.Constant(name=f"{node.name}_range_start", values=np.array(0, dtype=np.int64))
        range_step = gs.Constant(name=f"{node.name}_range_step", values=np.array(1, dtype=np.int64))
        range_out = gs.Variable(name=f"{node.name}_range_out", dtype=np.int64)
        range_node = gs.Node(op="Range", inputs=[range_start, dim_out, range_step], outputs=[range_out])
        axes_for_rows = gs.Constant(name=f"{node.name}_axes_rows", values=np.array([1], dtype=np.int64))
        axes_for_cols = gs.Constant(name=f"{node.name}_axes_cols", values=np.array([0], dtype=np.int64))
        rows_unsq_out = gs.Variable(name=f"{node.name}_rows_unsq")
        cols_unsq_out = gs.Variable(name=f"{node.name}_cols_unsq")
        rows_unsq_node = gs.Node(op="Unsqueeze", inputs=[range_out, axes_for_rows], outputs=[rows_unsq_out])
        cols_unsq_node = gs.Node(op="Unsqueeze", inputs=[range_out, axes_for_cols], outputs=[cols_unsq_out])
        equal_out = gs.Variable(name=f"{node.name}_equal_out", dtype=bool)
        equal_node = gs.Node(op="Equal", inputs=[rows_unsq_out, cols_unsq_out], outputs=[equal_out])
        cast_node = gs.Node(op="Cast", attrs={"to": output_dtype_proto}, inputs=[equal_out], outputs=[final_output_tensor])
        graph.nodes.extend([shape_node, gather_node, range_node, rows_unsq_node, cols_unsq_node, equal_node, cast_node])
        node.outputs.clear()

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    print(f"[SUCCESS] Polished ONNX saved to: {output_path}")

def main(args):
    temp_onnx_path = "temp_exported.onnx"
    
    print(f"--- Step 1: Loading model architecture for '{args.model_id}' ---")
    model = GroundingDinoForObjectDetection.from_pretrained(args.model_id)
    
    # --- Step 2: Load and remap weights ---
    inspect_and_remap_weights(model, args.pth_file, args.debug_keys)
    model.eval()

    print(f"\n--- Step 3: Exporting to Dynamic ONNX file: {temp_onnx_path} ---")
    dummy_inputs = (
        torch.randn(1, 3, args.img_size, args.img_size), 
        torch.randint(0, 100, (1, args.seq_len), dtype=torch.int32), 
        torch.ones(1, args.seq_len, dtype=torch.int32)
    )
    input_names = ["pixel_values", "input_ids", "attention_mask"]
    output_names = ["logits", "pred_boxes"]
    dynamic_axes = {
        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"}
    }

    torch.onnx.export(
        model, dummy_inputs, temp_onnx_path,
        input_names=input_names, output_names=output_names,
        opset_version=17, export_params=True, do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    print(f"[SUCCESS] Temporary ONNX file exported.")
    
    # --- Step 4: Polish ONNX for TensorRT ---
    polish_onnx_for_trt(temp_onnx_path, args.output_onnx)
    
    os.remove(temp_onnx_path)
    print("\n[WORKFLOW COMPLETE] Your TensorRT-ready dynamic ONNX file is ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Grounding DINO .pth file to a dynamic, TensorRT-ready ONNX file.")
    parser.add_argument("--pth_file", type=str, required=True, help="Path to the input .pth weight file from NGC.")
    parser.add_argument("--output_onnx", type=str, default="gd_dynamic.onnx", help="Path to save the final ONNX file.")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny", help="Hugging Face model ID to load the architecture.")
    parser.add_argument("--img_size", type=int, default=800, help="Image size (height and width) for dummy input.")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for dummy text input.")
    # --- [개선 사항] 디버그 플래그 추가 ---
    parser.add_argument("--debug_keys", action='store_true', help="Inspect and print mismatching keys between source .pth and target model.")
    
    args = parser.parse_args()
    main(args)
```

```bash
trtexec \
    --onnx=gd_dynamic.onnx \
    --saveEngine=fp16/gd_dynamic.engine \
    --fp16 \
    --minShapes=pixel_values:1x3x512x512,input_ids:1x8,attention_mask:1x8 \
    --optShapes=pixel_values:1x3x640x640,input_ids:1x32,attention_mask:1x32 \
    --maxShapes=pixel_values:1x3x1024x1024,input_ids:1x32,attention_mask:1x32

trtexec \
    --onnx=gd_dynamic.onnx \
    --saveEngine=int8/gd_dynamic.engine \
    --int8 \
    --minShapes=pixel_values:1x3x512x512,input_ids:1x8,attention_mask:1x8 \
    --optShapes=pixel_values:1x3x640x640,input_ids:1x32,attention_mask:1x32 \
    --maxShapes=pixel_values:1x3x1024x1024,input_ids:1x32,attention_mask:1x32
```