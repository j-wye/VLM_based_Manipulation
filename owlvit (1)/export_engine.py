import os
import argparse
import subprocess

# calib.py와 export_onnx.py에서 사용한 모델 속성 정보를 그대로 가져와 일관성을 유지합니다.
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
    command = ["trtexec"]
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