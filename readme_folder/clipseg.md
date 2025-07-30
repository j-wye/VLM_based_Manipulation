# CLIPSeg

- Make a folders and structures:
```bash
cd ~/vlm/src
mkdir -p clipseg/weights
cd clipseg
```

- I will use following models:
    - `clipseg-rd64`
    - `clipseg-rd64-refined`

<details>
<summary>Make a code for make a `.onnx` files</summary>

- Make a python code with **export_onnx.py**:
```python
import torch
import os
import argparse
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

class ClipSegWrapper(torch.nn.Module):
    def __init__(self, model: CLIPSegForImageSegmentation):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

# ONNX Exporter 클래스
class ClipSegOnnxExporter:
    def __init__(self, hf_id: str, device: str = "cuda"):
        super().__init__()
        self.hf_id = hf_id
        self.device = device

        print(f"Hugging Face에서 '{self.hf_id}' 모델을 로드합니다...")
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.hf_id).to(device).eval()
        self.processor = CLIPSegProcessor.from_pretrained(self.hf_id)
        print("모델 로드 완료.")

    @torch.no_grad()
    def export_model_onnx(self, path: str, max_text_len: int, opset: int):
        wrapper_model = ClipSegWrapper(self.model).to(self.device).eval()

        image_size = self.model.config.vision_config.image_size
        dummy_pixel_values = torch.randn(1, 3, image_size, image_size, device=self.device)
        dummy_input_ids = torch.ones(1, max_text_len, dtype=torch.long, device=self.device)
        dummy_attention_mask = torch.ones(1, max_text_len, dtype=torch.long, device=self.device)
        
        dummy_inputs = (dummy_pixel_values, dummy_input_ids, dummy_attention_mask)

        input_names = ["pixel_values", "input_ids", "attention_mask"]
        output_names = ["logits"]

        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }

        print(f"ONNX 변환을 시작합니다 (opset={opset})...")
        torch.onnx.export(
            wrapper_model,
            dummy_inputs,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            verbose=False
        )
        print(f"ONNX 모델이 성공적으로 '{path}'에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIPSeg 모델을 ONNX로 변환합니다.")
    
    parser.add_argument("--file_tag", type=str, choices=['refined'], 
                        help="'refined'로 설정 시 refined 모델을 사용합니다. 설정하지 않으면 기본 모델을 사용합니다.")
    parser.add_argument("--onnx_opset", type=int, default=17, help="ONNX opset 버전.")
    parser.add_argument("--max_text_length", type=int, default=16, help="텍스트 인코더의 최대 시퀀스 길이.")
    
    args = parser.parse_args()

    if args.file_tag == 'refined':
        model_hf_id = "CIDAS/clipseg-rd64-refined"
        output_filename = "clipseg_refined.onnx"
        print("선택된 모델: Refined (clipseg-rd64-refined)")
    else:
        model_hf_id = "CIDAS/clipseg-rd64"
        output_filename = "clipseg.onnx"
        print("선택된 모델: Base (clipseg-rd64)")

    output_dir = os.path.join("weights", "onnx")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Exporter 초기화 및 실행
    exporter = ClipSegOnnxExporter(hf_id=model_hf_id)
    exporter.export_model_onnx(output_path, args.max_text_length, args.onnx_opset)
```

- Execute Commands:
```bash
python export_onnx.py \
    --max_text_length 16

python export_onnx.py \
    --file_tag refined \
    --max_text_length 16
```

</details>

<details>
<summary> Make a code for make a `.engine` files</summary>

- Make a python code with **export_engine.py**:
```python
import os
import argparse
import subprocess

IMAGE_SIZE = 224

def build_engine(args):
    # 1. --file_tag 인자에 따라 입력/출력 파일명 결정
    if args.file_tag == 'refined':
        onnx_filename = "clipseg_refined.onnx"
        engine_filename = "clipseg_refined.engine"
        print("타겟 모델: Refined (입력: clipseg_refined.onnx)")
    else:
        onnx_filename = "clipseg.onnx"
        engine_filename = "clipseg.engine"
        print("타겟 모델: Base (입력: clipseg.onnx)")

    # 2. 단순화된 경로 구성
    onnx_path = os.path.join("weights", "onnx", onnx_filename)
    output_dir = os.path.join("weights", "tensorrt")
    engine_path = os.path.join(output_dir, engine_filename)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"입력 ONNX 파일이 없습니다: {onnx_path}\n"
                              f"먼저 `export_onnx.py --file_tag {args.file_tag if args.file_tag else ''}`를 실행하세요.")

    # 3. trtexec 명령어 조립
    command = ["trtexec"]
    command.append(f"--onnx={onnx_path}")
    command.append(f"--saveEngine={engine_path}")

    # 정밀도(Precision) 옵션 설정
    if args.precision == 'fp16':
        command.append("--fp16")
    elif args.precision == 'int8':
        command.append("--int8")
        cache_path = onnx_path.replace(".onnx", ".cache")
        print(f"INT8 모드: 캘리브레이션 캐시 파일을 찾습니다... 경로: {cache_path}")
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"INT8 캘리브레이션 캐시 파일이 없습니다: {cache_path}\n"
                f"먼저 `export_calib_cache.py`를 실행하여 캐시 파일을 생성해야 합니다."
            )
        
        command.append(f"--calib={cache_path}")
        print("캐시 파일을 찾았으며, --calib 인자에 추가합니다.")

    max_len = args.max_text_length
    command.append(f"--shapes=pixel_values:1x3x{IMAGE_SIZE}x{IMAGE_SIZE},input_ids:1x{max_len},attention_mask:1x{max_len}")
    
    # 추가 최적화 옵션
    if args.builder_optimization_level is not None:
        command.append(f"--builderOptimizationLevel={args.builder_optimization_level}")
    if args.min_timing is not None:
        command.append(f"--minTiming={args.min_timing}")
    if args.avg_timing is not None:
        command.append(f"--avgTiming={args.avg_timing}")

    # 4. 명령어 실행
    print("─" * 80)
    print(f"TensorRT 엔진 빌드를 시작합니다...")
    print(f"  - 입력 ONNX: {onnx_path}")
    print(f"  - 출력 Engine: {engine_path}")
    print(f"  - 정밀도: {args.precision.upper()}")
    print("실행될 trtexec 명령어:")
    print("  " + " \\\n    ".join(command))
    print("─" * 80)

    try:
        subprocess.run(command, check=True, text=True, capture_output=False)
        print(f"\n엔진 빌드 성공: {engine_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n엔진 빌드 실패. trtexec이 0이 아닌 코드를 반환했습니다: {e.returncode}")
        print("오류 출력:\n", e.stderr)
    except FileNotFoundError as e:
        if 'trtexec' in str(e):
             print("\n오류: 'trtexec'를 찾을 수 없습니다. TensorRT가 올바르게 설치되었고, 관련 bin 디렉터리가 PATH에 포함되어 있는지 확인하십시오.")
        else:
            print(f"\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIPSeg ONNX 파일로부터 TensorRT 엔진을 빌드합니다.")
    parser.add_argument("--file_tag", type=str, choices=['refined'],
                        help="'refined'로 설정 시 refined 모델을 타겟으로 합니다. 설정하지 않으면 기본 모델을 타겟으로 합니다.")
    parser.add_argument("--precision", type=str, default='fp16', choices=['fp32', 'fp16', 'int8'], 
                        help="엔진 빌드 정밀도. 'int8' 선택 시 캘리브레이션 캐시 파일이 필요합니다.")
    parser.add_argument("--max_text_length", type=int, default=16, 
                        help="입력 Shape 지정을 위한 최대 텍스트 시퀀스 길이.")
    parser.add_argument("--builder_optimization_level", type=int, default=5, help="TensorRT 빌더 최적화 레벨 (0-5).")
    parser.add_argument("--min_timing", type=int, default=8, help="trtexec의 최소 타이밍 반복 횟수.")
    parser.add_argument("--avg_timing", type=int, default=16, help="trtexec의 평균 타이밍 반복 횟수.")
    args = parser.parse_args()
    build_engine(args)
```

- Execute Commands:
```bash
python export_engine.py \
    --max_text_length 16

python export_engine.py \
    --file_tag refined \
    --max_text_length 16
```
</details>

<details>
<summary>Make a code for make a `.cache` files with calibration dataset for `int8 calibration`</summary>

- First, have to download coco128 dataset
```bash
cd ~/vlm/src/clipseg
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
import argparse
import json
from PIL import Image
from tqdm import tqdm
from transformers import CLIPSegProcessor

IMAGE_SIZE = 352

class ClipSegCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, onnx_path: str, batch_size: int, image_dir: str, captions_json_path: str, max_samples: int = 1024):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.cache_file = onnx_path.replace(".onnx", ".cache")
        self.index = 0
        
        model_name = "CIDAS/clipseg-rd64-refined" if "refined" in onnx_path else "CIDAS/clipseg-rd64"
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.image_size = IMAGE_SIZE
        self.max_text_len = 16

        print("효율적인 방식으로 COCO Paired Data를 구성합니다...")
        self.data = self._prepare_paired_data_efficiently(image_dir, captions_json_path)
        if len(self.data) > max_samples:
            self.data = sorted(self.data, key=lambda x: x['image_path'])
            self.data = self.data[:max_samples]
        print(f"총 {len(self.data)}개의 이미지-캡션 쌍을 캘리브레이션에 사용합니다.")
        
        self.device_inputs = []
        input_shapes = [(self.batch_size, 3, self.image_size, self.image_size), (self.batch_size, self.max_text_len), (self.batch_size, self.max_text_len)]
        input_types = [np.float32, np.int64, np.int64]
        for shape, dtype in zip(input_shapes, input_types):
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            self.device_inputs.append(cuda.mem_alloc(size))
    
    def _prepare_paired_data_efficiently(self, image_dir, captions_path):
        available_image_ids = set(int(os.path.splitext(f)[0]) for f in os.listdir(image_dir) if f.endswith('.jpg'))
        with open(captions_path, 'r') as f: captions_data = json.load(f)
        caption_map = {}
        for ann in captions_data['annotations']:
            if ann['image_id'] in available_image_ids:
                if ann['image_id'] not in caption_map: caption_map[ann['image_id']] = []
                caption_map[ann['image_id']].append(ann['caption'])
        paired_data = []
        for img_id, captions in caption_map.items():
            image_path = os.path.join(image_dir, f"{img_id:012d}.jpg")
            if os.path.exists(image_path):
                for caption in captions: paired_data.append({'image_path': image_path, 'caption': caption})
        return paired_data

    def get_batch_size(self): return self.batch_size
    def get_batch(self, names):
        if self.index >= len(self.data): return None
        end_idx = min(self.index + self.batch_size, len(self.data))
        current_batch_size = end_idx - self.index
        pixel_values_host = np.zeros((self.batch_size, 3, self.image_size, self.image_size), dtype=np.float32)
        input_ids_host = np.zeros((self.batch_size, self.max_text_len), dtype=np.int64)
        attention_mask_host = np.zeros((self.batch_size, self.max_text_len), dtype=np.int64)
        for i in range(current_batch_size):
            item = self.data[self.index + i]
            image = Image.open(item['image_path']).convert("RGB")
            inputs = self.processor(text=[item['caption']], images=[image], return_tensors="pt", padding="max_length", max_length=self.max_text_len, truncation=True)
            pixel_values_host[i], input_ids_host[i], attention_mask_host[i] = inputs['pixel_values'].numpy(), inputs['input_ids'].numpy(), inputs['attention_mask'].numpy()
        cuda.memcpy_htod(self.device_inputs[0], pixel_values_host.ravel())
        cuda.memcpy_htod(self.device_inputs[1], input_ids_host.ravel())
        cuda.memcpy_htod(self.device_inputs[2], attention_mask_host.ravel())
        self.index += self.batch_size
        return [int(d) for d in self.device_inputs]
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f: return f.read()
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f: f.write(cache)
        print(f"✅ 캘리브레이션 캐시가 성공적으로 '{self.cache_file}'에 저장되었습니다.")
    def free(self):
        for d_input in self.device_inputs: d_input.free()
        print("CUDA 메모리를 해제했습니다.")

def generate_calibration_cache(args):
    onnx_filename = "clipseg_refined.onnx" if args.file_tag == 'refined' else "clipseg.onnx"
    onnx_path = os.path.join("weights", "onnx", onnx_filename)
    if not os.path.exists(onnx_path): raise FileNotFoundError(f"ONNX 파일이 없습니다: {onnx_path}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            raise ValueError(f"ONNX 파일 파싱 실패: {onnx_path}")

    profile = builder.create_optimization_profile()
    img_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
    txt_shape = (args.max_text_length,)
    
    profile.set_shape("pixel_values", min=(1, *img_shape), opt=(args.batch_size, *img_shape), max=(args.batch_size, *img_shape))
    profile.set_shape("input_ids", min=(1, *txt_shape), opt=(args.batch_size, *txt_shape), max=(args.batch_size, *txt_shape))
    profile.set_shape("attention_mask", min=(1, *txt_shape), opt=(args.batch_size, *txt_shape), max=(args.batch_size, *txt_shape))
    config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = ClipSegCalibrator(onnx_path=onnx_path, batch_size=args.batch_size, image_dir=args.image_dir, captions_json_path=args.captions_path)
    config.int8_calibrator = calibrator

    print("\nINT8 캘리브레이션을 시작합니다 (캐시 파일 생성 목적)...")
    builder.build_serialized_network(network, config)
    print("\n캘리브레이션 완료.")
    
    calibrator.free()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIPSeg 모델을 위한 INT8 캘리브레이션 캐시를 생성합니다.")
    parser.add_argument("--file_tag", type=str, choices=['refined'], help="'refined' 설정 시 refined 모델을 타겟으로 합니다.")
    parser.add_argument("--batch_size", type=int, default=8, help="캘리브레이션에 사용할 배치 크기.")
    parser.add_argument("--image_dir", type=str, default="coco128/images/train2017", help="캘리브레이션용 이미지 파일이 있는 디렉터리.")
    parser.add_argument("--captions_path", type=str, default="annotations/captions_train2017.json", help="COCO 캡션 JSON 파일 경로.")
    parser.add_argument("--max_text_length", type=int, default=16, help="텍스트 입력의 최대 시퀀스 길이.")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir) or not os.path.exists(args.captions_path):
        print("필요한 데이터셋 파일이 없습니다.")
    else:
        try:
            generate_calibration_cache(args)
        except Exception as e:
            import traceback
            print(f"\n오류 발생: {e}")
            traceback.print_exc()

```

- Execute Commands:
```bash
python3 export_calib_cache.py

python3 export_calib_cache.py \
    --file_tag refined \

```
</details>

- Test with `onnx_latency.py`
```bash
python3 onnx_latency.py

python3 onnx_latency.py \
    --file_tag refined
```

- Test with `trt_latency.py`
```bash
python3 trt_latency.py

python3 trt_latency.py \
    --file_tag refined
```








