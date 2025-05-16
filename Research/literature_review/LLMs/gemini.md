# **실시간 인식-추적-분할 파이프라인 최적화 및 지능형 정보 융합 전략 기반 로봇 그리핑 연구: NanoOwl, NanoSAM 및 Jetson Orin AGX 활용**

## **1\. 서론**

### **1.1. 연구 배경 및 목표 상세 기술**

본 보고서는 인간과 로봇 간의 직관적인 상호작용을 목표로, NVIDIA Jetson Orin AGX 64GB와 같은 엣지 디바이스 환경에서 자연어 명령을 입력받아 로봇 매니퓰레이터가 특정 객체를 정교하게 파지하는 시스템 개발 연구를 지원하는 데 목적을 둔다. 핵심 기술 요소로는 오픈 어휘 기반 객체 탐지 모델인 NanoOwl과 분할 모델인 NanoSAM이 고려되며, 깊이 정보를 활용하는 접근 방식의 유효성이 선행 연구를 통해 확인되었다. 특히, 로봇 팔의 동적인 움직임과 연속적인 시각 입력 변화에 대응하기 위해, 초기 자연어 이해 및 객체 특정 후에는 경량화된 시각 객체 추적기(Visual Object Tracker, VOT)를 활용하여 실시간으로 목표를 추적하고, 일정 시간마다 NanoOwl을 재실행하여 추적 및 분할의 정확성을 검증하며 세그멘테이션 마스크를 정교화하는 "인식 후 추적(Detect-then-Track)" 및 "필요 기반 정교화(Refine-on-Demand)" 아키텍처 개념이 정립되었다.

이러한 기본 골격을 바탕으로, 실제 로봇 시스템의 강인함과 효율성을 극대화하기 위한 심층 연구가 요구된다. 주요 연구 질의 사항은 다음과 같다:

1. RGB-D 정보의 지능적 융합을 통한 6자유도(6-DOF) 그리핑 포즈 추정 방법론 분석  
2. 제안된 전체 시스템 아키텍처의 엣지 디바이스(Jetson Orin AGX) 환경에서의 종단간(end-to-end) 최적화 및 실증 방안  
3. 다양한 환경(비혼잡, 혼잡, 복잡)에서 미지/미등록(unknown/unlabeled) 객체에 대한 강인한 그리핑 성공률(Success Rate, SR) 극대화

### **1.2. 보고서의 목적 및 범위 명확화**

본 보고서는 상기 연구 목표 달성을 지원하기 위해, 2018년 이후 발표된 최신 오픈소스 6-DOF 그리핑 알고리즘 및 포인트 클라우드(Point Cloud) 업샘플링 기술에 대한 심층 분석을 제공하는 것을 목적으로 한다. 특히, 각 알고리즘의 입력 데이터 유형, 실시간 성능(Jetson Orin AGX 환경 고려), 다양한 환경에서의 미지 객체 파지 성공률, 특징 융합 방식, 그리고 오픈소스 현황(GitHub 링크 포함)을 중점적으로 다룬다. 또한, 포인트 클라우드 데이터의 희소성(sparsity) 문제를 해결하기 위한 업샘플링 기법의 적용 가능성과 그리핑 성능에 미칠 수 있는 영향을 검토한다. 궁극적으로 본 보고서는 사용자 시스템에 적합한 기술을 선택하고 적용하는 데 필요한 핵심 정보를 제공하고, 이론적 기반과 실용적 구현 방안을 제시하고자 한다.

### **1.3. 보고서의 구성**

본 보고서는 다음과 같이 구성된다. 2장에서는 로봇 그리핑을 위한 다양한 입력 데이터 유형(깊이 이미지, 포인트 클라우드, RGB-D)을 분석하고, 각 유형의 장단점 및 NanoOwl/NanoSAM 시스템과의 정합성을 논의한다. 3장에서는 2018년 이후 발표된 주요 오픈소스 6-DOF 그리핑 알고리즘들을 심층적으로 비교 분석하며, 입력 데이터, 특징 융합, 실시간 성능, 파지 성공률, Jetson Orin AGX 적용 가능성 등을 상세히 다룬다. 4장에서는 포인트 클라우드 업샘플링 기술의 개요와 주요 알고리즘을 소개하고, 그리핑 성능 향상에 대한 잠재력과 한계점을 논의한다. 마지막으로 5장에서는 연구 결과를 요약하고, 사용자의 연구 목표 및 특정 하드웨어 환경에 적합한 기술 전략과 향후 연구 방향을 제언한다.

## **2\. 로봇 그리핑을 위한 입력 데이터 유형 분석**

### **2.1. 개요: 그리핑을 위한 3D 정보의 중요성**

로봇이 객체를 정확하고 안정적으로 파지하기 위해서는 객체의 3차원 형상, 크기, 위치, 방향 등에 대한 정밀한 정보가 필수적이다. 2D 이미지 정보만으로는 이러한 3차원 공간 정보를 완벽하게 파악하기 어렵기 때문에, 깊이(depth) 정보를 활용하여 3D 공간을 인식하는 것이 일반적이다. 사용자의 시스템에서 NanoOwl과 NanoSAM은 주로 RGB 이미지를 기반으로 객체를 인식하고 의미론적 분할(semantic segmentation)을 수행한다. 따라서 이러한 2D 시각 정보와 3D 공간 정보를 효과적으로 결합하여 그리핑 알고리즘의 입력으로 사용하는 전략이 매우 중요하다.

### **2.2. Depth 이미지 단독 활용 전략**

Depth 이미지는 각 픽셀이 카메라로부터 해당 지점까지의 거리를 나타내는 2.5D 표현 방식이다. 이는 직접적인 3차원 좌표로 변환될 수 있으며, RGB 이미지와 유사한 격자 구조를 가져 특정 알고리즘에서 처리가 용이하다는 장점이 있다. 데이터 용량 또한 상대적으로 작아 엣지 디바이스 환경에서 전송 및 처리에 유리할 수 있다.

그러나 Depth 이미지는 시점에 따라 보이는 정보가 크게 달라지는 시점 의존성을 가지며, 객체의 뒷면이나 가려진 부분에 대한 정보를 제공하지 못한다. 또한, 센서 자체의 노이즈, 특정 재질(예: 투명하거나 반사가 심한 객체)에 대한 측정 오류 등으로 인해 부정확한 깊이 값을 가질 수 있다. ASGrasp와 같은 연구는 이러한 투명 객체 문제를 해결하기 위해 RGB와 함께 적외선(IR) 이미지를 활용하여 깊이를 추정하기도 한다.1

NanoOwl/NanoSAM 시스템과의 연동 시, NanoSAM을 통해 얻은 객체의 2D 마스크를 Depth 이미지에 투영하여 해당 객체 영역의 3D 포인트들을 선택적으로 추출하고, 이를 그리핑 후보 생성에 활용할 수 있다.

### **2.3. Point Cloud 단독 활용 전략 (XYZ)**

Point Cloud는 3D 공간상에 분포하는 점들의 집합으로, 객체의 기하학적 형상을 직접적으로 표현한다. 각 점은 X, Y, Z 좌표를 가지며, 이를 통해 객체의 실제 3차원 형상에 대한 풍부한 정보를 얻을 수 있다. 표면 법선(surface normal), 곡률(curvature) 등 다양한 기하학적 특징을 추출하여 그리핑 분석에 활용하기 용이하다는 장점이 있다.

반면, Point Cloud는 데이터의 양이 많을 수 있으며, 특히 엣지 디바이스 환경의 센서로부터 얻은 경우 밀도가 불균일하거나 희소(sparse)할 수 있다.2 이러한 희소성은 객체의 세밀한 부분을 파악하는 데 어려움을 줄 수 있으며, 그리핑 정확도에 부정적인 영향을 미칠 수 있다. 순수한 XYZ Point Cloud는 색상 정보를 포함하지 않아, 시각적 특징을 활용하는 데 한계가 있다.

NanoOwl/NanoSAM과의 연동은 NanoSAM으로 얻은 2D 마스크를 사용하여 전체 Point Cloud에서 특정 객체에 해당하는 부분만을 분리(segmentation)한 후, 이 분리된 객체 Point Cloud를 그리핑 알고리즘의 입력으로 사용하는 방식으로 이루어질 수 있다.

### **2.4. RGB-D (Point Cloud \+ RGB 이미지) 융합 전략 (XYZRGB)**

RGB-D 융합 전략은 각 3D Point Cloud 점에 해당하는 RGB 색상 정보를 결합하여 XYZRGB Point Cloud를 생성하는 방식이다. 이는 객체의 기하학적 정보와 함께 색상, 질감 등 시각적 특징을 동시에 활용할 수 있게 한다. 예를 들어, GraspNet-1Billion과 같은 대규모 그리핑 데이터셋은 RGB-D 이미지를 기반으로 구축되어, 색상과 깊이 정보를 함께 활용하는 연구를 촉진한다.4 ASGrasp는 RGB 이미지와 IR 이미지를 함께 사용하여 깊이 정보를 포함한 3D 재구성을 수행하고 이를 그리핑에 활용한다.1

이 방식의 가장 큰 장점은 객체의 재질이나 시각적 단서를 그리핑 결정에 통합할 수 있다는 점이다. 예를 들어, 자연어 명령이 "파란색 손잡이를 잡아라"와 같이 객체의 시각적 속성을 참조할 경우, XYZRGB Point Cloud는 이러한 명령을 수행하는 데 필수적인 정보를 제공한다. 이는 사용자가 잠정적으로 결론 내린 "RGB 정보를 활용하는 것이 유리하다"는 방향과도 일치한다.

그러나 XYZRGB Point Cloud는 데이터 처리의 복잡도를 증가시키며, 시각적 특징과 기하학적 특징을 효과적으로 융합하기 위한 정교한 알고리즘 설계가 요구된다. NanoOwl/NanoSAM 시스템과의 연동 시, NanoOwl의 객체 탐지 결과와 NanoSAM의 분할 마스크를 사용하여 RGB 정보와 정합된 전체 Point Cloud로부터 특정 대상 객체의 XYZRGB Point Cloud를 정확히 추출하는 과정이 중요하다.

### **2.5. 각 데이터 유형의 장단점 및 NanoOwl/NanoSAM 시스템과의 정합성 종합 비교**

각 입력 데이터 유형의 특성과 NanoOwl/NanoSAM 기반 시스템과의 정합성을 요약하면 다음과 같다.

| 특징 | Depth 이미지 (2.5D) | Point Cloud (XYZ) | Point Cloud (XYZRGB) |
| :---- | :---- | :---- | :---- |
| **장점** | 데이터량 적음, 유사 RGB 처리 가능 | 실제 3D 형상 표현, 기하학적 특징 추출 용이 | 시각+기하학 정보 통합, 풍부한 특징 활용, 자연어 연계 용이 |
| **단점** | 시점 의존성, 정보 부족(뒷면, 가려짐), 센서 노이즈/재질 취약 | 데이터량 많을 수 있음, 밀도 불균일/희소 가능, 색상 정보 부재 | 데이터 처리 복잡도 증가, 정교한 특징 융합 필요 |
| **NanoOwl/SAM 연동** | 마스크 투영으로 3D 영역 획득 | 마스크로 객체 포인트 분리 | 마스크로 객체 XYZRGB 포인트 분리, 시각 정보 직접 활용 |
| **그리핑 정확도 기여** | 제한적 | 기하학적 정보 기반 정확도 | 시각+기하학 정보 기반 고정확도 잠재력 높음 |
| **Jetson Orin 부하** | 낮음 | 중간 (밀도에 따라 변동) | 높음 (데이터 크기 및 처리 복잡도) |

사용자의 시스템은 NanoOwl과 NanoSAM을 통해 RGB 이미지로부터 풍부한 시각적, 의미론적 정보를 이미 추출하고 있다. 이러한 정보를 최대한 활용하면서 그리핑에 필요한 3D 기하 정보를 결합하는 것이 중요하다. Depth 이미지나 XYZ Point Cloud만을 사용하는 경우, NanoOwl/NanoSAM이 제공하는 시각적 컨텍스트(예: 객체의 색상, 특정 부분의 시각적 특징)를 그리핑 단계에서 직접 활용하기 어렵다. 반면, XYZRGB Point Cloud는 이러한 시각적 단서와 3D 기하 정보를 자연스럽게 통합하여, 특히 "미지/미등록 객체"에 대한 그리핑이나 자연어 명령에서 시각적 속성을 참조하는 경우(예: "빨간색 상자를 잡아라")에 더욱 강인하고 정확한 성능을 기대할 수 있다.

또한, 사용자가 구상하는 "인식 후 추적" 및 "필요 기반 정교화" 아키텍처는 초기 시각 기반 탐지(NanoOwl) 이후, 필요에 따라 상세한 3D 정보(NanoSAM으로 정교화된 마스크 기반)를 활용하여 그리핑을 수행하는 흐름을 가진다. 이 과정에서 XYZRGB 데이터는 시각적 일관성을 유지하면서 정교한 3D 기하 분석을 가능하게 하므로, 전체 파이프라인의 효율성과 정확성을 높이는 데 기여할 수 있다. NanoOwl/NanoSAM의 신뢰도 점수를 활용하여, 특정 상황(예: 혼잡한 장면에서 유사 객체 구분)에서 시각적 특징과 기하학적 특징 중 어느 쪽에 더 가중치를 둘지 동적으로 조절하는 지능형 융합 전략도 고려해볼 수 있다.

결론적으로, 사용자의 시스템 목표와 구성 요소의 특성을 고려할 때, **XYZRGB Point Cloud를 주 입력 데이터로 사용하는 것이 가장 효과적인 전략**으로 판단된다. 이는 NanoOwl/NanoSAM의 강점을 최대한 활용하고, "지능형 정보 융합"이라는 연구 목표에도 부합한다.

## **3\. 최신 6-DOF 그리핑 알고리즘 심층 분석 (2018년 이후)**

### **3.1. 선정 기준**

본 장에서는 로봇 그리핑 연구 분야에서 2018년 이후 발표된 6자유도(6-DOF) 그리핑 알고리즘들을 심층적으로 분석한다. 알고리즘 선정 기준은 다음과 같다:

* **발표 시기:** 2018년 이후 발표된 연구.  
* **발표 수준:** ICRA, IROS, RSS, CVPR, ICCV, ECCV, TRO, IJRR, RAL 등 최상위 국제 학술대회 및 저널에 게재된 논문.6  
* **오픈소스:** 실제 연구 및 개발에 활용 가능하도록 소스 코드가 공개되어 있고, GitHub 저장소 링크가 명확히 제공되는 경우. GitHub 링크가 없거나, "코드 공개 예정"이지만 현재 접근 불가능한 경우는 본 분석에서 제외한다.  
* **기능:** 6-DOF 그리핑 포즈 추정을 지원.  
* **성능:** 미지/미등록 객체(unknown/unlabeled objects) 및 다양한 환경(단순, 복잡, 혼잡)에서의 그리핑 성능이 언급된 경우.

상기 기준에 따라, 다음의 주요 알고리즘들을 중심으로 분석을 진행한다: ASGrasp, EconomicGrasp, Contact-GraspNet (NVlabs 공식 및 PyTorch 재구현), GSNet (Graspness Discovery), NeuGraspNet, Deep Object Pose Estimation (DOPE).

### **3.2. 알고리즘 분석**

각 선정된 알고리즘에 대해 핵심 아이디어, 입력 데이터, 특징 융합 방식, 실시간 성능, Jetson Orin AGX 적용 가능성, 파지 성공률, 강인성, 오픈소스 정보를 상세히 기술한다.

#### **3.2.1. ASGrasp**

* **명칭 및 발표 정보:** ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera (ICRA 2024).7  
* **핵심 아이디어 및 방법론:** 투명하거나 반사율이 높은 객체의 그리핑 문제를 해결하는 데 중점을 둔다. RGB-D 액티브 스테레오 카메라(RGB 이미지와 좌/우 적외선(IR) 이미지 쌍 사용)로부터 입력받아, 2계층 학습 기반 스테레오 네트워크를 통해 객체의 형상을 복원하고, 이를 기반으로 6-DOF 그리핑 포즈를 검출한다.1 깊이 복원 네트워크와 카메라의 깊이 맵 품질에 크게 의존하는 기존 RGB-D 기반 방식과 달리, 원시 IR 및 RGB 이미지를 직접 활용하여 투명 객체의 형상을 복원하는 것이 특징이다.8 그리핑 검출은 GSNet 아키텍처에 영향을 받은 2단계 접근 방식을 사용한다.1  
* **입력 데이터:** RGB-D (액티브 스테레오 카메라: RGB 1장, IR 이미지 쌍).1  
* **특징 융합 방식:** RGB 이미지와 IR 이미지에서 다양한 해상도의 특징을 추출하고, 미분 가능한 이중선형 샘플링(differentiable bilinear sampling)을 사용하여 IR 이미지를 RGB 참조 이미지에 정렬시켜 에피폴라 비용 볼륨(epipolar cost volume)을 생성한다. GRU(Gated Recurrent Unit) 기반의 순환 신경망 아키텍처를 사용하여 보이는 표면(1계층 깊이)과 가려진 부분(2계층 깊이)의 깊이를 동시에 예측하고 반복적으로 업데이트한다.1 복원된 포인트 클라우드(가시적 및 비가시적 깊이 정보 포함)를 그리핑 네트워크의 입력으로 사용한다.1  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** 논문이나 GitHub README에서 구체적인 FPS나 Jetson Orin AGX에서의 벤치마크 결과, TensorRT 최적화에 대한 명시적인 언급은 찾아보기 어렵다.7 프로젝트 페이지 7에서도 관련 정보는 제공되지 않았다.  
* **파지 성공률 (SR):** 실제 로봇 실험에서 미지(novel) 객체에 대해 90% 이상의 성공률을 달성했다고 보고된다.7 특히 투명 객체 그리핑에서 강점을 보인다. STD-GraspNet이라는 자체 제작 합성 데이터셋(GraspNet-1Billion 기반)에서 학습 및 평가가 이루어졌다.1  
* **강인성:** 투명 및 반사 재질 객체에 대한 깊이 추정의 어려움을 해결하여 강인성을 높였다.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/jun7-shi/ASGrasp 7  
  * 라이선스: 명시되어 있지 않으나, 일반적인 연구용 코드 공개 정책을 따를 것으로 예상된다. 코드 실행 환경으로 Python 3.8, PyTorch 1.9.1, CUDA 11.1이 언급되었다.7

#### **3.2.2. EconomicGrasp**

* **명칭 및 발표 정보:** An Economic Framework for 6-DoF Grasp Detection (ECCV 2024).10  
* **핵심 아이디어 및 방법론:** 6-DOF 그리핑 검출 시 학습 과정에서의 리소스 비용을 절감하면서도 효과적인 그리핑 성능을 유지하는 것을 목표로 한다. 기존 방법들의 병목 지점인 조밀한 지도(dense supervision) 방식의 문제를 해결하기 위해, 모호함이 적은 핵심 레이블을 선택하는 지도 선택 전략(supervision selection strategy)과 경제적인 학습 파이프라인을 포함하는 "경제적 지도 패러다임(economic supervision paradigm)"을 제안한다. 이를 통해 특정 그리핑에 집중할 수 있게 되어, "초점 표현 모듈(focal representation module)" (상호작용적 그리핑 헤드(interactive grasp head)와 복합 점수 추정(composite score estimation) 포함)을 통해 특정 그리핑을 더 정확하게 생성한다.10  
* **입력 데이터:** 포인트 클라우드. GraspNet 데이터셋(Kinect, Realsense 카메라로 수집된 RGB-D 데이터로부터 생성된 포인트 클라우드)을 사용한다.10  
* **특징 융합 방식:** 포인트 클라우드 입력(C∈Rn×3)을 기반으로 6-DOF 그리핑(G=\[c,v,a,d,w,s\])을 생성한다.11 RGB 정보와의 명시적인 융합보다는 포인트 클라우드의 기하학적 특징에 집중하는 것으로 보이며, "상호작용적 그리핑 헤드"와 "복합 점수 추정"이 특징 표현 및 선택에 관여한다.  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** 논문 초록에서 "매우 빠른 수렴(Super fast to converge)"과 약 1/4의 학습 시간 비용을 언급하며 효율성을 강조한다.10 구체적인 추론 FPS는 명시되지 않았으나, RTX 3090 GPU 1대로 학습 시간 및 메모리 비용을 테스트했다고 GitHub README에 언급되어 있다.10 Jetson Orin AGX 호환성이나 TensorRT 최적화에 대한 언급은 없다.10  
* **파지 성공률 (SR):** GraspNet-1Billion 데이터셋에서 SOTA 방법 대비 평균 3AP 높은 성능을 보인다고 주장한다.10 "Novel" 객체에 대한 AP는 Kinect 카메라에서 19.54, Realsense 카메라에서 25.48로 보고되었다.10 다양한 복잡도(uncluttered, cluttered, complex)에 따른 세부 SR은 제공되지 않았다.  
* **강인성:** 경제적 지도 패러다임을 통해 레이블의 모호성을 줄여 학습의 안정성과 모델의 강인성을 높이려 시도한다.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/iSEE-Laboratory/EconomicGrasp 10  
  * 라이선스: MIT License.10 Python 3.10, CUDA 12+ 환경을 지원한다.10

#### **3.2.3. Contact-GraspNet**

* **명칭 및 발표 정보:** Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes (ICRA 2021, Sundermeyer et al.).13  
* **핵심 아이디어 및 방법론:** 장면의 원시(raw) 깊이 기록으로부터 직접 6-DOF 병렬 조 그리퍼(parallel-jaw gripper) 그리핑 분포를 효율적으로 생성하는 종단간 네트워크이다. 제안된 그리핑 표현 방식은 기록된 포인트 클라우드의 3D 점들을 잠재적인 그리핑 접촉점(contact points)으로 취급한다. 관찰된 포인트 클라우드에 완전한 6-DOF 그리핑 포즈와 너비를 고정함으로써 그리핑 표현의 차원을 4-DOF로 줄여 학습 과정을 용이하게 한다.13 클래스에 무관한(class-agnostic) 접근 방식으로, 대규모 시뮬레이션 그리핑 데이터(ACRONYM 데이터셋 15)로 학습되며 실제 센서 데이터에 잘 일반화된다.  
* **입력 데이터:** 주로 깊이 이미지로부터 생성된 포인트 클라우드.13 RGB-D 데이터와 객체 분할 맵(segmentation map)을 함께 사용하는 것이 권장된다 (배경 그리핑 제거 및 더 조밀한 제안 생성).16  
* **특징 융합 방식:** 입력 포인트 클라우드에서 직접 그리핑 접촉점을 예측한다. RGB 정보는 주로 객체 분할을 통해 관심 영역을 특정하고, 필터링하는 데 사용될 수 있다. 핵심적인 그리핑 자세 추론은 포인트 클라우드의 기하학적 정보를 기반으로 이루어진다.  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** 원 논문에서는 구체적인 FPS를 명시하지 않았으나, "효율적(efficient)" 생성을 강조한다. PyTorch 재구현 버전의 경우, 추론 시 Nvidia GPU (VRAM \>= 8GB)를 권장한다.16 Jetson Orin AGX나 TensorRT 최적화에 대한 직접적인 언급은 공식 NVlabs 저장소나 PyTorch 재구현 저장소에서 찾기 어렵다.16  
* **파지 성공률 (SR):** 원 논문에서는 구조화된 혼잡 환경(structured clutter)에서 미지 객체에 대해 90% 이상의 성공률을 달성했다고 보고하며, 이는 당시 SOTA 방법 대비 실패율을 절반으로 줄인 결과이다.13 PyTorch 재구현 버전은 원 논문의 결과와 다를 수 있음을 명시하고 있다.16  
* **강인성:** 시뮬레이션에서 대규모 데이터로 학습하여 실제 환경에 대한 일반화 성능을 확보하려 하며, 충돌을 피하는 그리핑 생성을 목표로 한다.  
* **오픈소스 정보:**  
  * **NVlabs 공식 (TensorFlow):**  
    * GitHub: https://github.com/NVlabs/contact\_graspnet 16  
    * 라이선스: NVIDIA Source Code License (License.pdf 파일 참조).17  
  * **PyTorch 재구현 (elchun):**  
    * GitHub: https://github.com/elchun/contact\_graspnet\_pytorch 16  
    * 라이선스: 명시되어 있지 않음. 원 코드의 라이선스 정책을 따를 가능성이 있다. Python 3.9 환경에서 테스트되었다.16

#### **3.2.4. GSNet (Graspness Discovery)**

* **명칭 및 발표 정보:** Graspness Discovery in Clutters for Fast and Accurate Grasp Detection (ICCV 2021, Wang et al.).19  
* **핵심 아이디어 및 방법론:** 기존 6-DOF 그리핑 방법들이 장면 내 모든 점을 동등하게 취급하거나 균일 샘플링에 의존하는 점이 속도와 정확도를 저해한다고 보고, "파지 가능성(graspness)"이라는 개념을 제안한다. Graspness는 기하학적 단서에 기반하여 혼잡한 장면에서 파지 가능한 영역을 구분하는 품질이다. 이를 측정하기 위한 선행 탐색(look-ahead searching) 방법을 제안하고, 실제 적용을 위해 이 과정을 근사하는 신경망인 "graspness model"을 개발했다. GSNet은 이 graspness model을 통합하여 낮은 품질의 예측을 조기에 필터링하는 종단간 2단계 네트워크이다.19  
* **입력 데이터:** 조밀한 장면 포인트 클라우드(dense scene point cloud).19  
* **특징 융합 방식:** 포인트 클라우드의 지역적 기하학적 단서를 활용하여 점별 graspness 점수를 예측한다. RGB 정보는 명시적으로 활용되지 않는 것으로 보이며, 기하학적 특징에 집중한다. 샘플링 계층에서 graspness가 높은 점들을 먼저 선택하고, 나머지는 연산 효율을 위해 이후 과정에서 제외한다.19  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** Nvidia GTX 1080Ti GPU 환경에서 추론 시, RealSense 데이터 기준 Cascaded Graspness Model (CGM)이 0.08초, Grasp Operation Model (GOM)이 0.02초로 총 0.10초가 소요된다고 보고되었다.19 Jetson Orin AGX에서의 성능이나 TensorRT 최적화에 대한 언급은 없다.  
* **파지 성공률 (SR):** GraspNet-1Billion 데이터셋의 "novel" 객체에 대해 RealSense 카메라 기준 23.98 AP (Average Precision), Kinect 카메라 기준 19.01 AP를 달성했다.19 이는 이전 SOTA 방법들 대비 큰 폭의 AP 향상(30+ AP)이라고 주장한다.20  
* **강인성:** Graspness 모델이 객체에 무관(object agnostic)하며 시점, 장면, 센서 변화에 강인하여 일반적이고 전이 가능한 모듈이라고 주장한다.19  
* **오픈소스 정보:**  
  * GitHub: https://github.com/graspnet/graspness\_unofficial (비공식 구현, 원 저자들의 graspnet-baseline 기반).21 원 논문에서는 코드 및 모델 공개 예정이라고 밝혔다.19  
  * 라이선스: 비공식 구현 저장소에 명시된 라이선스 확인 필요.

#### **3.2.5. NeuGraspNet**

* **명칭 및 발표 정보:** NeuGraspNet: Learning Implicit Representations for NeuGraspNet: Learning Implicit Representations for Neural Grasping.2222  
* **핵심 아이디어 및 방법론:** 신경망 기반 부피 표현(neural volumetric representations)과 표면 렌더링(surface rendering) 기술을 활용하여 6-DOF 그리핑을 재해석한다. 로봇 엔드 이펙터와 객체 표면 간의 상호작용을 공동으로 학습하여 지역 객체 표면을 렌더링하고 공유 특징 공간에서 그리핑 함수를 학습한다. 전역적(장면 수준) 특징을 그리핑 생성에, 지역적(그리핑 수준) 신경 표면 특징을 그리핑 평가에 사용한다. 이를 통해 부분적으로 관찰된 장면에서도 효과적이고 완전한 암시적(fully implicit) 6-DOF 그리핑 품질 예측이 가능하다.22  
* **입력 데이터:** 단일 임의 시점 깊이 입력(single random-view depth input), 이를 처리하여 3D TSDF(Truncated Signed Distance Field) 그리드를 생성한다.22  
* **특징 융합 방식:** 장면을 암시적 특징 볼륨(implicit feature volume)으로 인코딩한다. 객체 일부가 그리핑 포즈에 반응하는 방식을 인코딩하는 공유 지역 특징(shared local features)을 학습하여 SE(3) 공간에서 완전한 암시적 그리핑 품질 평가를 가능하게 한다.22  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** 그리핑 샘플링 및 품질 추론에 총 865ms가 소요된다고 보고되었다 (Table VI in 22). 사용된 하드웨어나 Jetson Orin AGX, TensorRT에 대한 언급은 없다.22  
* **파지 성공률 (SR):** 미지 객체 및 혼잡 환경에서의 일반화 능력을 목표로 한다. Evolved Grasping Analysis Dataset (EGAD) 및 시뮬레이션된 "pile", "packed" 장면, 실제 혼잡 환경에서의 모바일 매니퓰레이터 실험을 통해 성능을 입증했다.22 구체적인 SR 수치는 요약 정보에서 확인되지 않았다.  
* **강인성:** 임의 시점에서의 부분 관찰에도 강인한 그리핑 품질 예측을 목표로 한다.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/iROSA-lab/neugraspnet (프로젝트 웹사이트 sites.google.com/view/neugraspnet 에서 코드 링크 제공).23  
  * 라이선스: GitHub 저장소에서 확인 필요.

#### **3.2.6. Deep Object Pose Estimation (DOPE)**

* **명칭 및 발표 정보:** Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects (CoRL 2018, Tremblay et al.).18  
* **핵심 아이디어 및 방법론:** RGB 카메라로부터 입력받아 알려진(known) 객체의 탐지 및 6-DOF 포즈를 추정한다. 이는 직접적인 그리핑 알고리즘이라기보다는, 객체 포즈를 정확히 알면 사전 정의된 그리핑을 적용할 수 있게 하는 선행 단계로 활용될 수 있다.  
* **입력 데이터:** RGB 이미지.18  
* **특징 융합 방식:** RGB 이미지 특징을 사용하여 객체의 2D 바운딩 박스와 3D 꼭짓점의 2D 투영 위치를 예측하고, 이를 PnP 알고리즘과 결합하여 6-DOF 포즈를 계산한다.  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** Isaac ROS DOPE 프로젝트를 통해 하드웨어 가속 ROS 2 추론이 가능하며, Jetson AGX Xavier (JetPack 4.6) 및 x86/Ubuntu 20.04 (NVIDIA Titan X, 2080Ti, Titan RTX)에서 테스트되었다.18 Jetson Orin AGX에서의 직접적인 벤치마크는 명시되지 않았으나, Jetson 계열 지원은 긍정적이다. TensorRT 최적화가 Isaac ROS 패키지에 포함되었을 가능성이 높다.  
* **파지 성공률 (SR):** 이 알고리즘은 "알려진 객체"의 포즈를 추정하므로, 사용자의 "미지 객체" 요구사항과는 다소 거리가 있다. YCB, HOPE 데이터셋에서 평가되었다.18 그리핑 SR보다는 포즈 추정 정확도(ADD, ADD-S 등)로 평가된다.  
* **강인성:** 합성 데이터(NVISII 또는 Blenderproc 사용) 생성을 지원하여 다양한 환경에 대한 학습 데이터 확보가 가능하다.18  
* **오픈소스 정보:**  
  * GitHub: https://github.com/NVlabs/Deep\_Object\_Pose 18  
  * 라이선스: NVIDIA Source Code License.18

### **3.3. 표 1: 6-DOF 그리핑 알고리즘 비교 분석**

다음 표는 위에서 분석한 주요 오픈소스 6-DOF 그리핑 알고리즘들을 요약 비교한다.

| 알고리즘명 (발표연도, 학회) | 입력 데이터 | 핵심 방법론 요약 | 특징 융합 방식 | 실시간 성능 (FPS/ms, GPU) | Jetson Orin AGX (성능/최적화) | SR (미지객체/Cluttered, 데이터셋) | 오픈소스 (GitHub 링크, 라이선스) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **ASGrasp** (2024, ICRA) | RGB-D (Active Stereo: RGB \+ IR pair) | 투명/반사 객체 특화, 2-layer stereo network, GSNet 기반 Grasp Detection | RGB+IR 이미지 특징 기반 깊이 복원 및 3D 재구성, 복원된 포인트 클라우드 사용 | 정보 없음 | 정보 없음 | \>90% SR (Novel objects, Real robot, 자체 STD-GraspNet) 8 | jun7-shi/ASGrasp 7, 라이선스 미명시 |
| **EconomicGrasp** (2024, ECCV) | Point Cloud (from RGB-D) | Economic supervision, focal representation module, 리소스 효율성 (학습 시간, 메모리 절감) | 포인트 클라우드 기하학적 특징 중심, 상호작용적 그리핑 헤드, 복합 점수 추정 | 학습 시간/메모리 효율성 강조 (RTX 3090\) 10, 추론 FPS 정보 없음 | 정보 없음 | Novel AP (GraspNet-1B): Kinect 19.54, Realsense 25.48 10 | iSEE-Laboratory/EconomicGrasp 10, MIT License |
| **Contact-GraspNet (Official)** (2021, ICRA) | Depth (Point Cloud from depth), RGB-D 권장 | End-to-end, 3D 점을 잠재적 접촉점으로 취급, 4-DOF로 차원 축소 | 포인트 클라우드 기하학적 정보 중심, RGB는 분할/필터링에 활용 가능 | "효율적" 강조, 구체적 FPS 정보 없음 (Nvidia GPU \>=8GB VRAM 추론 권장) 16 | 정보 없음 | \>90% SR (Unseen objects, Structured clutter, ACRONYM) 13 | NVlabs/contact\_graspnet 16, NVIDIA Source Code License |
| **Contact-GraspNet (PyTorch Impl.)** (N/A) | Depth (Point Cloud from depth), RGB-D 권장 | 공식 Contact-GraspNet의 PyTorch 재구현 | 공식 버전과 유사 | 정보 없음 (Nvidia GPU \>=8GB VRAM 추론 권장) 16 | 정보 없음 | 원 논문과 성능 다를 수 있음 16 | elchun/contact\_graspnet\_pytorch 16, 라이선스 미명시 |
| **GSNet (Graspness Discovery)** (2021, ICCV) | Dense Scene Point Cloud | "Graspness" 개념 도입, 기하학적 단서 기반 파지 가능 영역 필터링, 2단계 네트워크 | 포인트 클라우드 지역 기하학적 단서 기반 graspness 예측 | 0.10s (RealSense, 1080Ti) 19 | 정보 없음 | Novel AP (GraspNet-1B, RealSense): 23.98 19 | graspnet/graspness\_unofficial (비공식) 21, 원 논문 코드 공개 예정 언급 |
| **NeuGraspNet** (2024 (예상), RSS) | Single random-view depth \-\> TSDF | Neural volumetric representation, surface rendering, implicit grasp quality prediction | 암시적 특징 볼륨, 전역 특징(생성)/지역 특징(평가) 융합 | 865 ms total inference 22 | 정보 없음 | EGAD, 시뮬레이션, 실제 혼잡 환경 테스트 22, 구체적 SR 정보 부족 | iROSA-lab/neugraspnet 23, 라이선스 GitHub 확인 필요 |
| **DOPE (Deep Object Pose Estimation)** (2018, CoRL) | RGB 이미지 | 알려진 객체 6-DOF 포즈 추정 | RGB 이미지 특징 기반 2D-3D 대응점 예측 후 PnP | Isaac ROS 통해 HW 가속 가능 (Jetson AGX Xavier 테스트) 18 | Jetson 계열 지원, TensorRT 최적화 가능성 높음 | "알려진 객체" 대상 (YCB, HOPE) 18, 미지 객체 SR 해당 없음 | NVlabs/Deep\_Object\_Pose 18, NVIDIA Source Code License |

### **3.4. 심층 분석 및 고려사항**

많은 최신 그리핑 알고리즘들이 GraspNet-1Billion 데이터셋을 주요 벤치마크로 활용하고 있다.4 이 데이터셋은 "미지 객체(novel objects)" 범주를 포함하고 있어, 사용자의 미지 객체에 대한 그리핑 성능 요구사항을 평가하는 데 유용한 기준을 제공한다. 그러나 GraspNet-1Billion 데이터셋은 "보통 수준의 복잡도(moderately cluttered)"를 가진 환경으로 구성되어 있다는 점을 인지해야 한다.4 사용자가 고려하는 "복잡한 장면(complex scene)"의 모든 측면을 완벽하게 반영하지 못할 수 있다. GraspClutter6D와 같이 "매우 혼잡한 장면(highly cluttered scenes)"을 다루는 데이터셋에서의 성능 검증 결과가 있다면, 사용자의 요구사항에 더욱 부합하는 강인성을 판단하는 데 도움이 될 것이다.15 따라서 GraspNet-1Billion에서의 "novel" 객체에 대한 성능은 좋은 출발점이지만, 알고리즘이 더 복잡한 실제 환경에서도 잘 일반화될 수 있을지는 추가적인 검토가 필요하다.

Jetson Orin AGX와 같은 엣지 디바이스에서의 실시간 성능 및 TensorRT 최적화에 대한 직접적인 언급은 대부분의 연구에서 찾아보기 어렵다. ASGrasp, EconomicGrasp 등의 알고리즘 README나 논문에서도 Jetson Orin 관련 벤치마크나 최적화 논의는 부족했다.7 이는 학계 연구가 주로 알고리즘의 혁신성과 고성능 GPU에서의 성능에 초점을 맞추고, 엣지 디바이스로의 배포는 후속 엔지니어링 과제로 남겨두는 경향이 있기 때문으로 분석된다. 일반적인 Jetson Orin의 성능 26이나 YOLO와 같은 모델이 TensorRT를 통해 Jetson에서 높은 FPS를 달성하는 사례 28는 잠재력을 보여주지만, 복잡한 6-DOF 그리핑 네트워크에 그대로 적용되기는 어렵다. 따라서 사용자는 알고리즘 선택 시 모델의 구조적 복잡도, 연산량 등을 고려하여 Jetson Orin AGX에서의 실행 가능성을 가늠해야 하며, 필요한 경우 직접 포팅, 양자화, 프루닝, TensorRT 변환 등의 최적화 작업을 수행해야 할 가능성이 높다. EconomicGrasp과 같이 리소스 효율성을 명시적으로 강조하는 알고리즘 10이나, PointNet++ 3 또는 ResNet과 같이 TensorRT 변환 및 최적화가 비교적 용이하다고 알려진 백본 네트워크를 사용하는 알고리즘이 상대적으로 유리할 수 있다.

"RGB-D 정보의 지능적 융합"은 여전히 활발히 연구되는 분야로, 단순히 XYZRGB 포인트 클라우드를 입력으로 사용하는 것을 넘어 다양한 접근 방식이 존재한다. 예를 들어, ASGrasp는 투명 객체에 대응하기 위해 RGB와 IR 정보를 활용하여 깊이를 재구성하고 이를 그리핑에 사용하며 1, NeuGraspNet은 깊이 정보로부터 암시적 특징 볼륨을 만들고 공유 지역 특징을 학습한다.22 이는 "지능적 융합"이 단일 해결책이 아니며, 해결하고자 하는 특정 문제(예: 투명 객체, 부분적 관찰)나 전체 시스템 아키텍처에 따라 최적의 융합 전략이 달라질 수 있음을 시사한다. 사용자는 알고리즘이 RGB-D 데이터를 사용하는지 여부뿐만 아니라, *어떻게* 정보를 융합하는지, 그리고 그 방식이 NanoOwl/NanoSAM의 강점 및 예상되는 작업 환경/객체 유형과 잘 부합하는지를 면밀히 검토해야 한다.

## **4\. Point Cloud 업샘플링 기법과 그리핑 성능 향상 가능성**

### **4.1. Edge Device 환경에서의 희소한 Point Cloud 문제**

엣지 디바이스 환경에서 사용되는 깊이 카메라, 특히 저가형 센서나 특정 조명 조건 하에서는 획득되는 포인트 클라우드의 밀도가 낮거나(희소하거나) 불균일한 경우가 많다.2 이러한 희소한 포인트 클라우드는 객체의 세밀한 기하학적 특징을 충분히 표현하지 못하여, 그리핑 위치 선정의 정확도를 떨어뜨리고 결과적으로 그리핑 성공률을 저하시키는 주요 원인이 될 수 있다. 사용자의 연구에서도 이 문제를 해결하기 위한 방안으로 포인트 클라우드 업샘플링(upsampling) 적용을 고려하고 있다.

### **4.2. 주요 Point Cloud 업샘플링 알고리즘 (2018년 이후, 오픈소스 중심)**

포인트 클라우드 업샘플링은 주어진 희소한 포인트 클라우드로부터 더 조밀하고 균일한 포인트 클라우드를 생성하는 기술이다. 최근 딥러닝 기반의 다양한 업샘플링 알고리즘들이 제안되었으며, 이 중 오픈소스로 공개된 주요 알고리즘은 다음과 같다.

#### **4.2.1. PU-Net**

* **명칭 및 발표 정보:** PU-Net: Point Cloud Upsampling Network (CVPR 2018, Yu et al.).2  
* **핵심 아이디어 및 방법론:** 딥러닝 기반 포인트 클라우드 업샘플링의 초기 연구 중 하나이다. 각 포인트별로 다중 레벨 특징(multi-level features)을 학습하고, 특징 공간에서 다중 분기 컨볼루션 유닛(multi-branch convolution unit)을 통해 암시적으로 포인트 셋을 확장한다. 확장된 특징은 다수의 특징으로 분할된 후 업샘플링된 포인트 셋으로 재구성된다.30 패치 레벨에서 적용되며, 업샘플링된 포인트가 균일한 분포로 기본 표면 위에 남아 있도록 하는 공동 손실 함수(joint loss function)를 사용한다.  
* **입력 및 출력 특성:** 희소한 포인트 클라우드를 입력으로 받아 더 조밀한 포인트 클라우드를 출력한다. 입력 포인트 클라우드가 매우 클 경우 패치 단위로 나누어 처리하는 것이 권장된다.31  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** Tensorflow 기반으로 구현되었으며, PointNet++의 TF 연산자를 사용한다.31 실시간 성능이나 Jetson Orin AGX에서의 구체적인 성능 데이터는 제공되지 않았다.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/yulequan/PU-Net 31  
  * 라이선스: 명시되어 있지 않으나, 연구용 코드 공개로 추정된다.

#### **4.2.2. RepKPU**

* **명칭 및 발표 정보:** RepKPU: Point Cloud Upsampling with Kernel Point Representation and Deformation (CVPR 2024, Rong et al.).33  
* **핵심 아이디어 및 방법론:** 커널 포인트 표현(Kernel Point Representation)과 변형(Deformation)을 통해 포인트 클라우드를 업샘플링한다. 최신 연구로, 세부적인 방법론은 논문 참조가 필요하다.  
* **입력 및 출력 특성:** 임의의 스케일 업샘플링을 지원한다 (--flexible 옵션).33  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** Python 3.6, PyTorch 1.10.1, CUDA 12.2 환경에서 테스트되었다.33 엣지 디바이스 성능은 언급되지 않았다.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/EasyRy/RepKPU 33  
  * 라이선스: MIT License.33

#### **4.2.3. Semi-supervised Edge-aware Point Cloud Upsampling Network**

* **명칭 및 발표 정보:** 프로젝트명으로 보이며, 특정 논문 정보는 GitHub README에서 직접 확인 필요 (An-u-rag).34  
* **핵심 아이디어 및 방법론:** 3D 딥러닝과 전통적인 컴퓨터 비전 기술을 결합하여, 가장자리(edge)를 인식하고 미세한 디테일을 보존하면서 포인트 클라우드를 정확하게 업샘플링하는 것을 목표로 한다. 준지도 학습(semi-supervised) 방식을 사용한다.34  
* **입력 및 출력 특성:** S3DIS 데이터셋 등을 사용하여 학습 및 테스트가 가능하다.34  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** Python 3.9, PyTorch 1.13.0, CUDA 11.7 환경에서 테스트되었다.34 엣지 디바이스 성능은 언급되지 않았다.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/An-u-rag/pointcloud-upsampling 34  
  * 라이선스: 명시되어 있지 않음.

#### **4.2.4. SPU-IMR**

* **명칭 및 발표 정보:** SPU-IMR: Self-supervised Arbitrary-scale Point Cloud Upsampling via Iterative Mask-recovery Network (arXiv 2025.02).35  
* **핵심 아이디어 및 방법론:** 자가지도(self-supervised) 방식으로 임의의 스케일(arbitrary-scale) 업샘플링을 수행하며, 반복적인 마스크 복구 네트워크(iterative mask-recovery network)를 사용한다.  
* **입력 및 출력 특성:** 논문 확인 필요.  
* **실시간 성능 및 Jetson Orin AGX 적용 가능성:** 정보 없음.  
* **오픈소스 정보:**  
  * GitHub: https://github.com/nieziming/SPU-IMR (arXiv 페이지에서 링크 확인됨 35)  
  * 라이선스: GitHub 저장소에서 확인 필요.

### **4.3. 업샘플링 적용 시 그리핑 알고리즘의 정확도 및 강인성 변화 예측**

포인트 클라우드 업샘플링을 통해 입력 데이터의 밀도를 높이면, 객체 표면의 세밀한 디테일이 향상되어 그리핑 포인트 선정의 정확도가 개선될 잠재력이 있다. 이는 특히 복잡한 형상을 가진 객체나 작은 접촉 영역만이 허용되는 경우에 유리할 수 있다.

그러나 몇 가지 고려할 점이 있다. 첫째, 업샘플링 과정에서 원본 데이터에 존재하던 노이즈가 함께 증폭될 수 있다.2 일부 업샘플링 알고리즘은 노이즈 제거 기능을 포함하기도 하지만, 그렇지 않은 경우 오히려 그리핑 성능에 악영향을 줄 수 있다. 둘째, 업샘플링된 데이터에 대해 기존 그리핑 알고리즘이 얼마나 잘 일반화될 수 있을지는 미지수이다. 그리핑 알고리즘이 특정 밀도나 분포 특성을 가진 데이터에 과적합(overfitting)되어 있다면, 업샘플링된 데이터에서 기대만큼의 성능 향상을 보이지 못할 수도 있다.

현재까지 대부분의 포인트 클라우드 업샘플링 연구는 3D 형상 복원 품질(예: Chamfer Distance, Hausdorff Distance)이나 분류/분할 정확도 향상에 초점을 맞추고 있으며, 그리핑 성공률에 미치는 직접적이고 정량적인 평가는 상대적으로 부족하다.2 비록 이들 연구에서 로봇 작동(robotic operations)을 응용 분야로 언급하기는 하지만, 그리핑 작업 자체에 대한 심층적인 분석은 드물다. 이는 업샘플링 기법을 그리핑 시스템에 도입할 경우, 사용자가 자체적으로 그리핑 성능 변화를 실험하고 검증해야 함을 의미한다. 업샘플링이 특정 그리핑 알고리즘에 필요한 핵심적인 기하학적 특징을 실제로 개선하는지, 아니면 불필요한 아티팩트를 추가하는지 여부는 실제 실험을 통해 확인해야 한다.

### **4.4. Jetson Orin AGX에서의 실시간 전처리로서의 업샘플링 적용 가능성 검토**

Jetson Orin AGX와 같은 엣지 디바이스에서 업샘플링을 실시간 전처리 단계로 포함시키기 위해서는 업샘플링 알고리즘 자체의 계산 복잡도와 추론 시간을 신중하게 고려해야 한다. 업샘플링은 전체 인식-추적-분할-그리핑 파이프라인에 추가적인 지연 시간(latency)을 발생시킨다. 만약 업샘플링으로 인한 지연이 시스템의 실시간 요구사항을 만족시키지 못한다면, 그 효용성은 크게 떨어질 것이다.

경량화된 딥러닝 모델에 대한 일반적인 연구 36나 경량 포인트 클라우드 처리 관련 연구 35가 존재하지만, 이들이 Jetson Orin AGX와 같은 특정 엣지 디바이스에서 그리핑에 유의미한 수준의 품질 향상을 제공하면서도 충분한 실시간 성능을 보장할 수 있을지는 아직 불분명하다. 업샘플링으로 얻을 수 있는 잠재적인 그리핑 성능 향상과 추가되는 계산 비용 간의 트레이드오프를 면밀히 분석해야 한다. 예를 들어, 업샘플링에 수백 밀리초가 소요되지만 그리핑 성공률이 단 몇 퍼센트만 향상된다면, 실시간 시스템에서는 비효율적일 수 있다.

따라서, 만약 업샘플링을 적용한다면 극도로 효율적인 경량 알고리즘을 선택하거나, TensorRT와 같은 도구를 활용하여 Jetson Orin AGX에 대한 철저한 최적화가 선행되어야 한다. 대안으로는, 희소한 포인트 클라우드에 처음부터 강인하게 설계된 그리핑 알고리즘을 선택하거나, 깊이 센서의 해상도 설정 조정, 필터링 기법 적용 등을 통해 입력 데이터의 품질을 개선하는 방안도 고려할 수 있다. 일부 최신 그리핑 알고리즘은 내부적으로 포인트 클라우드 정제(refinement)나 업샘플링과 유사한 메커니즘을 포함하기도 한다 (예: E3GNet 논문에서 참조된 Tang et al. 은 업샘플링을 통한 개선 언급 39). 이러한 경우, 별도의 외부 업샘플링 단계가 불필요하거나 중복될 수 있으므로 확인이 필요하다.

### **4.5. 표 2: Point Cloud 업샘플링 알고리즘 비교**

다음 표는 위에서 분석한 주요 오픈소스 포인트 클라우드 업샘플링 알고리즘들을 요약 비교한다.

| 알고리즘명 (발표연도, 학회) | 핵심 아이디어 | 입력 Point Cloud 특성 | 출력 Point Cloud 특성 (밀도 증가율 등) | 실시간성 (FPS/ms, GPU) | Jetson Orin AGX 적용 가능성 | 오픈소스 (GitHub 링크, 라이선스) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **PU-Net** (2018, CVPR) | Multi-level feature, multi-branch convolution, patch-level processing | 희소 포인트 클라우드 | 조밀 포인트 클라우드, 균일 분포 목표 | 정보 없음 | 정보 없음 | yulequan/PU-Net 32, 라이선스 미명시 |
| **RepKPU** (2024, CVPR) | Kernel Point Representation, Deformation | 희소 포인트 클라우드 | 임의 스케일 업샘플링 지원 | 정보 없음 | 정보 없음 | EasyRy/RepKPU 33, MIT License |
| **Edge-aware Upsampling** (An-u-rag) | Semi-supervised, edge-aware, detail preservation | 희소 포인트 클라우드 | 가장자리 및 디테일 보존 업샘플링 | 정보 없음 | 정보 없음 | An-u-rag/pointcloud-upsampling 34, 라이선스 미명시 |
| **SPU-IMR** (2025 (예상), arXiv) | Self-supervised, arbitrary-scale, iterative mask-recovery network | 희소 포인트 클라우드 | 임의 스케일 업샘플링 | 정보 없음 | 정보 없음 | nieziming/SPU-IMR (GitHub 링크 35), 라이선스 GitHub 확인 필요 |

## **5\. 결론 및 시스템 적용을 위한 제언**

### **5.1. 주요 연구 결과 요약**

본 보고서는 Jetson Orin AGX 64GB 환경에서 자연어 명령 기반 로봇 그리핑 시스템 개발을 목표로 하는 사용자의 연구를 지원하기 위해, 최신 6-DOF 그리핑 알고리즘과 포인트 클라우드 업샘플링 기술을 심층 분석하였다.

* **입력 데이터 유형:** NanoOwl/NanoSAM과의 시너지 및 미지 객체에 대한 풍부한 정보 활용을 위해, 3D 기하 정보와 RGB 시각 정보를 결합한 \*\*XYZRGB 포인트 클라우드(RGB-D 융합)\*\*가 가장 적합한 입력 형태로 판단된다. 이는 사용자의 초기 판단과도 일치하며, 자연어 명령에서 언급될 수 있는 객체의 시각적 속성을 그리핑 단계까지 일관성 있게 전달하는 데 유리하다.  
* **6-DOF 그리핑 알고리즘:** 2018년 이후 발표된 다수의 오픈소스 알고리즘(ASGrasp, EconomicGrasp, Contact-GraspNet, GSNet, NeuGraspNet, DOPE 등)을 분석한 결과, 각 알고리즘은 서로 다른 접근 방식, 입력 데이터 특성, 성능적 강점을 가지고 있었다. 대부분 GraspNet-1Billion 데이터셋을 사용하여 미지 객체에 대한 성능을 평가하지만, Jetson Orin AGX에서의 직접적인 벤치마크나 TensorRT 최적화 사례는 드물어, 엣지 디바이스 적용 시 추가적인 엔지니어링 노력이 필요함을 시사한다.  
* **Point Cloud 업샘플링:** 희소한 포인트 클라우드로 인한 그리핑 성능 저하를 개선할 잠재력은 있으나, 대부분의 연구가 형상 복원 품질에 초점을 맞추고 있어 그리핑 성공률에 대한 직접적인 영향 평가는 부족하다. 또한, 엣지 디바이스에서의 추가적인 연산 부하로 인해 전체 시스템의 실시간성을 저해할 위험이 있어 신중한 접근이 요구된다.

### **5.2. 사용자 연구 목표 및 Jetson Orin AGX 환경에 적합한 그리핑 알고리즘 및 입력 데이터 전략 추천**

* **입력 데이터 전략:**  
  * 앞서 분석된 바와 같이, NanoOwl/NanoSAM과의 시너지, 미지 객체에 대한 시각적 및 기하학적 정보 활용 극대화를 위해 \*\*XYZRGB 포인트 클라우드(RGB-D 융합)\*\*를 기본 입력으로 사용할 것을 강력히 권장한다. 이는 사용자의 "인식 후 추적" 및 "필요 기반 정교화" 아키텍처와도 잘 부합하며, 자연어 명령 처리의 일관성을 유지하는 데 도움이 된다.  
* **그리핑 알고리즘 선정 제안:**  
  * 사용자의 핵심 요구사항인 **미지 객체에 대한 높은 SR, 다양한 환경(혼잡, 복잡)에서의 강인성, Jetson Orin AGX에서의 실시간성, 오픈소스**를 고려할 때, 다음 알고리즘들을 우선적으로 검토할 것을 제안한다.  
    * **1순위 후보군:**  
      * **EconomicGrasp:** 리소스 효율성을 명시적으로 강조하며 10, GraspNet-1Billion "Novel" 객체에 대한 준수한 AP를 보인다.10 포인트 클라우드 입력을 사용하므로 RGB-D 센서로부터 쉽게 변환 가능하며, MIT 라이선스로 공개되어 있다.10 Jetson Orin AGX에서의 성능은 사용자가 직접 검증하고 최적화해야 하지만, 모델의 "경제적" 설계 철학이 엣지 환경에 유리하게 작용할 수 있다.  
      * **Contact-GraspNet (NVlabs 공식 또는 PyTorch 재구현):** 널리 사용되고 검증된 알고리즘으로, 미지 객체 및 혼잡 환경에서 높은 성공률을 보고했다.13 포인트 클라우드 입력을 사용하며, RGB-D 데이터와 분할 마스크를 활용하여 성능을 더욱 향상시킬 수 있다.16 NVlabs 공식 코드는 NVIDIA Source Code License를 따르며 18, PyTorch 재구현 버전도 존재하여 선택의 폭이 있다. 다만, 모델 복잡도가 상대적으로 높아 Jetson Orin AGX에서의 실시간 구동을 위해서는 적극적인 최적화가 필요할 수 있다.  
    * **2순위 고려군:**  
      * **GSNet (Graspness Discovery):** "Graspness"라는 독창적인 개념을 통해 효율적인 그리핑 후보 생성을 시도하며, GraspNet-1Billion "Novel" 객체에 대해 우수한 AP를 기록했다.19 포인트 클라우드 입력을 사용한다. 비공식 구현 코드가 존재하며 21, 원 논문에서도 코드 공개를 약속했다.19 Jetson 환경에서의 최적화 여부는 불명확하다.  
      * **ASGrasp:** 투명 객체 그리핑이라는 특정 난제 해결에 매우 강력한 성능을 보이지만 8, 일반적인 불투명 객체 및 다양한 혼잡 환경에서의 성능, 그리고 Jetson Orin AGX에서의 효율성에 대한 추가 검증이 필요하다. 입력으로 액티브 스테레오 카메라의 IR 이미지를 요구하는 점도 고려해야 한다.1  
  * **선정 시 추가 고려사항:**  
    * 각 알고리즘의 학습에 사용된 데이터셋(GraspNet-1Billion, ACRONYM 등)의 특성과 사용자가 실제 적용할 환경 및 객체 간의 유사성을 고려해야 한다. Domain gap이 클 경우 Sim-to-Real 전이 학습 전략이 중요해진다.  
    * 사용자가 ROS 2 기반으로 각 모듈을 노드화하여 시스템을 구성할 계획이므로, 선택된 그리핑 알고리즘의 ROS 2 지원 여부 또는 ROS 2로의 통합 용이성도 중요한 실용적 고려사항이다. 예를 들어, DOPE는 Isaac ROS를 통해 ROS 2를 지원한다.18

### **5.3. Point Cloud 업샘플링 적용 여부에 대한 고려사항 및 제언**

* 현재로서는 포인트 클라우드 업샘플링을 기본 파이프라인에 포함하기보다는, **선택적으로 실험하거나 후순위로 고려**할 것을 제안한다.  
* **이유:**  
  1. **실시간성 저해 우려:** 업샘플링은 추가적인 계산 부하를 발생시켜, 이미 복잡한 인식-추적-분할-그리핑 파이프라인의 전체 지연 시간을 증가시킬 수 있다. 특히 Jetson Orin AGX와 같은 엣지 디바이스에서는 이 부담이 더욱 클 수 있다.  
  2. **그리핑 성능 향상 효과의 불확실성:** 대부분의 업샘플링 연구는 형상 복원 품질을 기준으로 평가되며, 실제 그리핑 성공률 향상에 대한 직접적이고 정량적인 증거는 부족하다. 업샘플링이 오히려 노이즈를 증폭시키거나, 그리핑에 중요하지 않은 디테일만 추가할 가능성도 배제할 수 없다.  
* **만약 적용을 고려한다면:**  
  * 매우 가볍고(lightweight) 빠른 업샘플링 알고리즘(예: SPU-IMR 등 오픈소스 경량 모델)을 신중하게 선택해야 한다.  
  * 선택된 알고리즘에 대해 Jetson Orin AGX 환경에서 TensorRT 등을 활용한 철저한 최적화를 수행하고, 그리핑 성능에 미치는 영향을 정량적으로 검증하는 과정이 반드시 선행되어야 한다.  
* **대안적 접근:**  
  * 희소한 포인트 클라우드에 상대적으로 강인한 그리핑 알고리즘을 우선적으로 탐색하고 선택한다.  
  * 사용하는 깊이 센서의 해상도 설정을 최적화하거나, 포인트 클라우드 전처리 단계에서 통계적 아웃라이어 제거(Statistical Outlier Removal)와 같은 필터링 기법을 적용하여 입력 데이터의 품질 자체를 개선하려는 노력이 더 효과적일 수 있다.

### **5.4. Jetson Orin AGX 환경에서의 End-to-End 최적화 방안 제언**

사용자가 언급한 바와 같이, Jetson Orin AGX 환경에서 전체 시스템의 실시간성과 효율성을 확보하기 위해서는 다음과 같은 소프트웨어 공학적 기법들의 적극적인 적용이 필수적이다:

* **TensorRT 최적화:** 학습된 딥러닝 모델(NanoOwl, NanoSAM, 선택된 그리핑 알고리즘, 업샘플링 알고리즘 등)을 TensorRT로 변환하여 추론 속도를 극대화해야 한다. 이는 FP16/INT8 정밀도 활용, 레이어 및 텐서 융합 등을 포함한다.26  
* **모델 경량화:** TensorRT 최적화와 더불어, 모델 프루닝(pruning), 지식 증류(knowledge distillation), 양자화(quantization) 등의 기법을 적용하여 모델 크기를 줄이고 연산 효율을 높이는 방안을 적극 고려해야 한다.26  
* **비동기 처리 (Asynchronous Processing):** ROS 2 노드 간 통신이나 각 모듈 내의 독립적인 작업들은 비동기적으로 처리하여 병목 현상을 최소화하고 시스템 전체의 응답성을 향상시켜야 한다.  
* **공유 메모리 활용 (Shared Memory):** 노드 간 대용량 데이터(예: 이미지, 포인트 클라우드) 전송 시, 데이터 복사로 인한 오버헤드를 줄이기 위해 ROS 2의 공유 메모리 기능이나 이와 유사한 IPC(Inter-Process Communication) 메커니즘을 활용하는 것이 효과적이다.  
* **파이프라인 병목 분석:** 전체 시스템을 구성하는 각 모듈(자연어 이해, 객체 탐지, 추적, 분할, (업샘플링), 그리핑 포즈 추정, 로봇 제어)의 연산 시간과 데이터 전송 시간을 면밀히 측정하고 분석하여, 최적화가 가장 시급한 병목 지점을 찾아 집중적으로 개선해야 한다.  
* **NVIDIA JetPack SDK 활용:** Jetson Orin AGX는 JetPack SDK를 통해 CUDA, cuDNN, TensorRT 등 최적화된 라이브러리를 제공하므로, 이를 최대한 활용해야 한다.26

### **5.5. 향후 연구 및 개발 방향 제안**

본 연구를 통해 개발하고자 하는 지능형 로봇 그리핑 시스템의 성공적인 구현과 지속적인 발전을 위해 다음과 같은 추가적인 연구 및 개발 방향을 제안한다:

* **Sim-to-Real Domain Gap 완화 전략 심층 연구:** 학습 기반 모델을 적극적으로 활용할 경우, 시뮬레이션 환경에서 학습된 모델이 실제 환경에서 성능 저하를 보이는 도메인 격차(domain gap) 문제가 발생할 수 있다. 이를 효과적으로 완화하기 위한 최신 연구 동향(예: domain randomization, adversarial training, MUNIT, CycleGAN 등)을 파악하고, 선택하려는 그리핑 알고리즘이 이러한 전략을 이미 포함하고 있거나 적용하기 용이한지 평가하는 것이 중요하다.  
* **지속적인 학습 및 일반화 성능 평가 방법론 구축:** 실제 환경에서 다양한 객체와 조작 시나리오에 대해 시스템의 일반화 성능을 지속적으로 평가하고 개선해나갈 수 있는 체계적인 방법론을 구축해야 한다. 이는 새로운 데이터에 대한 온라인/오프라인 학습 전략, 성능 저하 감지 및 재학습 트리거링 메커니즘 등을 포함할 수 있다.  
* **실제 로봇 시스템과의 통합 및 실증을 통한 반복적 개선:** 개발된 각 모듈과 전체 파이프라인을 실제 로봇 매니퓰레이터 및 센서 시스템과 통합하고, 다양한 실제 환경에서 사용자 시나리오 기반의 실증 실험을 수행해야 한다. 이를 통해 예상치 못한 문제점을 발견하고, 시스템의 강인성과 사용성을 반복적으로 개선해나가야 한다. 특히, NanoOwl/NanoSAM의 추적 기능과 그리핑 모듈 간의 데이터 흐름 및 동기화, "필요 기반 정교화" 전략의 실제 효용성 검증이 중요하다.

궁극적으로, 본 보고서에서 제시된 분석과 제언들이 사용자의 지능형 로봇 그리핑 시스템 개발 연구에 실질적인 도움이 되어, 자연어 명령에 따라 다양한 객체를 실시간으로, 정확하고, 안정적으로 파지할 수 있는 로봇 시스템을 성공적으로 구현하는 데 기여할 수 있기를 바란다.

#### **참고 자료**

1. \[Literature Review\] ASGrasp: Generalizable Transparent Object Reconstruction and Grasping from RGB-D Active Stereo Camera \- Moonlight, 5월 8, 2025에 액세스, [https://www.themoonlight.io/en/review/asgrasp-generalizable-transparent-object-reconstruction-and-grasping-from-rgb-d-active-stereo-camera](https://www.themoonlight.io/en/review/asgrasp-generalizable-transparent-object-reconstruction-and-grasping-from-rgb-d-active-stereo-camera)  
2. Representation Learning of Point Cloud Upsampling in Global and Local Inputs \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/html/2501.07076v2](https://arxiv.org/html/2501.07076v2)  
3. Representation Learning of Point Cloud Upsampling in Global and Local Inputs \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/pdf/2501.07076?](https://arxiv.org/pdf/2501.07076)  
4. GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping | Request PDF, 5월 8, 2025에 액세스, [https://www.researchgate.net/publication/343461561\_GraspNet-1Billion\_A\_Large-Scale\_Benchmark\_for\_General\_Object\_Grasping](https://www.researchgate.net/publication/343461561_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping)  
5. GraspNet-1Billion \- OpenDataLab, 5월 8, 2025에 액세스, [https://opendatalab.com/OpenDataLab/GraspNet-1Billion/download](https://opendatalab.com/OpenDataLab/GraspNet-1Billion/download)  
6. 6-DoF Closed-Loop Grasping with Reinforcement Learning \- SINTEF Open, 5월 8, 2025에 액세스, [https://sintef.brage.unit.no/sintef-xmlui/bitstream/handle/11250/3172888/Herland\_etal\_2024.pdf?sequence=4\&isAllowed=y](https://sintef.brage.unit.no/sintef-xmlui/bitstream/handle/11250/3172888/Herland_etal_2024.pdf?sequence=4&isAllowed=y)  
7. jun7-shi/ASGrasp: \[ICRA 2024\]ASGrasp: Generalizable ... \- GitHub, 5월 8, 2025에 액세스, [https://github.com/jun7-shi/ASGrasp](https://github.com/jun7-shi/ASGrasp)  
8. \[2405.05648\] ASGrasp: Generalizable Transparent Object Reconstruction and Grasping from RGB-D Active Stereo Camera \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/abs/2405.05648](https://arxiv.org/abs/2405.05648)  
9. arxiv.org, 5월 8, 2025에 액세스, [https://arxiv.org/pdf/2405.05648](https://arxiv.org/pdf/2405.05648)  
10. iSEE-Laboratory/EconomicGrasp: (ECCV 2024\) Official ... \- GitHub, 5월 8, 2025에 액세스, [https://github.com/iSEE-Laboratory/EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp)  
11. arXiv:2407.08366v1 \[cs.RO\] 11 Jul 2024, 5월 8, 2025에 액세스, [https://arxiv.org/pdf/2407.08366?](https://arxiv.org/pdf/2407.08366)  
12. An Economic Framework for 6-DoF Grasp Detection, 5월 8, 2025에 액세스, [https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/papers/03946.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03946.pdf)  
13. Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes | Request PDF, 5월 8, 2025에 액세스, [https://www.researchgate.net/publication/355434800\_Contact-GraspNet\_Efficient\_6-DoF\_Grasp\_Generation\_in\_Cluttered\_Scenes](https://www.researchgate.net/publication/355434800_Contact-GraspNet_Efficient_6-DoF_Grasp_Generation_in_Cluttered_Scenes)  
14. Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes | NVIDIA Seattle Robotics Lab, 5월 8, 2025에 액세스, [https://research.nvidia.com/labs/srl/publication/sundermeyer-2021-contact-grasp-net/](https://research.nvidia.com/labs/srl/publication/sundermeyer-2021-contact-grasp-net/)  
15. GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/html/2504.06866v1](https://arxiv.org/html/2504.06866v1)  
16. elchun/contact\_graspnet\_pytorch: Pytorch implementation of Contact-GraspNet \- GitHub, 5월 8, 2025에 액세스, [https://github.com/elchun/contact\_graspnet\_pytorch](https://github.com/elchun/contact_graspnet_pytorch)  
17. NVlabs/contact\_graspnet: Efficient 6-DoF Grasp Generation in Cluttered Scenes \- GitHub, 5월 8, 2025에 액세스, [https://github.com/NVlabs/contact\_graspnet](https://github.com/NVlabs/contact_graspnet)  
18. NVlabs/Deep\_Object\_Pose: Deep Object Pose Estimation (DOPE) – ROS inference (CoRL 2018\) \- GitHub, 5월 8, 2025에 액세스, [https://github.com/NVlabs/Deep\_Object\_Pose](https://github.com/NVlabs/Deep_Object_Pose)  
19. openaccess.thecvf.com, 5월 8, 2025에 액세스, [https://openaccess.thecvf.com/content/ICCV2021/papers/Wang\_Graspness\_Discovery\_in\_Clutters\_for\_Fast\_and\_Accurate\_Grasp\_Detection\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)  
20. Graspness Discovery in Clutters for Fast and Accurate Grasp Detection \- ICCV 2021 Open Access Repository, 5월 8, 2025에 액세스, [https://openaccess.thecvf.com/content/ICCV2021/html/Wang\_Graspness\_Discovery\_in\_Clutters\_for\_Fast\_and\_Accurate\_Grasp\_Detection\_ICCV\_2021\_paper.html](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html)  
21. graspnet/graspness\_unofficial: Unofficial implementation of ICCV 2021 paper "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" \- GitHub, 5월 8, 2025에 액세스, [https://github.com/graspnet/graspness\_unofficial](https://github.com/graspnet/graspness_unofficial)  
22. www.roboticsproceedings.org, 5월 8, 2025에 액세스, [https://www.roboticsproceedings.org/rss20/p046.pdf](https://www.roboticsproceedings.org/rss20/p046.pdf)  
23. NeuGraspNet \- Google Sites, 5월 8, 2025에 액세스, [https://sites.google.com/view/neugraspnet](https://sites.google.com/view/neugraspnet)  
24. 6-DoF Grasp Detection in Clutter with Enhanced Receptive Field and Graspable Balance Sampling \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/html/2407.01209v2](https://arxiv.org/html/2407.01209v2)  
25. 6-DoF Grasp Detection in Clutter with Enhanced Receptive Field and Graspable Balance Sampling \- arXiv, 5월 8, 2025에 액세스, [http://arxiv.org/pdf/2407.01209](http://arxiv.org/pdf/2407.01209)  
26. Deploying DeepSeek AI on NVIDIA Jetson AGX Orin: A Free, Open-Source MIT- Licensed Solution for High-Performance Edge AI in Natural Language Processing and Computer Vision \- ResearchGate, 5월 8, 2025에 액세스, [https://www.researchgate.net/publication/388401833\_Deploying\_DeepSeek\_AI\_on\_NVIDIA\_Jetson\_AGX\_Orin\_A\_Free\_Open-Source\_MIT-\_Licensed\_Solution\_for\_High-Performance\_Edge\_AI\_in\_Natural\_Language\_Processing\_and\_Computer\_Vision](https://www.researchgate.net/publication/388401833_Deploying_DeepSeek_AI_on_NVIDIA_Jetson_AGX_Orin_A_Free_Open-Source_MIT-_Licensed_Solution_for_High-Performance_Edge_AI_in_Natural_Language_Processing_and_Computer_Vision)  
27. Jetson Benchmarks \- NVIDIA Developer, 5월 8, 2025에 액세스, [https://developer.nvidia.com/embedded/jetson-benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)  
28. YOLO11 TensorRT Object Detection on Jetson Nano Orin with 100FPS on Live Webcam with Ultralytics \- YouTube, 5월 8, 2025에 액세스, [https://www.youtube.com/watch?v=nQBOkGR\_lg0](https://www.youtube.com/watch?v=nQBOkGR_lg0)  
29. A model-free 6-DOF grasp detection method based on point clouds of local sphere area, 5월 8, 2025에 액세스, [https://www.researchgate.net/publication/369933275\_A\_model-free\_6-DOF\_grasp\_detection\_method\_based\_on\_point\_clouds\_of\_local\_sphere\_area](https://www.researchgate.net/publication/369933275_A_model-free_6-DOF_grasp_detection_method_based_on_point_clouds_of_local_sphere_area)  
30. PU-Net: Point Cloud Upsampling Network | Papers With Code, 5월 8, 2025에 액세스, [https://paperswithcode.com/paper/pu-net-point-cloud-upsampling-network](https://paperswithcode.com/paper/pu-net-point-cloud-upsampling-network)  
31. PU-Net: Point Cloud Upsampling Network \- GitHub, 5월 8, 2025에 액세스, [https://github.com/yulequan/PU-Net](https://github.com/yulequan/PU-Net)  
32. PointCloudPapers/README.md at main \- GitHub, 5월 8, 2025에 액세스, [https://github.com/qinglew/PointCloudPapers/blob/main/README.md](https://github.com/qinglew/PointCloudPapers/blob/main/README.md)  
33. EasyRy/RepKPU: Point Cloud Upsampling with Kernel Point Representation and Deformation \- GitHub, 5월 8, 2025에 액세스, [https://github.com/EasyRy/RepKPU](https://github.com/EasyRy/RepKPU)  
34. An implementation of 3D Deep Learning and Traditional Computer Vision techniques to accurately upsample point clouds while being edge aware and respecting finer details. \- GitHub, 5월 8, 2025에 액세스, [https://github.com/An-u-rag/pointcloud-upsampling](https://github.com/An-u-rag/pointcloud-upsampling)  
35. lidq92/arxiv-daily: Automatically Update Interested Papers Daily using ... \- GitHub, 5월 8, 2025에 액세스, [https://github.com/lidq92/arxiv-daily](https://github.com/lidq92/arxiv-daily)  
36. (PDF) Lightweight Deep Learning Models For Edge Devices—A Survey \- ResearchGate, 5월 8, 2025에 액세스, [https://www.researchgate.net/publication/387782350\_Lightweight\_Deep\_Learning\_Models\_For\_Edge\_Devices-A\_Survey](https://www.researchgate.net/publication/387782350_Lightweight_Deep_Learning_Models_For_Edge_Devices-A_Survey)  
37. EdgeRegNet: Edge Feature-based Multimodal Registration Network between Images and LiDAR Point Clouds \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/html/2503.15284v1](https://arxiv.org/html/2503.15284v1)  
38. SPAC-Net: Rethinking Point Cloud Completion with Structural Prior \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/html/2411.15066v1](https://arxiv.org/html/2411.15066v1)  
39. Efficient End-to-End 6-Dof Grasp Detection Framework for Edge Devices with Hierarchical Heatmaps and Feature Propagation \- arXiv, 5월 8, 2025에 액세스, [https://arxiv.org/html/2410.22980v2](https://arxiv.org/html/2410.22980v2)  
40. NVIDIA Jetson AGX Orin Camera Module \- Supertek, 5월 8, 2025에 액세스, [https://www.supertekmodule.com/nvidia-jetson-agx-orin-camera-module/](https://www.supertekmodule.com/nvidia-jetson-agx-orin-camera-module/)