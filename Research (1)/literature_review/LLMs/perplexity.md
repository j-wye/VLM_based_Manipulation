# 로봇 그래스핑 기술 조사 연구 결과 보고서

최신 로봇 그래스핑 알고리즘에 대한 심층 조사 결과, RGB-D 카메라 데이터를 활용한 Edge Device 환경에서의 실시간 그래스핑 구현이 가능한 여러 접근법을 확인했습니다. 특히 PointNetGPD와 E3GNet 알고리즘의 조합이 Jetson Orin AGX 환경에서 NanoOwl/NanoSAM과 연계하여 높은 성능을 발휘할 수 있을 것으로 분석됩니다. 다양한 입력 데이터 유형별 알고리즘의 장단점과 성능을 비교했으며, 포인트 클라우드 업샘플링 기법 적용 시 그래스핑 성능이 향상될 가능성을 확인했습니다.

## 입력 데이터 유형별 알고리즘 조사

### RGB-D 기반 그래스핑 알고리즘

RGB-D 데이터를 입력으로 사용하는 주요 알고리즘으로는 E3GNet, GraspNet-baseline, FlexLoG 등이 있습니다. 이들은 색상 정보와 깊이 정보를 결합하여 객체의 시각적 특징과 3D 기하학적 특성을 모두 활용합니다.

E3GNet은 특히 엣지 디바이스를 위해 설계되었으며, 계층적 히트맵과 특징 전파 메커니즘을 통해 계산 효율성과 그래스핑 정확도를 모두 달성했습니다[3]. 이 알고리즘은 RGB-D 데이터를 입력으로 사용하여 6-DoF 그래스핑 포즈를 생성하며, 계층적 히트맵 표현과 특징 전파 기법을 활용하여 엣지 디바이스에서의 실시간 추론을 가능하게 합니다.

GraspNet-baseline은 GraspNet-1Billion 데이터셋에 최적화된 모델로, 190개의 복잡한 장면에서 캡처된 97,280개의 이미지와 1.1억 개 이상의 그래스핑 포즈 어노테이션을 활용합니다[6]. 이 알고리즘은 RGB-D 데이터를 처리하여 다양한 객체에 대한 6-DoF 그래스핑 감지를 지원합니다.

Lenz 등이 제안한 Deep Learning for Detecting Robotic Grasps 방법은 RGB-D 데이터를 활용하여 다단계 딥 네트워크 구조를 통해 그래스핑 감지 성능을 향상시켰습니다[1]. 이 접근법은 처음으로 딥러닝을 로봇 그래스핑에 적용한 초기 연구 중 하나입니다.

### 포인트 클라우드 기반 그래스핑 알고리즘

포인트 클라우드 기반 접근법으로는 PointNetGPD가 대표적입니다. 이 알고리즘은 그리퍼 내부의 3D 포인트 클라우드를 직접 처리하여 그래스핑 품질을 평가합니다[2]. 희소 포인트 클라우드에서도 우수한 성능을 보이며, YCB 객체 데이터셋으로 훈련되었습니다.

PointNetGPD는 포인트 클라우드에서 직접 로봇 그래스핑 구성을 감지하는 엔드투엔드 모델로, 복잡한 기하학적 구조를 효과적으로
포착할 수 있습니다[2]. 이 알고리즘은 포인트 클라우드 처리에 최적화되어 있어 NanoSAM의 세그멘테이션 결과와 직접 연동이 가능합니다.

IoT 기반 지능형 로봇 포인트 클라우드 그래스핑 연구에서는 PointNet 네트워크에 어텐션 메커니즘을 추가하여 그래스핑 품질을 향상시켰습니다[7]. 이 접근법은 객체의 로컬 포인트 클라우드에 더 집중하도록 네트워크를 설계하여 그래스핑 성공률을 높였습니다.

Ensemble of GNNs 접근법(GtG 2.0)은 그래프 신경망을 활용하여 복잡한 환경에서의 그래스핑 성능을 향상시켰습니다[18]. 이 모델은 실제 로봇 테스트에서 91%의 그래스핑 성공률을 기록했으며, 5개의 모델 앙상블을 통해 견고한 성능을 보였습니다[18].

### 깊이 이미지 기반 그래스핑 알고리즘

GQ-STN은 깊이 이미지만을 사용하여 92.4%의 정확도로 로버스트한 그래스핑을 감지할 수 있습니다[19]. 기존의 샘플링 기반 방식보다 60배 이상 빠르다는 장점이 있습니다.

Breyer 등의 연구에서는 실시간 6 DOF 그래스핑 감지 알고리즘을 제안하여 10ms만에 그래스핑 계획이 가능하며, 복잡한 환경에서 92%의 물체 제거 성공률을 달성했습니다[14].

GG-CNN(Generative Grasping Convolutional Neural Network)은 깊이 이미지의 모든 픽셀에 대해 그래스핑 품질, 각도, 그리퍼 너비를 직접 생성하는 생성형 접근법을 사용합니다[10]. 이 모델은 전체 그래스핑 파이프라인을 19ms 내에 처리할 수 있어 실시간 애플리케이션에 적합합니다[10].

Dusty-nv의 DepthNet은 Jetson 디바이스에서 실행 가능한 모노큘러 깊이 추정 모델로, 단일 색상 이미지에서 깊이 맵을 생성합니다[9]. 이 모델은 그래스핑 이전 단계에서 깊이 정보를 보강하는 데 활용될 수 있습니다.

### 포인트 클라우드 업샘플링 고려사항

포인트 클라우드의 희소성 문제를 해결하기 위한 업샘플링 기법이 그래스핑 성능 향상에 도움이 될 수 있습니다. PointNetGPD가 희소 포인트 클라우드에서도 잘 작동하지만, 추가적인 업샘플링 기법을 적용하면 더 나은 결과를 얻을 가능성이 있습니다.

RepKPU는 커널 포인트 표현과 변형을 통한 효율적인 포인트 클라우드 업샘플링 방법을 제안합니다[11]. 이 모델은 대규모 수용 필드와 위치 적응형 특성을 가진 표현을 활용하여 기하학적 패턴을 효과적으로 포착합니다[11].

EGP3D는 RGB-D 카메라를 위한 에지 가이드 기하학적 보존 3D 포인트 클라우드 초해상도 기법으로, 투영된 2D 공간에서의 에지 제약 조건을 통해 3D 포인트 클라우드의 품질을 향상시킵니다[12].

옥트리 기반 CNN은 패치 단위가 아닌 전체 포인트 클라우드를 한 번에 처리하여 기존 방법보다 47배 빠른 추론 속도를 제공합니다[13]. 이 접근법은 다양한 해상도의 포인트 클라우드를 하이퍼파라미터 조정 없이 처리할 수 있습니다[13].

## 그래스핑 알고리즘 성능 분석

### 실시간성 관점

E3GNet은 Jetson Xavier NX에서 126.7ms, Jetson TX2에서 157.9ms의 추론 시간을 달성했습니다. 이는 초당 약 6-8프레임의 처리가 가능한 수준으로, 실시간 그래스핑 애플리케이션에 적합합니다.

GG-CNN은 전체 그래스핑 파이프라인을 19ms 내에 처리할 수 있어 매우 빠른 추론 속도를 보여줍니다[10]. 이는 약 50fps에 해당하는 속도로, 실시간 시스템에 이상적입니다.

Breyer의 연구에서 제안된 알고리즘은 10ms만에 그래스핑 계획이 가능하여 매우 빠른 응답 시간을 제공합니다[14]. 이는 복잡한 환경에서도 실시간으로 그래스핑 결정을 내릴 수 있음을 의미합니다.

GQ-STN은 기존의 샘플링 기반 방식보다 60배 이상 빠르게 깊이 이미지에서 그래스핑을 감지할 수 있습니다[19]. 이러한 속도 향상은 실시간 로봇 시스템에서 중요한 장점입니다.

Multi-stage lightweight 6-DoF grasp pose 연구에서는 추론 시간이 약 40ms로, 대부분의 실시간 감지 시나리오 요구사항을 충족합니다[16].

### 6DOF 그래스핑 성공률

E3GNet은 실제 로봇 실험에서 94%의 그래스핑 성공률을 보였습니다[3]. 이는 복잡한 환경에서도 높은 신뢰성을 갖는 그래스핑이 가능함을 의미합니다.

FlexLoG/GtG 2.0은 95%의 성공률을 보고했습니다. 이 접근법은 다양한 객체와 환경에서 안정적인 성능을 보여줍니다.

PointNetGPD는 환경과 객체 세트에 따라 61%~89%의 성공률을 나타냈습니다. 이러한 변동성은 다양한 객체 유형과 장면 복잡성에 따른 성능 차이를 보여줍니다.

GtG 2.0은 실제 환경에서 91%의 성공률을 달성했으며, 5개의 테스트 장면에서 모든 장면을 성공적으로 처리했습니다[18]. 이는 높은 완료율과 함께 다양한 객체 처리 능력을 보여줍니다.

RSPFG-Net은 공개 데이터셋에서 97.85%의 평균 그래스핑 감지 정확도를 기록했으며, 실제 환경에서는 94.31%의 그래스핑 성공률을 달성했습니다[5]. 이는 multi-target 그래스핑 문제에서 매우 높은 성능을 보여줍니다.

### 엣지 디바이스 적합성

E3GNet은 엣지 디바이스에서 실시간 6-DoF 그래스핑 감지를 구현한 최초의 프레임워크입니다. Jetson 디바이스에서 우수한 성능을 보이며, 특히 Jetson Orin AGX와 같은 고성능 엣지 디바이스에서는 더 나은 성능이 기대됩니다.

PointNetGPD는 상대적으로 가벼운 네트워크 구조를 가져 엣지 디바이스에 적합합니다. 파라미터 수가 적어 메모리 사용량이 적고, 연산 효율성이 높습니다.

Jetson AGX Orin은 초당 200-275 TOPS(INT8)의 AI 성능을 제공하며, NVIDIA Ampere 아키텍처 GPU와 ARM Cortex CPU를 통합하여 고속 인터페이스와 빠른 메모리 대역폭을 제공합니다[15]. 이러한 하드웨어 성능은 복잡한 그래스핑 알고리즘도 실시간으로 실행할 수 있는 가능성을 제시합니다.

GR-ConvNet v2는 최소 기록 추론 시간인 20ms로 Cornell Grasp 데이터셋과 Jacquard 데이터셋에서 최첨단 정확도를 달성했습니다[17]. 이러한 성능은 엣지 디바이스에서의 효율적인 그래스핑에 적합합니다.

## 오픈소스 구현 가능성

다음 알고리즘들은 GitHub에 공개 저장소가 있습니다:

- PointNetGPD: https://github.com/lianghongzhuo/PointNetGPD[2]
  - MIT 라이센스로 제공되어 연구 및 상업적 목적으로 자유롭게 사용 가능
  - Python 환경에서 구현되어 있으며, PyTorch 기반 코드 제공
  - 설치 및 사용 방법이 문서화되어 있어 접근성이 좋음

- GraspNet-baseline: https://github.com/graspnet/graspnet-baseline[6]
  - GraspNet-1Billion 데이터셋과 함께 제공
  - 공식 API와 함께 다양한 그래스핑 알고리즘 평가 도구 제공
  - 벤치마크 평가 및 시각화 도구 포함

- LGD: https://github.com/Fsoft-AIC/LGD[20]
  - 자연어 기반 그래스핑 감지 구현
  - PyTorch 기반으로 구현되어 있으며, 다양한 데이터셋 지원
  - 학습 및 테스트 코드 제공

- LGD-MaskedGuideAttention: https://github.com/Fsoft-AIC/LGD-MaskedGuideAttention[22]
  - 마스크 가이드 어텐션을 활용한 언어 기반 그래스핑 감지
  - IROS 2024에 발표된 최신 연구 결과
  - 추론 예제 코드 및 체크포인트 제공

E3GNet과 FlexLoG/GtG 2.0의 공개 코드는 확인되지 않았습니다.

## 연구 주제에 가장 적합한 알고리즘 추천

### 추천 알고리즘: PointNetGPD + E3GNet 접근법 조합

#### 추천 근거:

1. **데이터 처리 적합성**:
   - PointNetGPD는 포인트 클라우드를 직접 처리하는 능력이 있어 NanoSAM의 세그멘테이션 결과와 직접 연동 가능합니다[2].
   - E3GNet의 계층적 히트맵 기반 접근법은 세그멘테이션 마스크와 효과적으로 결합될 수 있습니다[3].

2. **엣지 디바이스 최적화**:
   - E3GNet은 엣지 디바이스에서 실시간 성능을 입증했으며, Jetson Orin AGX에서 더 나은 성능을 기대할 수 있습니다[3].
   - PointNetGPD는 상대적으로 가벼운 네트워크 구조를 가져 엣지 디바이스에 적합합니다[2].

3. **오픈소스 구현 가능성**:
   - PointNetGPD는 공개 GitHub 저장소가 있어 연구 목적으로 바로 사용 가능합니다[2].
   - GraspNet-baseline의 구조를 참조하여 E3GNet과 유사한 최적화된 네트워크를 구현할 수 있습니다[6].

4. **실시간 인식-추적-세그멘테이션 파이프라인과의 통합**:
   - PointNetGPD의 그래스핑 품질 평가는 "추적 후 정교화" 전략과 잘 맞습니다.
   - E3GNet의 실시간 성능은 전체 파이프라인의 지연 시간을 최소화하는 데 도움이 됩니다[3].

5. **연구 확장성**:
   - 포인트 클라우드 업샘플링 알고리즘을 추가하여 PointNetGPD의 성능을 향상시킬 여지가 있습니다[11][12][13].
   - E3GNet의 계층적 히트맵 접근법을 NanoOwl/NanoSAM 시스템에 통합할 수 있는 가능성이 있습니다.

따라서, PointNetGPD의 그래스핑 품질 평가 메커니즘과 E3GNet의 엣지 디바이스 최적화 접근법을 결합하는 하이브리드 방식이 연구 목표인 "자연어 명령을 입력으로 로봇 매니퓰레이터가 특정 객체를 정교하게 그래스핑하는 시스템" 개발에 가장 적합할 것으로 판단됩니다.

또한, LGD와 같은 자연어 기반 그래스핑 알고리즘을 추가로 고려하여 자연어 명령을 그래스핑 포즈로 변환하는 능력을 향상시킬 수 있을 것입니다[20][22].

## 결론 및 제안

본 조사 연구를 통해 로봇 그래스핑 기술의 최신 동향과 성능을 분석했습니다. 특히 엣지 디바이스에서의 실시간 성능과 오픈소스 구현 가능성을 중점적으로 고려했습니다.

연구 목표인 NanoOwl과 NanoSAM을 활용한 자연어 기반 그래스핑 시스템 개발을 위해 다음과 같은 접근법을 제안합니다:

1. **입력 데이터 선택**: RGB-D 데이터를 기본 입력으로 사용하되, 포인트 클라우드 변환을 통해 PointNetGPD와 같은 알고리즘을 활용할 수 있습니다.

2. **포인트 클라우드 업샘플링 적용**: 희소 포인트 클라우드의 품질을 향상시키기 위해 RepKPU나 EGP3D와 같은
업샘플링 알고리즘을 전처리 단계로 적용하는 것이 효과적일 수 있습니다[11][12].

3. **하이브리드 그래스핑 접근법**: PointNetGPD의 그래스핑 품질 평가와 E3GNet의 계층적 히트맵 기반 접근법을 조합하여 정확도와 속도를 모두 확보할 수 있습니다.

4. **자연어 통합**: LGD와 같은 자연어 기반 그래스핑 알고리즘을 참고하여 자연어 명령을 효과적으로 처리하는 메커니즘을 개발할 수 있습니다[20][22].

5. **실시간 파이프라인 최적화**: GG-CNN이나 GQ-STN과 같은 매우 빠른 그래스핑 알고리즘의 최적화 전략을 참고하여 전체 시스템의 지연 시간을 최소화할 수 있습니다[10][19].

이러한 접근법을 통해 Jetson Orin AGX와 같은 엣지 디바이스에서도 효율적으로 동작하는 실시간 자연어 기반 로봇 그래스핑 시스템을 개발할 수 있을 것으로 기대됩니다.

Citations:
[1] https://www.roboticsproceedings.org/rss09/p12.pdf
[2] https://github.com/lianghongzhuo/PointNetGPD
[3] https://arxiv.org/html/2410.22980v3
[4] https://www.sciencedirect.com/science/article/abs/pii/S0031320324010690
[5] https://www.nature.com/articles/s41598-025-93490-8
[6] https://graspnet.net
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC9693196/
[8] https://arxiv.org/abs/2111.11114
[9] https://github.com/dusty-nv/jetson-inference/blob/master/docs/depthnet.md
[10] https://www.roboticsproceedings.org/rss14/p21.pdf
[11] https://openaccess.thecvf.com/content/CVPR2024/papers/Rong_RepKPU_Point_Cloud_Upsampling_with_Kernel_Point_Representation_and_Deformation_CVPR_2024_paper.pdf
[12] https://arxiv.org/abs/2412.11680
[13] https://arxiv.org/html/2410.17001v1
[14] https://jenjenchung.github.io/anthropomorphic/Papers/Breyer2020volumetric.pdf
[15] https://www.mdstech.co.kr/AGXOrin
[16] https://www.mdpi.com/2075-1702/12/8/506
[17] https://pmc.ncbi.nlm.nih.gov/articles/PMC9415764/
[18] https://arxiv.org/html/2505.02664v1
[19] https://arxiv.org/abs/1903.02489
[20] https://github.com/Fsoft-AIC/LGD
[21] https://opensource.stackexchange.com/questions/217/what-are-the-essential-differences-between-the-bsd-and-mit-licences
[22] https://github.com/Fsoft-AIC/LGD-MaskedGuideAttention
[23] https://www.frontiersin.org/articles/10.3389/fnbot.2023.1136882/full
[24] https://arxiv.org/abs/2306.07392
[25] https://www.sciencedirect.com/science/article/abs/pii/S0893608023006998
[26] https://3d.snu.ac.kr/assets/NFL/NFL_RAL_final.pdf
[27] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02933.pdf
[28] https://jeasinema.github.io/file/pointnetgpd_icra19.pdf
[29] https://arxiv.org/html/2403.06173v1
[30] https://arxiv.org/abs/1301.3592
[31] https://github.com/IRVLUTD/GraspTrajOpt
[32] https://arxiv.org/pdf/1802.08753.pdf
[33] https://arxiv.org/html/2403.05466v2
[34] https://paperswithcode.com/task/robotic-grasping/latest
[35] https://www.mdpi.com/2076-3417/12/15/7573
[36] https://github.com/atenpas/gpd
[37] https://www.sciencedirect.com/science/article/pii/S2667241322000052
[38] https://arxiv.org/pdf/2403.05466.pdf
[39] https://www.sciencedirect.com/science/article/pii/S009457652400211X
[40] https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1038658/full
[41] https://arxiv.org/html/2310.04349v2
[42] https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3c104b0e182a5f514d3aebecc93629bbcf1434ac
[43] https://www.ijcai.org/proceedings/2018/0677.pdf
[44] https://www.sciencedirect.com/science/article/abs/pii/S0957417422016736
[45] https://github.com/SamsungLabs/RGBD-FGN
[46] https://arxiv.org/abs/2410.22980
[47] https://paperswithcode.com/datasets?task=robotic-grasping&mod=rgb-d
[48] https://developer.nvidia.com/blog/generate-synthetic-data-for-deep-object-pose-estimation-training-with-nvidia-isaac-ros/
[49] https://www.mdpi.com/2079-9292/13/22/4432
[50] https://github.com/luyh20/FGC-GraspNet
[51] https://www.mdpi.com/2072-666X/13/2/293
[52] https://arxiv.org/html/2503.11163v1
[53] https://arxiv.org/html/2410.22980v2
[54] https://github.com/ivalab/KGN
[55] https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet
[56] https://github.com/GeorgeDu/vision-based-robotic-grasping
[57] https://github.com/rhett-chen/Robotic-grasping-papers
[58] https://github.com/unit-mesh/edge-infer
[59] https://github.com/graspnet/graspnet-baseline
[60] https://cw.fel.cvut.cz/wiki/_media/courses/hro/lectures/hro_8_lab_graspit_gpd.pdf
[61] https://arxiv.org/pdf/2102.00205.pdf
[62] https://www.roboticsproceedings.org/rss13/p58.pdf
[63] https://github.com/fangvv/EdgeKE
[64] https://berkeleyautomation.github.io/dex-net/
[65] https://mediatum.ub.tum.de/doc/1720412/d0rg0kp16l5shyo6ag683v2hi.lieGrasPFormer.pdf
[66] https://github.com/Savoie-Research-Group/EGAT
[67] https://github.com/graspnet/suctionnetAPI
[68] https://github.com/tnikolla/robot-grasp-detection
[69] https://arxiv.org/abs/2303.02133
[70] https://www.mdpi.com/2218-6581/8/3/63
[71] https://rllab.snu.ac.kr/publications/papers/2020_icra_gads.pdf
[72] https://ymxlzgy.com/publication/monograspnet/
[73] https://arxiv.org/abs/2107.05287
[74] https://dspace.mit.edu/bitstream/handle/1721.1/127544/1193031788-MIT.pdf?sequence=1&isAllowed=y
[75] https://arxiv.org/pdf/2110.10924.pdf
[76] https://www.bmvc2021-virtualconference.com/assets/papers/0692.pdf
[77] https://www.mdpi.com/1424-8220/24/13/4205
[78] https://arxiv.org/pdf/1905.13675.pdf
[79] https://jkros.org/_EP/view/?aidx=29292&bidx=2855
[80] https://arxiv.org/html/2501.07076v2
[81] https://www.mdpi.com/1999-4893/15/4/124
[82] http://arxiv.org/pdf/1809.06267.pdf
[83] https://arxiv.org/abs/2306.01081
[84] https://www.sciencedirect.com/science/article/abs/pii/S0957417423033833
[85] https://www.mdpi.com/2079-9292/13/13/2521/pdf
[86] https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Point_Cloud_Upsampling_via_Disentangled_Refinement_CVPR_2021_paper.pdf
[87] https://lava.kaist.ac.kr/wp-content/uploads/2024/11/ELMO-Enhanced-Real-time-LiDAR-Motion-Capture-through-Upsampling.pdf
[88] https://github.com/GeorgeDu/vision-based-robotic-grasping/blob/master/README.md
[89] https://paperswithcode.com/task/point-cloud-super-resolution/codeless
[90] https://www.sshowbiz.xyz/f4ee0987-97fe-40c8-9b13-a1b0778b6bab
[91] https://openaccess.thecvf.com/content/ACCV2020/papers/Son_SAUM_Symmetry-Aware_Upsampling_Module_for_Consistent_Point_Cloud_Completion_ACCV_2020_paper.pdf
[92] https://www.mdpi.com/2072-4292/13/13/2526
[93] https://www.mdpi.com/1099-4300/26/12/1022
[94] https://www.sciencedirect.com/science/article/abs/pii/S0168169922005671
[95] https://arxiv.org/html/2503.20820v1
[96] https://arxiv.org/html/2410.22980v1
[97] https://elib.dlr.de/146134/1/ainetter2021GraspCNN.pdf
[98] https://www.aimodels.fyi/papers/arxiv/efficient-end-to-end-6-dof-grasp
[99] https://www.semanticscholar.org/paper/94462768e14fc12dac55d988fd5194ea591fac3a
[100] https://antmicro.com/blog/2022/12/benchmarking-dnn-on-nvidia-jetson-agx-orin-with-kenning/
[101] https://research.rug.nl/files/829739165/Instance-wise_Grasp_Synthesis_for_Robotic_Grasping.pdf
[102] https://paperswithcode.com/task/robotic-grasping
[103] https://www.semanticscholar.org/paper/Generalizing-6-DoF-Grasp-Detection-via-Domain-Prior-Ma-Shi/1ca1e315a7fdda632ddafc8303863ee925e65a59
[104] https://forums.developer.nvidia.com/t/object-detection-networks-that-can-run-100-on-dla-orin-agx-with-performance-of-100fps/270104
[105] https://www.mdpi.com/2076-0825/14/1/25
[106] https://arxiv.org/html/2402.01303v2
[107] https://developer.nvidia.com/blog/new-nvidia-research-helps-robots-improve-their-grasp/
[108] https://ar5iv.labs.arxiv.org/html/1412.3128
[109] https://openreview.net/forum?id=clqzoCrulY&noteId=clqzoCrulY
[110] https://arxiv.org/html/2403.15054v1
[111] https://qmro.qmul.ac.uk/xmlui/bitstream/123456789/90711/2/Jamone%20Statistical%20Stratification%20and%20Benchmarking%20of%20Robotic%20Grasping%20Performance%202023%20Accepted.pdf
[112] https://arxiv.org/pdf/1412.3128.pdf
[113] https://openaccess.thecvf.com/content/CVPR2024/papers/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.pdf
[114] https://arxiv.org/html/2503.20820v2
[115] https://me336.ancorasir.com/wp-content/uploads/2024/03/Team6-1stPaperReview-Report.pdf
[116] https://arxiv.org/pdf/1301.3592.pdf
[117] https://github.com/ultralytics/ultralytics/issues/8280
[118] https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf
[119] https://arxiv.org/html/2407.01209v2
[120] https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Generalizing_6-DoF_Grasp_Detection_via_Domain_Prior_Knowledge_CVPR_2024_paper.pdf
[121] https://arxiv.org/html/2407.13842v1
[122] https://proceedings.mlr.press/v205/ma23a/ma23a.pdf
[123] https://www.themoonlight.io/de/review/efficient-end-to-end-6-dof-grasp-detection-framework-for-edge-devices-with-hierarchical-heatmaps-and-feature-propagation
[124] http://ras.papercept.net/images/temp/IROS/files/2807.pdf
[125] https://2023.ictc.org/media?key=site%2Fictc2023a%2Fabs%2FP3-24.pdf
[126] https://www.roboticsproceedings.org/rss17/p024.pdf
[127] https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-062122-025215
[128] https://motion.cs.illinois.edu/papers/RSS2017Workshop-Zhou-6DOFGraspPlanning.pdf
[129] https://www.sciencedirect.com/science/article/abs/pii/S0736584522000588
[130] https://category.yahboom.net/products/dofbot-jetson_nano
[131] https://docs.nvidia.com/jetson/archives/r35.2.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
[132] https://d-scholarship.pitt.edu/46386/1/Qi_Yin_Thesis.pdf
[133] https://arxiv.org/abs/2403.15054
[134] https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/
[135] https://www.sysgen.de/media/pdf/15/c8/f8/sysgen_nvidia-jetson-agx-orin-technical-brief_v1.pdf
[136] https://www.mdpi.com/1424-8220/23/14/6548
[137] https://developer.nvidia.com/embedded/jetson-benchmarks
[138] https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/synthetica_detr
[139] https://h2t.iar.kit.edu/pdf/BaekPohl2022.pdf
[140] https://m.coupang.com/vm/products/8501632745?itemId=24610809884&vendorItemId=91630954146
[141] https://www.sciencedirect.com/science/article/pii/S2665917423003495
[142] https://partners.amazonaws.com/ko/devices/a3G8W000000800IUAQ/reComputer%20J4012-Edge%20AI%20Device%20with%20Jetson%20Orin%20NX%2016GB%20Module
[143] https://docs.ultralytics.com/integrations/tensorrt/
[144] https://developer.nvidia.com/blog/create-design-and-deploy-robotics-applications-using-new-nvidia-isaac-foundation-models-and-workflows/
[145] https://www.intermodalics.ai/blog/nvidia-isaac-ros-in-under-5-minutes
[146] https://github.com/trending
[147] https://github.com/explore
[148] https://github.com/topics/open-source
[149] https://www.atlassian.com/git/tutorials/setting-up-a-repository
[150] https://www.youtube.com/watch?v=gp3D4mf6MHQ
[151] https://www.dataschool.io/how-to-contribute-on-github/
[152] https://github.com/Hymwgk/PointNetGPD
[153] https://github.com/OpenSees/OpenSees
[154] https://tev-fbk.github.io/FreeGrasp/
[155] https://www.coursera.org/articles/what-is-git
[156] https://paperswithcode.com/paper/graspnet-1billion-a-large-scale-benchmark-for
[157] https://lianghongzhuo.github.io/PointNetGPD/
[158] https://github.com/scikit-learn/scikit-learn
[159] https://github.com/IRVLUTD/isaac_sim_grasping
[160] https://www.reddit.com/r/StallmanWasRight/comments/t7vprq/are_mit_and_bsdlicensed_software_the_plague/
[161] https://klarasystems.com/community/licensing/
[162] https://sktelecom.github.io/guide/use/obligation/
[163] https://softwareengineering.stackexchange.com/questions/278126/is-it-legal-to-include-bsd-licensed-code-within-an-mit-licensed-project
[164] https://www.gnu.org/licenses/agpl-3.0.html
[165] https://github.com/open-telemetry/opentelemetry-js/pull/33
[166] https://arxiv.org/html/2409.14403v1
[167] https://en.wikipedia.org/wiki/BSD_licenses
[168] https://github.com/graspnet/graspnet-baseline/blob/main/LICENSE
[169] https://github.com/zhaobinglei/REGNet_for_3D_Grasping
[170] https://github.com/purvaten/FLEX
[171] https://fossa.com/blog/open-source-software-licenses-101-bsd-3-clause-license/

---
Perplexity로부터의 답변: pplx.ai/share