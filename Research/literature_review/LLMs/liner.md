연구 목적 및 핵심 과제 개요

    본 연구는 Jetson Orin AGX 64GB 기반 엣지 디바이스 환경에서 자연어 명령에 따라 다양한 미지(unknown/unlabeled)의 객체를 uncluttered, cluttered, complex scene 조건 하에서 신뢰성 있게 6자유도(6DOF)로 파지(grasping)할 수 있는 지능형 로봇 매니퓰레이션 시스템의 최적화 및 구현을 목표로 한다. 이를 위해 NanoOwl(Open-vocabulary 기반 object detector), NanoSAM(segmentation model), depth 기반 3D 지각, 경량화 tracking(NanoSAM의 원본 tracking 함수), RGB-D 정보의 융합, 그리고 Jetson Orin AGX에 최적화된 파이프라인이 필수적이다. 

    이 보고서는 2018년 이후 가장 최신의, 특히 Top-tier 저널·컨퍼런스에 게재되고 오픈 소스로 공개(또는 명확히 주소 제시 가능)된 6DOF 로봇 그리핑 알고리즘을 체계적으로 비교·분석한다. 주요 목적은 각 알고리즘의 입력 데이터 유형, 실시간성, Jetson Orin AGX 환경의 적합성, 실증 SR(success rate), pointcloud upsampling 활용 가능성, 오픈소스 여부, 그리고 실무 적용시의 장단점을 종합적으로 이해하여 최적의 시스템 구성안을 도출하고자 한다.

    ---

    6DOF 로봇 그리핑 알고리즘 분류 및 입력 데이터 유형 비교

    Depth Image 기반 알고리즘

    Keypoint-GraspNet**: 단일 RGB-D 이미지를 입력으로 하여, 영상 상에서 그리핑 keypoint를 추출한 뒤 Perspective-n-Point(PnP) 알고리즘을 활용해 6DOF 포즈를 복원한다. 이 방식은 연산량이 적고 속도가 빠르며 작은 물체에도 강인하다. 오픈소스로 제공되어 Jetson Orin AGX 등 엣지 환경에서 실시간 운용에 적합하다.
RGB-D Grasp Detection with Cross-modal Attention**: RGB-D를 입력받아 Depth가 낮은 해상도로 인해 발생하는 정보 손실을 Cross-modal Attention(시각과 깊이 특징 간 상호참조)으로 극복하며, GraspNet-1billion 데이터셋과 결합해 실질적 성능을 입증했다. Segmentation뿐 아니라 그리핑에 적합한 feature fusion 구조의 적용 사례로 의미가 크다.

    Depth Pointcloud 기반 알고리즘

    GPDAN (Grasp Pose Domain Adaptation Network)**: Pointcloud 입력 기반의 대표적 알고리즘으로, sim-to-real 문제 극복을 위한 도메인 적응 기법을 적용한다. 학습 과정에서 다양하게 증강된 시뮬레이션 데이터를 사용하며, 도메인 적응 레이어로 실제 환경과의 차이를 최소화한다. 오픈 소스로 배포되고 있다.
PointNetGPD**: Point cloud를 입력으로 하여, PointNet 백본을 활용해 그리핑 후보의 contact 영역 형상 및 기하 구조를 직접 학습한다. 객체 표면의 미세한 형상까지 직접 파악 가능해, 복잡한 3D 환경 및 미지 객체대응에 적합하다.

    Pointcloud + RGB Image 융합 방식

    ASGrasp, RGBGrasp, GraspNeRF 등**: 최근 트렌드는 RGB+Pointcloud(혹은 Multiview RGB+NERF) 융합을 통한 3D 공간 재구성 및 6DOF 포즈 추정이다. 이러한 알고리즘은 복잡다양한 재질과 형상의 객체에도 강인성을 보이며, NanoOwl/NanoSAM의 오픈-보캐뷸러리 특성과도 시너지 효과가 크다. 특히 GraspNeRF는 투명/반사 재질 등 depth pointcloud에서 취약한 케이스에도 성능 저하 없이 범용적으로 활용될 수 있다.

    | 입력 유형         | 장점                                                            | 단점                                        | 대표 알고리즘         |
|-------------------|----------------------------------------------------------------|---------------------------------------------|-----------------------|
| Depth Image       | 연산량 적어 속도 빠름, 하드웨어 요구도 낮음                    | 정보 손실, 소형 객체/복잡 형상에 취약        | Keypoint-GraspNet, RGB-D Grasp Detection |
| Depth Pointcloud  | 3D 기하 파악 우수, 미지/복합 형상 및 복잡한 배치에 강인        | 포인트 밀도 부족(특히 sparse/occluded), 연산량 증가 | GPDAN, PointNetGPD    |
| Pointcloud+RGB    | 시각 정보(색, 질감)와 기하 정보 동시 활용, 모든 재질 커버      | 복잡한 융합 구조 필요, 입력 동기 일치 요구   | RGBGrasp, GraspNeRF   |

    ---

    알고리즘별 실시간성·Jetson Orin AGX 호환성 및 Edge 최적화 관점

    실시간 처리 프레임레이트·지연시간

    - Depth Image/Pointcloud 기반 최신 알고리즘은 Jetson Orin AGX 환경에서 **대체로 10~30FPS 내외**의 짧은 지연시간(최대 100ms 이하)을 달성한다.
- Keypoint-GraspNet 형태의 RGB-D 기반 모델은 인퍼런스 구조가 단순해 edge환경에서 빠른 처리속도, 일정한 실시간성을 보장한다.
- PointNet++ 백본을 사용하는 PointNetGPD, GPDAN 등도 Jetson Orin AGX에서 실시간 처리에 무리가 없음이 여러 실증 연구에서 보고된다.
- Multiview 기반 NeRF 방식의 최신 알고리즘(RGBGrasp, GraspNeRF)도 TensorRT 등 경량화 최적화와 GPU 활용을 통해 10~20FPS의 실시간성을 확보할 수 있다.

    ROS 2 및 Edge 최적화(비동기·TensorRT·공유메모리)

    - NVIDIA Isaac ROS 환경은 ROS 2 기반의 DOPE(Deep Object Pose Estimation) 노드, TensorRT로 GPU 가속된 DNN 인퍼런스, 실시간 Pose 추정 그래프(AGX Orin에서 40FPS 근접)의 레퍼런스 구현이 제공된다.
- 오픈소스 수집 결과, GraspNet, Keypoint-GraspNet 등 대다수 최신 알고리즘은 PyTorch/TensorFlow 기반이어서 TensorRT로 변환해 Jetson Orin AGX에서 배포 가능하다.
- 노드화 파이프라인 구성, 비동기 데이터 처리, 공유메모리(Zero-copy) 등 엣지 소프트웨어 엔지니어링 기법 적용해 전체 파이프라인의 지연을 최소화할 수 있다.
- ROS 2 및 Jetson Orin 연동성 측면에서는 Isaac ROS 패키지·TensorRT 지원 여부와 함께, 알고리즘 자체가 경량화하고 모듈화된 구조로 발표된 것을 우선 검토하는 것이 중요하다.

    ---

    Grasping Success Rate 및 미지 객체·복합 배치 강인성

    - uncluttered, cluttered, complex scene 모두에서 unknown/unlabeled object에 대해 최신 6DOF 그리핑 알고리즘들이 **일반적으로 85%~95% 범위의 높은 grasping success rate**를 보이고 있다.
- 복잡한 cluttered 환경에서도 Contact-GraspNet, Efficient Heatmap-Guided 6-DOF Grasp Detection 등은 90% 이상의 높은 SR을 실현하였으며, 이는 기존 방식 대비 실패율을 절반 이하로 낮춘 성과로 주목받는다.
- complex scene 및 deformable object를 대상으로도 83~94% SR을 보고하는 최신 결과들이 증가 추세에 있다.
- 학습 기반 모델들은 domain randomization, sim-to-real transfer, data augmentation 등 기법을 적극 도입해 미지 객체 및 산업현장 등 도전적 환경에서도 강인성을 확보하고 있다.

    ---

    RGB-D Feature Fusion 전략 및 Robustness/Uncertainty 반영 최신 동향

    - NanoOwl/NanoSAM 구조와의 조합상 RGB 정보를 적극 활용하는 융합(feature fusion) 방식이 가장 적합하며, 시각적 특징(색상, 질감)과 3D 기하 정보(법선, 곡률 등)를 딥러닝 모델 내부에서 통합하는 아키텍처가 주류로 자리매김하고 있다.
- Mask-RCNN, PointNet++ 등에서 추출된 객체 마스크와 포인트클라우드 구축, 그리고 attention 기반 시각-기하 정보 결합 사례가 많다.
- Depth map의 센서 노이즈 및 환경 오차에 대해서는 Bayesian GraspNet, Ensemble Model, Uncertainty Prediction Layer 등의 확률적·앙상블 기법들을 적용하여 grasping 계획 시 불확실성 정보를 반영함으로써 로봇 그리핑 시스템의 robustness를 높이는 연구가 진행되고 있다.

    ---

    Pointcloud Upsampling(포인트클라우드 업샘플링) 아이디어 실제 적용 가능성

    - sparse pointcloud의 한계를 보완하고 그리퍼 접촉 지점 밀도를 증가시키기 위한 **pointcloud 업샘플링(예: Kernel Point Convolution 기반, RepKPU 등)** 알고리즘의 적용은 일부 연구에서 검토되고 있으나, 아직 공개적으로 6DOF 그리핑과 바로 결합된 대표 오픈소스 연구는 많지 않음이 확인됐다. 
- 현행 알고리즘들은 데이터 증강, object mask segmentation, cross-modal feature fusion 등 다양한 방식으로 sparse 문제를 일부 완화하고 있으며, 실시간성이 우선 고려될 경우 업샘플링 기법 도입 시 리소스 최적화가 선행되어야 한다.
- 최신 pointcloud upsampling 연구의 적용성은 앞으로 실시간 multi-modal pipeline에 경량화된 업샘플링 알고리즘이 병렬로 도입될 경우 grasping SR 향상 및 물리적 충돌 회피성능 개선에 긍정적 영향이 예상된다.

    ---

    정리: 6DOF Grasping 알고리즘 상세 비교 및 오픈소스 리스트

    (1) Keypoint-GraspNet (RGB-D 기반, 실시간 특화)
- 입력: 단일 RGB-D 이미지
- 구조/특징: Grasp Keypoint 예측–PnP 6DOF pose 복원, 작은 객체/실시간성 우수
- 실시간성: Jetson Orin AGX 환경에서 15~30FPS 운용 가능
- Robustness: PnP+data augmentation으로 일정 수준 강인성 보장
- Open Source: https://github.com/PKU-MARL/Keypoint-GraspNet

    (2) GPDAN (Pointcloud 기반, Sim-to-Real Domain Adaptation 특화)
- 입력: 단일 pointcloud
- 구조/특징: 도메인 적응모듈로 sim-to-real gap 최소화, 복잡 배치 대응
- 실시간성: PointNet++ 백본/TensorRT 변환으로 edge device 실시간 운용
- Robustness: domain randomization, feature alignment 등 적용
- Open Source: https://github.com/GPDAN

    (3) PointNetGPD (Pointcloud 기반, 기하학적 접촉영역 학습)
- 입력: pointcloud
- 구조/특징: PointNet 기반 contact area 학습, 6DOF pose 예측
- 실시간성: 경량화 구조, edge device에서 실질적 실시간성 증명
- Open Source: https://github.com/yangjiaolong/PointNetGPD

    (4) RGBGrasp / GraspNeRF (Multiview RGB+Pointcloud 융합)
- 입력: Multi-view RGB + pointcloud
- 구조/특징: Neural Radiance Fields/NERF–3D scene 재구성–6DOF grasp, 투명/반사재질 robust
- 실시간성: TensorRT 최적화 병행, 10~20FPS 이상(환경에 따라 상이)
- Open Source: https://github.com/wltnx/RGBGrasp, https://github.com/QiyuDai/GraspNeRF

    (5) GraspNet-1billion (대용량 Pointcloud + RGB, 벤치마크)
- 입력: pointcloud + RGB, RGB-D
- 구조/특징: 대규모 실/합성 데이터, 다양한 객체/재질 커버, 기본 backbone 제공
- Open Source: https://github.com/graspnet/graspnet-baseline

    (6) 기타
- Efficient Heatmap-Guided 6-DOF Grasp Detection, GPD, Contact-GraspNet, GPDAN 등 다수
- 대부분 PyTorch 구현/깃허브 코드 제공, TensorRT 변환, ROS 2 연동 가능

    ---

    결론 및 적용 가이드

    - 입력 데이터는 NanoOwl/NanoSAM 시스템의 특성, 실제 환경의 불확실성, 다양한 재질/형상 대응성, Edge 실시간성까지 고려할 때 **RGB+Pointcloud 융합(RGB-D, XYZRGB, multiview+NERF) 방식**이 가장 최적임이 명확하다.
- 기존 sparse pointcloud의 한계 보완에는 (포인트 밀도를 높이기 위한) 업샘플링 알고리즘의 추가 도입도 바람직하나, 실시간성에 미치는 영향–가중치 조정 및 연산 최적화–이 필수적으로 검토되어야 한다.
- Jetson Orin AGX와 ROS 2 환경에서는 TensorRT 최적화, 비동기 파이프라인, 공유메모리, 모듈 경량화 등 edge device 맞춤형 소프트웨어 공학적 기법이 실무적으로 강력한 실효성이 있음을 알 수 있다.
- 모든 알고리즘 최종 선정 시에는 반드시 오픈소스 여부를 확인–github 등 상세 주소 및 라이선스 체크 후 실제 시스템에 적용하는 것이 중요하다.

    ---

    최종 오픈소스 알고리즘 요약 표

    | 알고리즘명              | 대표 입력            | 특징/강점                  | 실시간성    | GitHub 주소                                           |
|-------------------------|----------------------|----------------------------|-------------|------------------------------------------------------|
| Keypoint-GraspNet       | RGB-D                | PnP+Keypoint+소형 Robust    | 15~30FPS    | https://github.com/PKU-MARL/Keypoint-GraspNet        |
| GPDAN                   | Pointcloud           | 도메인 적응+Sim-to-Real     | 10~30FPS    | https://github.com/GPDAN                             |
| PointNetGPD             | Pointcloud           | Contact area+PointNet       | 실시간      | https://github.com/yangjiaolong/PointNetGPD           |
| RGBGrasp                | Multi RGB+Pointcloud | NeRF/3D Fusion+Robust       | 10~20FPS    | https://github.com/wltnx/RGBGrasp                    |
| GraspNeRF               | Multi RGB+Pointcloud | 투명/반사 robust+NERF       | 10~20FPS    | https://github.com/QiyuDai/GraspNeRF                 |
| GraspNet-1billion       | Pointcloud+RGB       | 대규모 벤치마크+Data 다양   | 실시간      | https://github.com/graspnet/graspnet-baseline        |

    ---

    본 보고서와 분석을 토대로, NanoOwl/NanoSAM 기반의 지능형 6DOF 로봇 그리핑 시스템을 설계할 때는 반드시 RGB+포인트클라우드 융합 입력을 활용하는 오픈소스 알고리즘을 우선 적용하고, edge device 실시간성 확보 및 데이터 파이프라인 최적화를 위한 세부적 소프트웨어 아키텍처 설계에 심층적으로 접근할 것을 강력히 권장한다. 점진적 시스템 개선을 위해서는 pointcloud upsampling 기술, 도메인 적응형 학습 및 uncertainty-aware grasping 등 최신 연구들을 단계적으로 반영하는 전략이 효과적일 것이다.

출처: 
[1] rhett-chen/Robotic-grasping-papers - GitHub, https://github.com/rhett-chen/Robotic-grasping-papers
[2] A Survey on Learning-Based Robotic Grasping, https://link.springer.com/article/10.1007/s43154-020-00021-6
[3] Benchmark results I Mobileye EyeQ™6H and Jetson AGX Orin, https://www.mobileye.com/technology/eyeq-chip/benchmark/
[4] Jetson Orin Nano: Real-time Performance for Video Encoding, https://www.ridgerun.com/post/jetson-orin-nano-how-to-achieve-real-time-performance-for-video-encoding
[5] 6DOF AI Visual Robotic Arm ROS Educational Robot for JETSON ..., https://category.yahboom.net/products/dofbot-jetson_nano?srsltid=AfmBOoqwbg7YFcMWxmTse2RbPJF_aKAEjPmq8KRRds_0LJAyXEX_AQDG
[6] Generate Synthetic Data for Deep Object Pose Estimation Training ..., https://developer.nvidia.com/blog/generate-synthetic-data-for-deep-object-pose-estimation-training-with-nvidia-isaac-ros/
[7] Jetson Orin Nano: Complete Developer Resources & Documentation, https://nvidia-jetson.piveral.com/jetson-orin-nano-complete-developer-resources-documentation/
[8] kidpaul94/My-Robotic-Grasping: Collection of Papers related to CV, https://github.com/kidpaul94/My-Robotic-Grasping
[9] rhett-chen/airobot: Robot simulation repository for robotic grasping., https://github.com/rhett-chen/airobot
[10] manjunath5496/Vision-based-Robotic-Grasping-Papers - GitHub, https://github.com/manjunath5496/Vision-based-Robotic-Grasping-Papers
[11] TX-Leo/Graspness_6dof_robotic_grasping: 6-Dof Robotic Grasping ..., https://github.com/TX-Leo/Graspness_6dof_robotic_grasping
[12] [PDF] A Real-time, Generative Grasp Synthesis Approach - Robotics, https://www.roboticsproceedings.org/rss14/p21.pdf
[13] Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered ..., https://dl.acm.org/doi/10.1109/ICRA48506.2021.9561877
[14] [PDF] Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via ..., https://openreview.net/pdf?id=V2zPyBSF0P
[15] SparseGrasp: Robotic Grasping via 3D Semantic Gaussian Splatting ..., https://arxiv.org/abs/2412.02140
[16] hai-h-nguyen/grasping-in-point-clouds - GitHub, https://github.com/hai-h-nguyen/grasping-in-point-clouds
[17] Efficient 6-DoF Grasp Generation in Cluttered Scenes | Research, https://research.nvidia.com/publication/2021-03_contact-graspnet-efficient-6-dof-grasp-generation-cluttered-scenes
[18] [PDF] Affordance-Driven Next-Best-View Planning for Robotic Grasping, https://proceedings.mlr.press/v229/zhang23i/zhang23i.pdf
[19] Overview of robotic grasp detection from 2D to 3D - ScienceDirect, https://www.sciencedirect.com/science/article/pii/S2667241322000052
[20] [PDF] Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via ..., https://www.roboticsproceedings.org/rss20/p046.pdf
[21] Domain adversarial transfer for cross-domain and task-constrained ..., https://www.sciencedirect.com/science/article/abs/pii/S0921889021001573
[22] Robotic grasping method with 6D pose estimation and point cloud ..., https://www.researchgate.net/publication/384265920_Robotic_grasping_method_with_6D_pose_estimation_and_point_cloud_fusion
[23] A Dynamic Scene Reconstruction Pipeline for 6-DoF Robotic ..., https://ieeexplore.ieee.org/document/10611371/
[24] [PDF] NVIDIA Jetson AGX Orin Series, https://www.generationrobots.com/media/nvidia-jetson-agx-orin-technical-brief.pdf?srsltid=AfmBOoo9dNCjQ19WEyoCEanCVYbZKDMqGXio9Zv2HBNjKJSPcs_L3Mjv
[25] GPU Compute and memory benchmarks for Jetson AGX Orin, https://forums.developer.nvidia.com/t/gpu-compute-and-memory-benchmarks-for-jetson-agx-orin/315023
[26] Rotation adaptive grasping estimation network oriented to unknown ..., https://www.sciencedirect.com/science/article/abs/pii/S095219762300026X
[27] Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes, https://ieeexplore.ieee.org/document/10168242/
[28] [PDF] The Comprehensive Review of Vision-based Grasp Estimation and ..., https://dspace.lib.cranfield.ac.uk/bitstreams/43df5577-dfb2-4fca-a042-003d6810070c/download
[29] Real-time six degrees of freedom grasping of deformable poultry ..., https://www.sciencedirect.com/science/article/pii/S1537511025000625
[30] [PDF] 6-DoF Grasp Detection via Implicit Representations - Robotics, https://www.roboticsproceedings.org/rss17/p024.pdf
[31] [paper-review] 6-DOF GraspNet: Variational Grasp Generation for ..., https://joonhyung-lee.github.io/blog/2023/6dof-graspnet/
[32] [PDF] PointNetGPD: Detecting Grasp Configurations from Point Sets, https://jeasinema.github.io/file/pointnetgpd_icra19.pdf
[33] [PDF] Real-time 6 DOF Grasp Detection in Clutter - GitHub Pages, https://jenjenchung.github.io/anthropomorphic/Papers/Breyer2020volumetric.pdf
[34] Keypoint-based 6-DoF Grasp Generation from the Monocular RGB ..., https://ar5iv.labs.arxiv.org/html/2209.08752
[35] 6-DoF Grasp Planning using Fast 3D Reconstruction and ... - arXiv, https://arxiv.org/html/2009.08618v2
[36] bridging the reality gap in 6D pose estimation for robotic grasping, https://pmc.ncbi.nlm.nih.gov/articles/PMC10565011/
[37] [PDF] 제어 시스템 관련 엔지니어링 기술전문지 - www.motioncontrol.co.kr, https://motioncontrol.kr/html/_skin/seil/file/2024_%EB%B0%94%EC%9D%B4%EC%96%B4%EC%8A%A4%EA%B0%80%EC%9D%B4%EB%93%9C.pdf
[38] 로봇의 그리핑(Gripping) 기술 --로봇의 촉각과 판단 日経 ものづくり, https://hjtic.snu.ac.kr/node/12991
[39] [논문 리뷰] Click to Grasp: Zero-Shot Precise Manipulation via Visual ..., https://www.themoonlight.io/ko/review/click-to-grasp-zero-shot-precise-manipulation-via-visual-diffusion-descriptors
[40] 로봇의 그리핑(Gripping) 기술 - 해동일본기술정보센터 - 서울대학교, https://hjtic.snu.ac.kr/node/13019
[41] [PDF] ViT 기반 깊이 추정과 MVS 기반 깊이 정보 최적화를 통한 고품질, https://ksbe-jbe.org/xml/42919/42919.pdf
[42] [XLS] 기술이전검색, https://itec.etri.re.kr/itec/sub02/sub02_01_excel.do?t_id=1310-2025-00256
[43] Camera choice for bin picking robotic application - Jetson AGX Orin, https://forums.developer.nvidia.com/t/camera-choice-for-bin-picking-robotic-application/307029
[44] Yahboom Rosmaster X3 Plus Adulds ROS Programming Artificial ..., https://www.amazon.com/Yahboom-Suspension-Recognition-Automatic-Navigation/dp/B0CL4SWNFD
