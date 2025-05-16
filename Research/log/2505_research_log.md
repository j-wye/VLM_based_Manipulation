#### 2025-05-07
- Perception Algorithm : NanoOWL + NanoSAM
- Grasping 관련 알고리즘 및 자료 조사 진행 중
    1. Depth Pointcloud가 sparse하니까 이를 Upsampling하는 방법
        - a
    2. Depth Pointcloud
        - a
    3. Depth Pointcloud + RGB Image
        - a

- DFU : Upsampling
- OGNI-DC : Upsampling
- Antipodal Grasp : Grasping concept
- GG-CNN : Depth image -> Grasp Map 생성
- Contact GraspNet : 3D pointcloud Grasp Map 생성
- VGN : 
- PointNetGPD : 3D pointcloud 

- Upsampling
    - DFU
    - OGNI-DC
- Depth Image
    - GG-CNN
    - GR-ConvNet
- Pointcloud
    - Contact-GraspNet
    - PointNetGPD
- 3D Volumetric (3D Voxel Grid)
    - VGN

- Idea Summary and Final Considerations
    ```
    ---
    A Study on the Optimization of a Real-Time Perception-Tracking-Segmentation Pipeline and intelligent Information Fusion Strategies for Robotic Grasping

    with NanoOwl, NanoSAM, Edge Device(Jetson Orin AGX 64GB)

    ---
    본 연구는 인간과 로봇 간의 직관적인 상호작용을 목표로, Jetson Orin AGX 64GB와 같은 edge device 환경에서 natural language command를 input으로 로봇 매니퓰레이터가 특정 객체를 정교하게 grasping하는 시스템 개발을 목표로 한한다. Background Research를 통해, 핵심 기술 요소로서 Open-Vocabulary based Object Detection Model인 NanoOwl과 Segmentation Model인 NanoSAM, 그리고 depth 정보를 활용하는 Approach의 Validity를 확인했다. 특히, robot arm의 dynamic한 움직임과 continuous한 vision input 변화에 대응하기 위해, 초기 자연어 이해 및 객체 특정 후에는 경량화된 시각 객체 추적기(Visual Object Tracker, VOT)를 통해 real-time으로 target tracking의 필요성을 느꼈다. 하지만 이를 NanoSAM에서 제공하는 Tracking 함수를 사용하여 진행하여 해결하기로 했다. 그리고 일정 시간마다 NanoOWL을 실행해 제대로 Tracking 하면서 Segmentation이 잘 되고 있는지 검증하는 과정을 진행한다. 이를 통해 세그멘테이션 마스크를 정교화하는 "인식 후 추적(Detect-then-Track)" 및 "필요 기반 정교화(Refine-on-Demand)" 아키텍처의 개념을 정립했다.

    이러한 기본 골격을 바탕으로, 실제 로봇 시스템의 강인함과 효율성을 극대화하기 위한 심층적인 연구가 필요합니다.

    첫째로, RGB-D 정보의 지능적 융합을 통한 그리핑 포즈 추정 방법론의 심층 분석이 중요합니다. Point cloud만을 사용하는 방식과 RGB 정보를 적극적으로 활용하는 방식(예: XYZRGB point cloud) 간의 장단점을 고려할 때, NanoOwl/NanoSAM 시스템의 특성상 RGB 정보를 활용하는 것이 유리하다는 잠정적 결론에 도달했습니다. 이에, GraspNet, PointNetGPD, 또는 기타 최신 학습 기반 그리핑 알고리즘들이 RGB-D 데이터를 어떻게 효과적으로 처리하여 6자유도 그리핑 포즈를 추정하는지, 그리고 이들이 Jetson Orin AGX 환경에서 어느 정도의 실시간성과 정확도를 보일 수 있는지에 대한 비교 연구가 필요합니다. 특히, 객체의 시각적 특징(색상, 질감)과 3차원 기하학적 특징(표면 법선, 곡률)을 어떤 방식으로 융합(feature fusion)해야 복잡한 형상이나 다양한 재질의 객체에 대해서도 안정적인 그리핑 품질을 확보할 수 있는지, 나아가 센서 노이즈나 모델 자체의 불확실성을 정량화하고 이를 그리핑 계획에 반영하여 강인함(robustness)을 높일 수 있는 최신 연구 동향 및 적용 가능성을 탐색하고자 합니다.

    둘째로, 제안된 전체 시스템 아키텍처의 엣지 디바이스(Jetson Orin AGX) 환경에서의 end-to-end 최적화 및 실증 방안에 대한 면밀한 검토가 필요합니다. ROS 2를 기반으로 각 모듈(자연어 이해, 초기 객체 탐지, 실시간 추적, 조건부 세그멘테이션, 그리핑 포즈 추정, 로봇 제어)을 노드화하여 구성할 때, 각 노드 간의 데이터 통신 효율성, 연산 부하 분배, 그리고 전체 파이프라인의 지연 시간(latency)을 최소화하기 위한 구체적인 소프트웨어 공학적 기법(예: TensorRT 최적화, 비동기 처리, 공유 메모리 활용)들을 조사하고 적용 방안을 모색해야 합니다. 또한, 다양한 실제 객체들과 조작 시나리오에 대한 시스템의 일반화 성능을 어떻게 평가하고 지속적으로 개선해나갈 수 있을지에 대한 방법론, 그리고 만약 학습 기반 모델을 적극적으로 활용한다면 Sim-to-Real 전이 학습 과정에서 발생할 수 있는 도메인 격차(domain gap) 문제들을 효과적으로 완화할 수 있는 전략에 대해서도 심도 있는 고찰이 요구됩니다. 이 과정은 연구과정에서 직접 진행할 파트는 아니고, 가능하면 이러한 부분을 고려해서 진행된 알고리즘을 선택해서 사용하기 위함이 가장 큰 목적이다.

    궁극적으로, 본 심층 연구를 통해 개발하고자 하는 지능형 로봇 그리핑 시스템이 실제 환경에서 사용자의 자연어 명령에 따라 다양한 객체를 실시간으로, 정확하고, 안정적으로 파지할 수 있도록 하는 데 필요한 핵심 기술적 난제들을 해결하고, 그 이론적 기반과 실용적 구현 방안을 제시하고자 합니다.

    ---

    추가적인 고려사항:

    Uncluttered, cluttered, complex scene 모두에서 unknown/unlabeled object에 대해서 manipulator arm을 통해 6DOF grasping SR(success rate)을 극대화하기 위한 연구로 생각하면 된다. 그리고 결국 input으로 사용할 데이터를 depth image로 할 것인지, depth pointcloud로 할 것인지, pointcloud + RGB Image를 사용할 것인지에 대한 설정부터 진행해줬으면 좋겠어. 그리고 추가적인 아이디어는 pointcloud를 사용할 때, pointcloud upsampling 알고리즘을 먼저 진행하고 grasping 알고리즘을 실행하는 것은 어떤지 궁금하다. sparse하다는 단점을 극복하기 위해 생각해봤다.

    내가 위와 같이 연구주제를 설정하고 연구를 진행하기 위해 자료조사를 진행하고 있는데, 너한테 요구하는 점은 알고리즘에 대한 자세한 조사를 요청한다. 특히 어떤 input데이터를 사용하는 알고리즘 방식인지, 어느정도의 real-time 성능이 나오는지 등 내 연구에 필요한 부분을 포함해서 같이 조사해줬으면 좋겠어. 지금은 grasping에 대해서 집중하고 있으니까 이 부분을 집중해서 2018년 이후부터 현재까지 가장 최신의 자료를 전부 찾아줬으면 좋겠어. 신뢰도 있는 정보 수집을 위해서 논문은 가능하면 top tier journal or conference를 선택해줬으면 좋겠어. 그리고 내가 사용하기 위함이니까 open source인지 여부가 중요해서 마지막에 github 주소를 작성해줘야해. 만약 조사했는데 open source가 아니라서 github주소가 없다면 해당 알고리즘은 필요없으니까 제거해.

    그리고 반드시 한국어로 완벽한 보고서를 작성해줬으면 좋겠어!
    ```

- For Future Work Algorithms
    - DexGraspNet/DexGraspNet2 : for dexterous grasping (with hand)