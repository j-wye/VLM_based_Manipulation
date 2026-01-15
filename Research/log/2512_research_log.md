#### 2025-11-30
- 기존까지의 진행 과정 : 각각을 벤치마킹하는 코드까지 완성했었음
- Integrated Perception Module class 작성 완료
- GPU to CPU로 데이터가 움직이는 **H2D overhead**를 minimize하도록 극한의 optimization 진행 완료

#### 2025-12-01 ~ 2025-12-04
- Decision Module Specification
    - input : RGB, Depth, Bbox, Seg mask
    - Crop(ROI Slicing) the RGB and Depth with bbox
    - Depth Enhancement with cropped image (Additional plug-and-play module)
    - Topology-Aware Denoising
    - Adaptive Z-Filtering
    - Back-Projection (3D PCD generate)
    - PCA & Density Centroid

#### 2025-12-05 ~ 2025-12-
- Decision Module *Ablation Study* Specification
    - About **Depth Enhancement**
        1. Non-preprocessing
            - Objective : "데이터가 거칠고 구멍이 뚫려 있으면 중심점 추정(Centroid)과 PCA가 얼마나 불안정한가"
        2. OpenCV Inpainting + Gaussian Smoothing
            - Objective : "단순히 구멍만 메우면(Inpainting) 점의 개수는 늘어나지만, Edge가 뭉개져서(Blurring) 손잡이와 몸통이 붙어버리거나 경계가 모호해진다"
        3. RGB-Guided Filter(Bilateral)
            - Objective : "RGB의 선명한 경계선을 가이드로 삼았기에, 구멍은 메우되 물체의 형상(Topology)은 완벽히 유지된다."는 것을 증명
- Control Module Specification
    - Hierarchical Path Planning Module Architecture
        - 이렇게 진행한다면, 3d path planning도 real-time + safety까지 챙길 수 있을 것이라 생각