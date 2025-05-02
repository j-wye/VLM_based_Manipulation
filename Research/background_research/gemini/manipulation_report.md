# **매니퓰레이터 암을 이용한 파지 방법론: 2023년 이후 AI 로봇 분야 최신 연구 동향**

**1. 서론**

로봇 파지는 제조 자동화에서 서비스 로봇, 일상생활 지원에 이르기까지 다양한 분야에서 필수적인 기술입니다. 특히 인공지능(AI)의 발전은 로봇 파지 시스템의 성능과 적응성을 크게 향상시키고 있습니다. AI 기반 로봇 파지는 복잡한 환경, 불확실한 물체, 다양한 작업 요구 사항에 효과적으로 대응할 수 있도록 로봇에게 지능적인 의사 결정 능력을 부여합니다. 본 보고서는 2023년 이후 AI 로봇 분야에서 매니퓰레이터 암을 사용하여 물체를 파지하는 최신 방법론을 분석하고, 주요 기술 동향, 해결 과제, 그리고 미래 연구 방향을 제시하는 것을 목표로 합니다. 보고서에서는 딥러닝 기반 파지 기술, 생성 모델의 활용, 언어 및 비전 통합, 시뮬레이션에서 실제 로봇으로의 전환, 촉각 센서의 역할, 그리고 파지 연구를 위한 데이터셋 및 벤치마크의 중요성을 중심으로 논의를 진행할 것입니다.

**2. 딥러닝 기반 파지 기술**

딥러닝은 현대 AI 기반 로봇 파지 기술의 핵심적인 역할을 수행하고 있습니다. 다양한 딥러닝 아키텍처와 기술은 파지의 여러 측면, 즉 물체 감지, 파지 자세 추정, 그리고 파지 제어에 적용되고 있습니다.

* **시각 데이터를 이용한 파지 감지:** 컨볼루션 신경망(CNN)은 RGB 이미지, 깊이 이미지 또는 RGB-D 이미지와 같은 시각적 입력으로부터 직접 파지 매개변수(예: 위치, 방향, 그리퍼 폭)를 예측하는 데 널리 사용됩니다. 1에 따르면, 컨볼루션 신경망(CNN)을 파지 알고리즘에 통합함으로써 로봇은 작업 환경 변화에 더 잘 적응할 수 있게 되었습니다. 이는 CNN이 시각 정보를 처리하여 파지에 필요한 특징을 추출하는 데 효과적이기 때문입니다. 2에서 Grasp-DSC 방법이 GG-CNN 2 아키텍처를 기반으로 한다는 점은 CNN 기반 아키텍처가 파지 기술의 기초를 이루고 있으며 지속적으로 발전하고 있음을 보여줍니다. CNN 기반 방법은 명시적인 3D 물체 모델링 없이도 복잡한 장면에서 새로운 물체를 효율적으로 파지할 수 있는 잠재력을 제공하지만, 성능은 학습 데이터의 다양성과 품질에 크게 의존합니다.  
* **파지를 위한 6D 자세 추정:** 정확하고 성공적인 파지를 위해서는 목표 물체의 6자유도(6D) 자세(3차원 위치 및 방향)를 추정하는 것이 매우 중요합니다. 특히 복잡한 조작 작업에서 이는 필수적입니다. 3과 4에서는 산업 환경에서 로봇 파지 정확도와 적응성을 향상시키기 위해 개발된 새로운 6D 자세 데이터셋을 소개합니다. 이 데이터셋에서 EfficientPose 및 FFB6D와 같은 모델이 각각 97.05% 및 98.09%의 높은 정확도를 달성했다는 점은 6D 자세 추정 분야의 상당한 발전을 의미합니다. 정확한 6D 자세 추정은 로봇이 물체를 매우 정밀하게 조작해야 하는 제조 및 조립 작업에서 특히 중요합니다. 이러한 데이터셋 개발에 대한 집중은 이 연구 분야의 실질적인 중요성을 강조합니다.  
* **파지 정책을 위한 심층 강화 학습:** 심층 강화 학습(DRL) 알고리즘은 시뮬레이션 또는 실제 환경과의 상호 작용을 통해 최적의 파지 정책을 학습하도록 로봇을 훈련하는 데 사용됩니다. 이는 로봇이 성공적인 파지 확률을 최대화하는 행동을 선택하고 실행하는 방법을 학습하는 과정을 포함합니다. 5에서는 로봇 팔이 물체의 분류 및 주어진 작업을 기반으로 자율적으로 파지할 수 있도록 하는 심층 결정론적 정책 경사 접근 방식을 제안합니다. 이는 DRL이 복잡한 파지 행동 학습에 어떻게 적용될 수 있는지 보여줍니다. 그러나 6에서 언급된 바와 같이, 실제 로봇에 DRL을 적용하는 데에는 광범위한 훈련 데이터 요구 사항 및 시뮬레이션에서 실제 로봇으로의 정책 전환(sim-to-real 격차)과 같은 과제가 여전히 존재합니다. DRL은 명시적인 프로그래밍 없이도 매우 복잡하고 적응적인 파지 기술을 학습할 수 있는 잠재력을 제공하지만, 실제 로봇 공학에서의 실질적인 적용은 샘플 효율성, 탐색 전략 및 sim-to-real 격차 해결과 관련된 과제를 해결해야 합니다.

**3. 로봇 조작에서의 생성 모델**

생성 모델은 데이터 증강, 파지 합성, 복잡한 조작 정책 학습 등 로봇 파지의 다양한 과제를 해결하는 데 점점 더 중요한 역할을 하고 있습니다.

* **파지 합성을 위한 VAE(Variational Autoencoders):** VAE는 성공적인 파지 구성의 확률적 잠재 공간 표현을 학습하는 데 사용됩니다. 이 잠재 공간에서 샘플링함으로써 VAE는 주어진 물체에 대해 다양하고 잠재적으로 새로운 파지 자세를 생성할 수 있습니다. 7와 8에서는 VAE를 로봇 조작을 위한 생성 모델의 정책 계층 내에서 파지 생성 방법으로 분류합니다. 이는 VAE가 생성 AI를 사용하여 행동 생성을 위한 더 넓은 맥락에 있음을 보여줍니다. 7에 나열된 Contact-GraspNet은 파지 생성을 위한 VAE 기반 방법의 특정 예입니다. VAE는 성공적인 파지의 고유한 불확실성과 가변성을 모델링하는 방법을 제공하여 로봇이 더 넓은 범위의 파지 전략을 탐색하고 새로운 물체나 상황에 더 잘 적응할 수 있도록 합니다.  
* **파지 및 궤적 생성을 위한 확산 모델:** 이미지 및 비디오 생성에서 놀라운 성공을 거둔 확산 모델은 로봇 조작 작업, 특히 파지 자세 및 전체 조작 궤적 생성에 점점 더 많이 적용되고 있습니다. 이러한 모델은 점진적인 노이즈 추가 프로세스를 역전시켜 현실적이고 복잡한 행동 시퀀스를 생성하는 방법을 학습합니다. 7에서는 파지 예측 및 궤적 계획을 포함하여 로봇 조작에서 확산 모델의 적용이 증가하고 있음을 명시적으로 강조합니다. 9에서는 이를 "유망한 접근 방식"이라고까지 칭합니다. 10의 DexGrasp Anything은 훈련 및 샘플링 단계 모두에서 확산 모델에 물리적 제약 조건을 통합하여 더 실용적이고 강력한 섬세한 파지를 생성하는 새로운 방법으로, 물리적 현실성 보장이라는 초기 확산 기반 접근 방식의 주요 한계를 해결합니다. 11의 GraspLDM은 VAE의 잠재 공간에서 확산 모델을 사전 확률로 사용하여 6D 파지 합성을 위한 모듈형 생성 프레임워크를 제시합니다. 이는 VAE와 확산 모델의 강점을 결합한 하이브리드 접근 방식을 보여줍니다. 확산 모델은 복잡하고 고차원적인 분포를 모델링하는 데 탁월하며 다양하고 현실적인 파지 및 궤적을 생성할 수 있어 로봇 조작의 복잡성을 해결하는 강력한 도구를 제공합니다. 물리적 제약 조건 통합 및 하이브리드 VAE-확산 접근 방식은 이러한 모델을 실제 로봇 작업에 더 효과적으로 만드는 데 있어 발전된 모습을 보여줍니다.  
* **데이터 증강 및 파지 학습을 위한 GAN(Generative Adversarial Networks):** GAN은 실제 데이터셋을 보강하여 파지 모델의 견고성과 일반화 능력을 향상시킬 수 있는 합성 데이터를 생성하는 데 사용될 수 있습니다. GAN은 파지 정책을 직접 학습하거나 파지 품질을 예측하는 데에도 사용될 수 있습니다. 7와 8에서는 데이터 생성을 포함하여 로봇 조작을 위한 생성 모델의 맥락에서 GAN을 언급합니다. 7의 GraspAda는 파지에서 도메인 적응을 위해 조건부 GAN(cGAN)을 사용하는 예로, sim-to-real 격차를 해소하는 데 도움이 됩니다. GAN은 다양한 훈련 데이터를 생성하여 파지 모델의 다양성을 향상시키는 강력한 도구를 제공하며, 이는 광범위한 물체와 다양한 환경에서 잘 작동하는 모델을 개발하는 데 중요합니다. 도메인 적응에서의 활용은 sim-to-real 과제를 해결하는 데 있어 또 다른 유용성을 강조합니다.

**4\. 향상된 파지를 위한 언어 및 비전 통합**

자연어 이해와 시각적 인식을 통합하여 보다 직관적이고 작업 지향적인 로봇 파지를 가능하게 하는 추세가 증가하고 있습니다. 이는 인간이 자연어 명령을 사용하여 로봇을 지시할 수 있도록 하는 비전-언어 모델(VLM)을 사용하는 것을 포함합니다.

VLM은 높은 수준의 명령을 해석하고 장면에서 관련 물체를 식별하며 적절한 파지 행동을 생성할 수 있습니다. 12에서는 인터넷 규모의 기초 모델(예: LLM 및 비전 모델)에서 생성된 언어-추론 분할 마스크를 활용하여 로봇 조작 작업을 조건화하는 새로운 패러다임을 제안합니다. 이 스니펫은 GPT-4를 사용하여 언어 명령에서 대상 물체에 대한 추론을 수행하고 SAM을 사용하여 분할 마스크를 생성한 다음 로봇 행동을 안내하는 파이프라인을 자세히 설명합니다. 13의 DexGraspVLA는 기초 모델과 모방 학습의 상호 보완적인 강점을 통합하는 일반적인 섬세한 파지를 위한 계층적 비전-언어-행동(VLA) 프레임워크로 설명됩니다. 이 프레임워크는 사전 훈련된 비전-언어 모델을 높은 수준의 작업 플래너로 사용하고 확산 기반 정책을 낮은 수준의 행동 컨트롤러로 학습합니다. 언어와 비전의 통합은 보다 사용자 친화적이고 다재다능한 로봇 파지 시스템을 향한 중요한 진전을 나타냅니다. 인간의 언어 명령을 이해함으로써 로봇은 보다 복잡하고 상황 인식적인 조작 작업을 수행할 수 있어 인간-로봇 협업의 새로운 가능성을 열어줍니다.

**5. Sim-to-Real 격차 해소**

"Sim-to-real 격차"는 시뮬레이션 환경에서 훈련된 로봇 제어 정책을 실제 로봇으로 전송할 때 발생하는 성능 저하를 의미하는 지속적인 과제입니다. 이 격차는 감지, 작동 및 물리학의 불일치를 포함하여 시뮬레이션 모델의 충실도와 실제 세계의 복잡성 간의 차이로 인해 발생합니다.

이 격차를 해소하기 위한 다양한 방법론과 연구 노력이 진행 중입니다. 도메인 무작위화는 모델을 더 강력하게 만들기 위해 시뮬레이션 매개변수에 무작위 변동을 도입하는 것을 포함하는 일반적인 기술입니다. 도메인 적응은 시뮬레이션 및 실제 데이터의 특징 분포를 정렬하는 것을 목표로 합니다. 7에서는 GraspAda가 cGAN을 사용하여 도메인 적응을 수행하는 것을 언급합니다. ICRA와 함께 개최되는 로봇 Sim2Real 챌린지와 같은 대회는 14 파지를 포함한 로봇 작업에서 sim-to-real 전송에 대한 다양한 접근 방식을 테스트하고 비교할 수 있는 플랫폼을 제공합니다. 17과 18에서는 ICRA 2024 로봇 Sim2Real 챌린지의 광물 탐색 작업에서 1위를 차지한 완전 자율 로봇 시스템을 제시합니다. 이 시스템의 핵심 기능은 알고리즘 수정 없이 sim-to-real 전송 중에 감지 및 작동 불일치를 극복할 수 있다는 점으로, 강력한 전송 달성에 상당한 진전을 보여줍니다. 19에서 논의된 바와 같이, 디지털 트윈(DT)을 Embodied AI와 통합하는 것은 실제 시스템을 더 정확하게 반영하는 동적이고 데이터가 풍부한 가상 환경을 만들어 sim-to-real 격차를 해소하는 방법이 될 수 있습니다. Sim-to-real 격차를 해결하는 것은 AI 기반 로봇 파지의 실제 응용 프로그램에 널리 배포하는 데 매우 중요합니다. Sim2Real 챌린지와 같은 대회에서 시스템의 성공과 디지털 트윈과 같은 개념의 탐구는 이 중요한 분야에서 관심과 발전이 증가하고 있음을 나타냅니다.

**6. 고급 파지에서의 촉각 센서의 역할**

촉각 피드백은 특히 시각 정보가 제한되거나 불확실하거나 가려진 상황에서 로봇 파지의 민첩성, 견고성 및 신뢰성을 향상시키는 데 중요한 역할을 합니다. 촉각 센서는 로봇에게 접촉력, 질감 및 모양과 같은 물체 속성, 잠재적인 미끄러짐에 대한 정보를 제공합니다.

최근에는 촉각 센서 기술의 발전과 AI 알고리즘과의 통합을 통해 파지가 개선되고 있습니다. 20에서는 Sanctuary AI가 Phoenix 범용 로봇에 새로운 촉각 센서 기술을 개발 및 통합한 것을 논의합니다. 이러한 스니펫은 시각 정보가 없는 상황에서의 물체 집기, 미끄러짐 감지, 과도한 힘 적용 방지와 같은 작업을 가능하게 하는 인간 수준의 로봇 손재주를 달성하는 데 있어 촉각의 중요성을 강조합니다. 23에 설명된 F-TAC 핸드는 인간과 유사한 손재주를 유지하면서 손 표면의 상당 부분에 걸쳐 고해상도 촉각 감지 기능을 제공하여 포괄적인 촉각 피드백을 갖춘 로봇 손 개발의 진전을 보여줍니다. 24에서 언급된 생체 모방 촉각 센서는 생물학적 시스템에서 영감을 얻었습니다. 이러한 센서는 자연스러운 촉각의 민감도, 반응성 및 다중 모드 인식을 복제하여 로봇의 촉각 인식을 향상시키는 것을 목표로 합니다. 고급 촉각 감지 기능을 AI 알고리즘과 통합하는 것은 로봇이 더 복잡하고 미묘한 파지 작업을 수행할 수 있도록 하는 핵심 요소입니다. 물체를 "느낄" 수 있는 능력은 로봇이 파지를 실시간으로 조정하여 성공률을 높이고 깨지기 쉬운 물체의 손상을 방지할 수 있도록 합니다.

**7. 로봇 파지 연구를 위한 데이터셋 및 벤치마크**

고품질의 대규모 및 다양한 데이터셋과 표준화된 벤치마크는 AI 기반 로봇 파지 분야의 발전을 주도하는 데 매우 중요한 역할을 합니다. 이러한 리소스는 복잡한 AI 모델을 훈련하는 데 필요한 데이터를 제공하고 다양한 접근 방식의 성능을 평가하고 비교할 수 있는 공통 기반을 제공합니다.

최근 연구와 관련된 데이터셋 및 벤치마크를 살펴보겠습니다. 25에서는 일반적인 물체 파지를 위한 광범위한 훈련 데이터와 표준 평가 플랫폼을 제공하는 중요한 대규모 벤치마크인 GraspNet-1Billion 데이터셋을 언급합니다. 3과 4에서는 일본 시바우라 공업대학 연구진이 개발한 새로운 6D 자세 데이터셋을 강조합니다. 다양한 물체의 RGB 및 깊이 이미지를 특징으로 하는 이 데이터셋은 산업 환경에서 로봇 파지 정확도를 크게 향상시킬 수 있는 잠재력을 입증했으며, 최첨단 딥러닝 모델로 높은 정확도를 달성했습니다. 26에서는 다양한 시나리오(더미 및 표면 집기)에서 여러 물체를 파지하고 조작하는 로봇의 능력을 평가하는 프로토콜을 도입하는 다중 물체 파지 벤치마크를 제시합니다. 이 벤치마크는 로봇이 더 복잡한 다중 항목 작업을 처리해야 하는 필요성이 증가함에 따라 등장했습니다. 14에서는 ICRA 및 IROS에서 진행 중인 로봇 파지 및 조작 대회(RGMC)를 언급합니다. 이 대회 시리즈는 실제 시나리오에서 로봇 파지 및 조작 능력의 한계를 뛰어넘도록 설계된 일련의 어려운 작업을 제공합니다. 특정 과제(예: 6D 자세 추정 또는 다중 물체 파지)에 초점을 맞춘 특수 데이터셋 개발과 벤치마크 대회의 지속은 실제 복잡성을 해결하고 로봇 조작의 한계를 뛰어넘는 데 점점 더 초점을 맞추는 성숙한 분야를 나타냅니다.

**표 2: 최근 로봇 파지용 데이터셋 및 벤치마크**

| 데이터셋/벤치마크 이름 | 주요 특징/초점 | 도입/강조된 연도(스니펫 기반) | 스니펫 참조 |
| :---- | :---- | :---- | :---- |
| GraspNet-1Billion | 일반 파지를 위한 대규모 데이터셋 | 2020 | 25 |
| 새로운 6D 자세 데이터셋 (시바우라 공업대학) | 산업용 로봇을 위한 정확한 6D 자세 추정 | 2024 | 3 |
| 다중 물체 파지 벤치마크 | 다양한 시나리오에서 다중 물체 파지 능력 평가 | 2025 | 26 |
| 로봇 파지 및 조작 대회 (RGMC) | 다양한 파지 및 조작 작업을 포함한 대회 | 진행 중 (2023, 2024, 2025 언급) | 14 |

**8. 결론 및 향후 연구 방향**

본 보고서에서 강조된 AI 기반 로봇 파지 방법론의 주요 발전 사항은 딥러닝, 생성 모델, 언어 및 비전 통합, sim-to-real 전송 기술, 그리고 촉각 감지 기술의 상당한 기여를 보여줍니다.

여전히 해결해야 할 과제로는 완전히 새로운 물체와 환경에 대한 일반화 능력 향상, 매우 복잡하고 역동적인 장면에서의 견고성 향상, 계산 집약적인 AI 모델을 사용한 실시간 성능 달성, 그리고 지속적인 sim-to-real 격차 해소 등이 있습니다. 향후 연구 방향으로는 물리적 제약 조건과 작업별 요구 사항을 통합할 수 있는 보다 정교한 생성 모델 개발, 보다 직관적이고 높은 수준의 로봇 조작 제어를 위한 대규모 언어 모델 및 고급 비전 모델의 통합 심층 탐구, 더 높은 해상도, 더 큰 감도, 그리고 더 강력한 설계를 포함한 촉각 감지 기술의 지속적인 혁신과 풍부한 촉각 피드백을 효과적으로 활용할 수 있는 AI 알고리즘 개발, 보다 현실적이고 포괄적인 시뮬레이션 환경 구축 및 학습된 정책을 실제 로봇으로 전송하기 위한 보다 효과적이고 효율적인 기술 개발, 불확실성 처리 및 장기 조작을 포함하여 실제 파지 작업의 복잡성을 더 잘 반영하는 보다 도전적이고 다양한 벤치마크 데이터셋 구축, 그리고 새로운 물체, 작업 및 환경에 대한 반응으로 로봇이 파지 기술을 지속적으로 학습하고 적응할 수 있도록 하는 평생 학습 및 메타 학습 접근 방식 연구 등이 있습니다.

AI 기반 로봇 파지 분야는 최근 몇 년 동안 놀라운 발전을 이루었지만, 로봇이 진정으로 인간과 유사한 손재주와 적응력을 달성하기 위해서는 여전히 상당한 과제가 남아 있습니다. 미래 연구는 이러한 한계를 극복하고 광범위한 응용 분야에서 로봇 조작의 잠재력을 최대한 발휘하기 위해 다양한 AI 기술과 감지 방식의 시너지 효과에 초점을 맞출 것으로 예상됩니다.

**표 1: 로봇 파지를 위한 주요 AI 방법론 요약**

| 방법론 | 파지에서의 주요 응용 분야 | 대표 스니펫 | 주요 장점 | 현재 한계/과제 |
| :---- | :---- | :---- | :---- | :---- |
| 딥러닝 (CNN, DRL) | 파지 감지, 자세 추정, 제어 | 1 | 데이터 기반 학습, 새로운 물체 처리, 직관적인 제어 | 데이터 요구 사항, sim-to-real 격차, 계산 비용, 매우 다양한 시나리오에 대한 일반화 |
| 생성 모델 (VAE, 확산 모델, GAN) | 파지 합성, 궤적 계획, 데이터 증강 | 7 | 다양한 파지 생성, 복잡한 분포 모델링, 데이터 다양성 향상 | 물리적 현실성 보장, 샘플링 속도, 훈련 안정성 |
| 비전-언어 모델 (VLM) | 작업 지향적 파지, 고수준 제어 | 12 | 직관적인 제어, 상황 인식 조작 | 언어적 모호성 처리, 물리적 세계에 대한 접지 |
| 촉각 감지 | 향상된 견고성, 미끄러짐 감지, 물체 속성 인식 | 20 | 불확실한 환경에서의 견고성 향상, 섬세한 조작 가능 | 센서 기술의 한계 (해상도, 내구성), AI와의 통합 |

#### **참고 자료**

1. Robotic Grasping of Unknown Objects Based on Deep Learning-Based Feature Detection, 5월 1, 2025에 액세스, [https://www.mdpi.com/1424-8220/24/15/4861](https://www.mdpi.com/1424-8220/24/15/4861)  
2. A Light-Weight Grasping Pose Estimation Method for Mobile Robotic ..., 5월 1, 2025에 액세스, [https://www.mdpi.com/2076-0825/14/2/50](https://www.mdpi.com/2076-0825/14/2/50)  
3. Innovative 6D pose dataset sets new standard for robotic grasping performance, 5월 1, 2025에 액세스, [https://www.sciencedaily.com/releases/2025/01/250116133546.htm](https://www.sciencedaily.com/releases/2025/01/250116133546.htm)  
4. Boosting Robotic Grasping Performance with Six Degrees of Freedom Dataset \- Tech Briefs, 5월 1, 2025에 액세스, [https://www.techbriefs.com/component/content/article/52807-boosting-robotic-grasping-performance-with-six-degrees-of-freedom-dataset](https://www.techbriefs.com/component/content/article/52807-boosting-robotic-grasping-performance-with-six-degrees-of-freedom-dataset)  
5. Robotic Grasping Based on Deep Learning: A Survey \- ResearchGate, 5월 1, 2025에 액세스, [https://www.researchgate.net/publication/376577884\_Robotic\_Grasping\_Based\_on\_Deep\_Learning\_A\_Survey](https://www.researchgate.net/publication/376577884_Robotic_Grasping_Based_on_Deep_Learning_A_Survey)  
6. Sim-to-Real Transfer Learning using Robustified Controllers in Robotic Tasks involving Complex Dynamics | Request PDF \- ResearchGate, 5월 1, 2025에 액세스, [https://www.researchgate.net/publication/335139069\_Sim-to-Real\_Transfer\_Learning\_using\_Robustified\_Controllers\_in\_Robotic\_Tasks\_involving\_Complex\_Dynamics](https://www.researchgate.net/publication/335139069_Sim-to-Real_Transfer_Learning_using_Robustified_Controllers_in_Robotic_Tasks_involving_Complex_Dynamics)  
7. arxiv.org, 5월 1, 2025에 액세스, [https://arxiv.org/pdf/2503.03464](https://arxiv.org/pdf/2503.03464)  
8. Generative Artificial Intelligence in Robotic Manipulation: A Survey \- arXiv, 5월 1, 2025에 액세스, [https://arxiv.org/html/2503.03464v1](https://arxiv.org/html/2503.03464v1)  
9. Diffusion Models for Robotic Manipulation: A Survey \- arXiv, 5월 1, 2025에 액세스, [https://arxiv.org/html/2504.08438v1](https://arxiv.org/html/2504.08438v1)  
10. arXiv:2503.08257v2 \[cs.CV\] 16 Mar 2025, 5월 1, 2025에 액세스, [https://arxiv.org/pdf/2503.08257?](https://arxiv.org/pdf/2503.08257)  
11. GraspLDM: Generative 6-DoF Grasp Synthesis Using Latent Diffusion Models \- Orbi.lu \- University of Luxembourg, 5월 1, 2025에 액세스, [https://orbilu.uni.lu/handle/10993/63367](https://orbilu.uni.lu/handle/10993/63367)  
12. arxiv.org, 5월 1, 2025에 액세스, [https://arxiv.org/pdf/2306.05716](https://arxiv.org/pdf/2306.05716)  
13. DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping, 5월 1, 2025에 액세스, [https://dexgraspvla.github.io/assets/paper/DexGraspVLA.pdf](https://dexgraspvla.github.io/assets/paper/DexGraspVLA.pdf)  
14. Robotic Grasping and Manipulation Competition \- IEEE ICRA 2025, 5월 1, 2025에 액세스, [https://2025.ieee-icra.org/event/robotic-grasping-and-manipulation-competition/](https://2025.ieee-icra.org/event/robotic-grasping-and-manipulation-competition/)  
15. Competitions \- IEEE ICRA 2025, 5월 1, 2025에 액세스, [https://2025.ieee-icra.org/competitions/](https://2025.ieee-icra.org/competitions/)  
16. The 4th Robotic Sim2Real Challenges \- IEEE ICRA 2025, 5월 1, 2025에 액세스, [https://2025.ieee-icra.org/event/the-4th-robotic-sim2real-challenges/](https://2025.ieee-icra.org/event/the-4th-robotic-sim2real-challenges/)  
17. Robotic Sim-to-Real Transfer for Long-Horizon Pick-and-Place Tasks in the Robotic Sim2Real CompetitionThis paper has been accepted for presentation at ICRA 2025\. The final version will be available in IEEE Xplore. \- arXiv, 5월 1, 2025에 액세스, [https://arxiv.org/html/2503.11012v1](https://arxiv.org/html/2503.11012v1)  
18. \[2503.11012\] Robotic Sim-to-Real Transfer for Long-Horizon Pick-and-Place Tasks in the Robotic Sim2Real CompetitionThis paper has been accepted for presentation at ICRA 2025\. The final version will be available in IEEE Xplore. \- ar5iv, 5월 1, 2025에 액세스, [https://ar5iv.labs.arxiv.org/html/2503.11012](https://ar5iv.labs.arxiv.org/html/2503.11012)  
19. Digital twins to embodied artificial intelligence: review and perspective \- OAE Publishing Inc., 5월 1, 2025에 액세스, [https://www.oaepublish.com/articles/ir.2025.11](https://www.oaepublish.com/articles/ir.2025.11)  
20. Sanctuary AI Equips General Purpose Robots with New Touch Sensors for Performing Highly Dexterous Tasks, 5월 1, 2025에 액세스, [https://www.sanctuary.ai/blog/sanctuary-ai-equips-general-purpose-robots](https://www.sanctuary.ai/blog/sanctuary-ai-equips-general-purpose-robots)  
21. Sanctuary AI integrates tactile sensors into Phoenix general purpose robots, 5월 1, 2025에 액세스, [https://www.therobotreport.com/sanctuary-ai-integrates-tactile-sensors-into-phoenix-general-purpose-robots/](https://www.therobotreport.com/sanctuary-ai-integrates-tactile-sensors-into-phoenix-general-purpose-robots/)  
22. Sanctuary AI unveils new tactile sensor technology to enhance robot dexterity, 5월 1, 2025에 액세스, [https://www.roboticsandautomationmagazine.co.uk/news/robotics-as-a-service/sanctuary-ai-unveils-new-tactile-sensor-technology-to-enhance-robot-dexterity.html](https://www.roboticsandautomationmagazine.co.uk/news/robotics-as-a-service/sanctuary-ai-unveils-new-tactile-sensor-technology-to-enhance-robot-dexterity.html)  
23. Embedding high-resolution touch across robotic hands enables adaptive human-like grasping1footnote 1FootnoteFootnoteFootnotesFootnotes1footnote 1A video demonstration of the system is available on Vimeo (up to 4k, with English subtitle). The complete URL is https://vimeo.com/1039184307. \- arXiv, 5월 1, 2025에 액세스, [https://arxiv.org/html/2412.14482v2](https://arxiv.org/html/2412.14482v2)  
24. Recent Developments and Applications of Tactile Sensors with Biomimetic Microstructures, 5월 1, 2025에 액세스, [https://www.mdpi.com/2313-7673/10/3/147](https://www.mdpi.com/2313-7673/10/3/147)  
25. GraspNet-1Billion Benchmark (Robotic Grasping) | Papers With Code, 5월 1, 2025에 액세스, [https://paperswithcode.com/sota/robotic-grasping-on-graspnet-1billion](https://paperswithcode.com/sota/robotic-grasping-on-graspnet-1billion)  
26. Benchmarking Multi-Object Grasping \- arXiv, 5월 1, 2025에 액세스, [https://arxiv.org/html/2503.20820v2](https://arxiv.org/html/2503.20820v2)  
27. Robotic Grasping and Manipulation Competition at the 2024 IEEE/RAS International Conference on Robotics and Automation \- National Institute of Standards and Technology, 5월 1, 2025에 액세스, [https://www.nist.gov/publications/robotic-grasping-and-manipulation-competition-2024-ieeeras-international-conference](https://www.nist.gov/publications/robotic-grasping-and-manipulation-competition-2024-ieeeras-international-conference)