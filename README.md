# 프로젝트 소개

ADHD (Anomaly Detection with Human Data) 프로젝트는 최근 급증하는 무인점포 범죄 등을 해결하고자, Movenet + LSTM으로 AI model을 학습시켜 CCTV(웹캠) 등에서 realtime으로 Humanpose를 추출하여 폭행 등의 이상 상황을 탐지할 수 있도록 한 AI 프로젝트입니다.

---

### 개발 팀원

- 공영재 (DGIST 19학번)
- 김가연 (성균관대학교 20학번)

---

### 개발환경

- library : react, socket.io
- API : webRTC
- DeepLearning Framework : Movenet
- Language : javascript, css
- Model : LSTM
- Server Deployment : AWS EC2

---

### Major feature

1. Data collection & processing 
    
    폭행 상황 영상(abnormal)과 정상 상황(normal)을 다양한 viewpoint에서 자체적으로 촬영 후(100 sec), 3 fps로 이미지를 나눔 (총 frame 수 = 600)
    
2. Data augmentation
    
    기존에 augmentation없이 model을 학습했을 땐 특정 좌표에서 pose estimation 시 prediction이 되는 문제가 있었는데, data augmentation을 통해 x 좌표를 flip해서 학습을 진행했더니 상대적으로 훨씬 detection을 잘 수행하였음.
    
    <img width="521" alt="image" src="https://github.com/Yeongjae-Kong/madcamp_week3/assets/67358433/994feb5d-d850-4727-97c6-17a974be34a9">

    
3. Pose estimation 
    
    posenet, mediapipe 등의 pose estimation model이 있었지만 최근 Google research에서 성능과 속도면에서 큰 향상을 보인 movenet을 사용하기로 결정. Frame 별로 Human pose를 17개의 (x, y, score) shape를 가지는 Keypoints로 나눠 이를 LSTM model에 넣어 학습함. 
    
![KakaoTalk_20240117_173501232](https://github.com/Yeongjae-Kong/madcamp_week3/assets/67358433/dc911563-235a-4c2a-bdac-57cd378a79de)

1. modeling
    
    동영상 시계열 데이터를 학습하기 위한 Recurrent Neural Network의 일종인 LSTM을 사용함. 이때, 각 frame에 대해 pose estimation을 하여 도출된 output인 (17, 3) keypoints를 입력값으로 사용함
    
2. hyperparameter
    
    Loss : CELoss 
    
    Optimizer : Adam
    
    activation : softmax
    
    epoch : 100 
    
    batchsize : 12 
    
3. Web
    
    React와 webRTC를 활용해 실시간 웹캠 영상을 송출하는 사이트를 만들고, AWS EC2에 [socket.io](http://socket.io) server를 열어 통신하였음.
    

---

### 문제상황 및 보완

1. 다양한 이상 상황을 고려하기 위해 2명 이상의 human pose를 추출하는 movenet multipose를 사용하려 했으나, human pose 수에 따라 동적으로 변화하는 input shape을 처리하는데 어려움을 겪고 결국 singlepose로 학습하고 실행하였음. 기획 단계에서 이를 고려하여 개발했다면 더 효율적으로 프로젝트를 진행할 수 있었을 것이라는 점에서 아쉬움이 남음.
2. 공공 API의 CCTV 이상상황 데이터를 통해 LSTM model 학습을 시도했으나, 비슷한 세트장의 비슷한 viewpoint를 가진 dataset 때문인지 실시간 웹캠에서 model을 연결하니 학습이 제대로 되지 않았음. 
3. 이에 따라 자체적으로 학습 데이터를 녹화하여 생성하였음. 처음 dataset을 학습할 때 fps를 10으로 설정했는데, overfitting되서인지 test accuracy는 높게 나왔지만 realtime webcam에서는 prediction을 제대로 수행하지 못했음. 이후 fps를 3으로 낮춰서 재학습했더니 이전보다 상대적으로 높은 퀄리티를 보여주었음.
