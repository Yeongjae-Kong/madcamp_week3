import React, {useRef, useEffect, useState} from 'react';
import './App.css';
import Webcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-wasm';


function App() {
  const webcamRef = useRef(null); 
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);

  // MoveNet 모델 로드
  useEffect(() => {
    const loadModel = async () => {
      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
        enableTracking: true,
        trackerType: poseDetection.TrackerType.BoundingBox 
      };
      const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
      setDetector(detector);
    };
    loadModel();
  }, []);

  // 포즈 감지 및 캔버스에 결과 그리기
  const detect = async () => {
    if (webcamRef.current && detector) {
      const webcam = webcamRef.current;
      const video = webcam.video;

      if (video.readyState === 4) {
        const videoWidth = webcam.video.videoWidth;
        const videoHeight = webcam.video.videoHeight;

        webcam.video.width = videoWidth;
        webcam.video.height = videoHeight;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const poses = await detector.estimatePoses(video);
        drawPose(poses);
      }
    }
  };

  // 감지된 포즈를 캔버스에 그리는 함수
  const drawPose = (poses) => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    poses.forEach((pose) => {
      pose.keypoints.forEach((keypoint) => {
        if (keypoint.score > 0.5) {
          ctx.beginPath();
          ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "Aqua";
          ctx.fill();
        }
      });
    });
  };

  useEffect(() => {
    const interval = setInterval(() => {
      detect();
    }, 100);
    return () => clearInterval(interval);
  }, [detector]);

  
  return (
    <div className="App">
        <header className="App-header">
          <Webcam
            ref={webcamRef}
            style = {{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zindex : 9,
              width: 640,
              height: 480
            }}
            />

            <canvas 
              ref={canvasRef}
              style = {{
                position: "absolute",
                marginLeft: "auto",
                marginRight: "auto",
                left: 0,
                right: 0,
                textAlign: "center",
                zindex : 9,
                width: 640,
                height: 480
              }} />
        </header>
    </div>
  );
}

export default App;
