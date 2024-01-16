import React, { useRef, useEffect, useState } from 'react';
import './App.css';
import Webcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs';
const model = await tf.loadLayersModel('file://model/model.json');

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);

  useEffect(() => {
    tf.ready().then(() => { 
      tf.setBackend('webgl').then(() => {
        const loadModel = async () => {
          const detectorConfig = {
            modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
            enableTracking: true,
            trackerType: poseDetection.TrackerType.BoundingBox,
          };
          const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
          setDetector(detector);
        };
        loadModel();
      });
    });
  }, []);

  // 포즈 감지 및 캔버스에 결과 그리기
  const detect = async () => {
    if (!detector) {
      console.log('Detector is not ready');
      return;
    }

    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const poses = await detector.estimatePoses(video);
      console.log("poses: ", poses);
      drawPose(poses);
      poses.forEach((pose) => {
        drawConnections(video, canvasRef.current.getContext("2d"), pose.keypoints, EDGES, 0.5);
      });
      console.log(poses);
    }
  };

  function getColorForEdge(colorCode) {
    switch(colorCode) {
        case 'm':
            return 'magenta';
        case 'c':
            return 'cyan';
        case 'y':
            return 'yellow';
        default:
            return 'black';
    }
}

// 연결 정보를 정의합니다.
const EDGES = {
    "0,1": 'm',
    "0,2": 'c',
    "1,3": 'm',
    "2,4": 'c',
    "0,5": 'm',
    "0,6": 'c',
    "5,7": 'm',
    "7,9": 'm',
    "6,8": 'c',
    "8,10": 'c',
    "5,6": 'y',
    "5,11": 'm',
    "6,12": 'c',
    "11,12": 'y',
    "11,13": 'm',
    "13,15": 'm',
    "12,14": 'c',
    "14,16": 'c'
};

function drawConnections(video,ctx, keypoints, edges, confidenceThreshold) {
  Object.keys(edges).forEach(edge => {
    const scaleX = canvasRef.current.width / video.videoWidth;
    const scaleY = canvasRef.current.height / video.videoHeight;
    const [p1, p2] = edge.split(',').map(Number);
    const color = edges[edge];
    const kp1 = keypoints[p1];
    const kp2 = keypoints[p2];

    if (kp1.score > confidenceThreshold && kp2.score > confidenceThreshold) {
      // 색상을 기반으로 선을 그립니다.
      ctx.beginPath();
      ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
      ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
      ctx.strokeStyle = getColorForEdge(color);
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
}

  // 감지된 포즈를 캔버스에 그리는 함수
  const drawPose = (poses) => {
    const ctx = canvasRef.current.getContext("2d");
    const video = webcamRef.current.video;

    const scaleX = canvasRef.current.width / video.videoWidth;
    const scaleY = canvasRef.current.height / video.videoHeight;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  
    poses.forEach((pose) => {
      pose.keypoints.forEach((keypoint) => {
        if (keypoint.score > 0.5) {
          const x = keypoint.x * scaleX;
          const y = keypoint.y * scaleY;
          console.log(`Drawing keypoint at (${x}, ${y}) with score ${keypoint.score}`);
          console.log(2*Math.PI)
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, 2 * Math.PI);
          ctx.fillStyle = "green";
          ctx.fill();
        }
      });
    });
  };

  useEffect(() => {
    if (detector) {
      const interval = setInterval(() => {
        detect();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [detector]);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;

