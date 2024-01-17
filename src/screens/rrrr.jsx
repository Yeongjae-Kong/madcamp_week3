import React, { useEffect, useState, useRef } from "react";
import ReactPlayer from "react-player";
import peer from "../service/peer";
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs';
import './Room.css';

const videoPath = '/2.mp4'; // 저장된 비디오 파일의 경로  

const checkVideoExistence = async () => {
  try {
    const response = await fetch(videoPath, { method: 'HEAD' });

    if (response.status === 200) {
      console.log(`비디오 파일이 존재합니다: ${videoPath}`);
    } else {
      console.log(`비디오 파일이 존재하지 않습니다: ${videoPath}`);
    }
  } catch (error) {
    console.error('비디오 파일 확인 중 오류 발생:', error);
  }
};

const RoomPage = () => {
  const [remoteStream, setRemoteStream] = useState();
  const [model, setModel] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const loadedModel = await tf.loadLayersModel('/model/model.json');
      setModel(loadedModel);
    };
    checkVideoExistence();
  
    loadModel();
  }, []);

  useEffect(() => {
    peer.peer.addEventListener("track", async (ev) => {
      const remoteStream = ev.streams;
      console.log("GOT TRACKS!!");
      setRemoteStream(remoteStream[0]);
    });
  }, []);

  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);

  // TensorFlow Pose Detection 모델 로드
  useEffect(() => {
    tf.ready().then(() => {
      tf.setBackend('webgl').then(() => {
        const loadModel = async () => {
          const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
          const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
          setDetector(detector);
        };
        loadModel();
      });
    });
  }, []);

  const playerRef = useRef(null);

  // 포즈 추정 및 그리기
  const detectPose = async () => {
    if (detector && model && playerRef.current) {
      const video = playerRef.current.getInternalPlayer();
      if (video && video.readyState >= 2) {
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const poses = await detector.estimatePoses(video);
        const processedPoses = processPoses(poses);

        let inputTensor;
        if (processedPoses.length > 0) {
          inputTensor = tf.tensor3d(processedPoses, [processedPoses.length, 17, 3]);
        } else {
          inputTensor = tf.zeros([1, 17, 3]);
        }

        const predictionTensor = await model.predict(inputTensor);
        const prediction = await predictionTensor.data();

        const drawPrediction = (prediction) => {
          if (prediction[0] > 0.5) { // 예측 결과에 따라 "절도 발생" 메시지 표시
            const ctx = canvasRef.current.getContext("2d");
            const text = "절도 발생";
            ctx.font = "24px Arial";
            ctx.fillStyle = "red";
            ctx.fillText(text, (canvasRef.current.width - ctx.measureText(text).width) / 2, canvasRef.current.height - 20);
          }
        };

        drawPrediction(prediction);
        poses.forEach((pose) => {
          drawConnections(video, canvasRef.current.getContext("2d"), pose.keypoints, EDGES, 0.5);
        });
      }
    }
  };

  function processPoses(poses) {
    return poses.map(poseArray => {
      const defaultKeypoints = new Array(17).fill([1, 1, 0]);
      if (!poseArray || !poseArray.keypoints) {
          console.error('Invalid pose data:', poseArray);
          return defaultKeypoints;
      }
  
      const keypoints = poseArray.keypoints;
  
      return keypoints.map(keypoint => {
          return [keypoint.x, keypoint.y, keypoint.score];
      });
    });
  }

  function getColorForEdge(colorCode) {
    switch(colorCode) {
        case 'red':
            return 'red';
        case 'blue':
            return 'green';
        default:
            return 'black'; // 기본값을 설정합니다.
      }
  }

  const EDGES = {
    "5,7": 'blue',
    "7,9": 'blue',
    "6,8": 'blue',
    "8,10": 'blue',
    "5,6": 'blue',
    "5,11": 'blue',
    "6,12": 'blue',
    "11,12": 'blue',
    "11,13": 'blue',
    "13,15": 'blue',
    "12,14": 'blue',
    "14,16": 'blue'
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
        ctx.beginPath();
        ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
        ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
        ctx.strokeStyle = getColorForEdge(color);
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  };

  useEffect(() => {
    if (detector && model) {
      const interval = setInterval(() => {
        detectPose();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [detector, model]);

  const onVideoReady = () => {
    const checkVideoLoaded = setInterval(() => {
      const video = playerRef.current.getInternalPlayer();
      if (video && video.readyState >= 2) {
        clearInterval(checkVideoLoaded);
        detectPose();
      }
    }, 100);
  };
  

  return (
    <div className="app-container">
      <h1>Room Page</h1>
      <div className="video-container">
          <>
            <h1>My Stream</h1>
            <div className="player-wrapper">
              <ReactPlayer
                ref={playerRef}
                className="react-player"
                playing
                muted
                height="100%"
                width="100%"
                url={videoPath}
                controls
                onReady={onVideoReady}
              />
              <canvas ref={canvasRef} className="overlay-canvas"></canvas>
            </div>
          </>
        {remoteStream && (
          <>
            <h1>Remote Stream</h1>
            <ReactPlayer
              className="react-player"
              playing
              muted
              height="100%"
              width="100%"
              url={remoteStream}
            />
          </>
        )}
        <canvas ref={canvasRef} className="overlay-canvas"></canvas>
      </div>
    </div>
  );
};

