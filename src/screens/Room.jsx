import React, { useEffect, useCallback, useState, useRef } from "react";
import ReactPlayer from "react-player";
import peer from "../service/peer";
import { useSocket } from "../context/SocketProvider";
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs';
import './Room.css';

const RoomPage = () => {
  const socket = useSocket();
  const [remoteSocketId, setRemoteSocketId] = useState(null);
  const [myStream, setMyStream] = useState();
  const [remoteStream, setRemoteStream] = useState();
  const [model, setModel] = useState(null);

  useEffect(() => {
    tf.ready().then(() => {
      tf.loadLayersModel('./model/model.json').then(loadedModel => {
        setModel(loadedModel);
      });
    });
  }, []);

  const handleUserJoined = useCallback(({ email, id }) => {
    console.log(`Email ${email} joined room`);
    setRemoteSocketId(id);
  }, []);

  const handleCallUser = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: { width: 640, height: 640 },
    });
    const offer = await peer.getOffer();
    socket.emit("user:call", { to: remoteSocketId, offer });
    setMyStream(stream);
  }, [remoteSocketId, socket]);

  const handleIncommingCall = useCallback(
    async ({ from, offer }) => {
      setRemoteSocketId(from);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: { width: 640, height: 640 },
      });
      setMyStream(stream);
      console.log(`Incoming Call`, from, offer);
      const ans = await peer.getAnswer(offer);
      socket.emit("call:accepted", { to: from, ans });
    },
    [socket]
  );

  const sendStreams = useCallback(() => {
    for (const track of myStream.getTracks()) {
      peer.peer.addTrack(track, myStream);
    }
  }, [myStream]);

  const handleCallAccepted = useCallback(
    ({ from, ans }) => {
      peer.setLocalDescription(ans);
      console.log("Call Accepted!");
      sendStreams();
    },
    [sendStreams]
  );

  const handleNegoNeeded = useCallback(async () => {
    const offer = await peer.getOffer();
    socket.emit("peer:nego:needed", { offer, to: remoteSocketId });
  }, [remoteSocketId, socket]);

  useEffect(() => {
    peer.peer.addEventListener("negotiationneeded", handleNegoNeeded);
    return () => {
      peer.peer.removeEventListener("negotiationneeded", handleNegoNeeded);
    };
  }, [handleNegoNeeded]);

  const handleNegoNeedIncomming = useCallback(
    async ({ from, offer }) => {
      const ans = await peer.getAnswer(offer);
      socket.emit("peer:nego:done", { to: from, ans });
    },
    [socket]
  );

  const handleNegoNeedFinal = useCallback(async ({ ans }) => {
    await peer.setLocalDescription(ans);
  }, []);

  useEffect(() => {
    peer.peer.addEventListener("track", async (ev) => {
      const remoteStream = ev.streams;
      console.log("GOT TRACKS!!");
      setRemoteStream(remoteStream[0]);
    });
  }, []);

  useEffect(() => {
    socket.on("user:joined", handleUserJoined);
    socket.on("incomming:call", handleIncommingCall);
    socket.on("call:accepted", handleCallAccepted);
    socket.on("peer:nego:needed", handleNegoNeedIncomming);
    socket.on("peer:nego:final", handleNegoNeedFinal);

    return () => {
      socket.off("user:joined", handleUserJoined);
      socket.off("incomming:call", handleIncommingCall);
      socket.off("call:accepted", handleCallAccepted);
      socket.off("peer:nego:needed", handleNegoNeedIncomming);
      socket.off("peer:nego:final", handleNegoNeedFinal);
    };
  }, [
    socket,
    handleUserJoined,
    handleIncommingCall,
    handleCallAccepted,
    handleNegoNeedIncomming,
    handleNegoNeedFinal,
  ]);

  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);

  // TensorFlow Pose Detection 모델 로드
  useEffect(() => {
    tf.ready().then(() => {
      tf.setBackend('webgl').then(() => {
        const loadModel = async () => {
          const detectorConfig = {
            modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
            enableTracking: true,
          };
          const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
          setDetector(detector);
        };
        loadModel();
      });
    });
  }, []);
  const getVideoElement = () => {
    return document.querySelector('.react-player video');
  };

  const playerRef = useRef(null);
  const getPlayerElement = () => {
    return playerRef.current.getInternalPlayer();
  };

  // 포즈 추정 및 그리기
  const detectPose = async () => {
    if (detector) {
      const video = getPlayerElement();
      if (video && video.readyState >= 2) {
         // 캔버스 크기 조정
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const poses = await detector.estimatePoses(video);
        // 캔버스에 포즈 그리는 로직
        drawPose(poses, video);
        poses.forEach((pose) => {
          drawConnections(video, canvasRef.current.getContext("2d"), pose.keypoints, EDGES, 0.5);
        });
      }
    }
  };

  function getColorForEdge(colorCode) {
    switch(colorCode) {
        case 'red':
            return 'red';
        case 'blue':
            return 'blue';
      }
  }

// 연결 정보를 정의합니다.
  const EDGES = {
    // // Head (머리 부분)
    // "0,1": 'red', // nose to left_eye
    // "0,2": 'red', // nose to right_eye
    // "1,3": 'red', // left_eye to left_ear
    // "2,4": 'red', // right_eye to right_ear

    // Body (몸통 부분)
    "5,7": 'blue', // left_shoulder to left_elbow
    "7,9": 'blue', // left_elbow to left_wrist
    "6,8": 'blue', // right_shoulder to right_elbow
    "8,10": 'blue', // right_elbow to right_wrist
    "5,6": 'blue', // left_shoulder to right shoulder
    "5,11": 'blue', // left_shoulder to left_hip
    "6,12": 'blue', // right_shoulder to right_hip
    "11,12": 'blue', // left_hip to right_hip
    "11,13": 'blue', // left_hip to left_knee
    "13,15": 'blue', // left_knee to left_ankle
    "12,14": 'blue', // right_hip to right_knee
    "14,16": 'blue'  // right_knee to right_ankle
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
  const drawPose = (poses, video) => {
    if (!video) return;

    const ctx = canvasRef.current.getContext("2d");
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
          ctx.fillStyle = "white";
          ctx.fill();
        }
      });
    });
  };


  useEffect(() => {
    if (detector) {
      const interval = setInterval(() => {
        detectPose();
      }, 100);
      return () => clearInterval(interval);
    }
  }, [detector, myStream]);

  return (
    <div className="app-container">
    <h1>Room Page</h1>
    <h4>{remoteSocketId ? "Connected" : "No one in room"}</h4>
    {myStream && <button onClick={sendStreams}>Send Stream</button>}
    {remoteSocketId && <button onClick={handleCallUser}>CALL</button>}
    <div className="video-container">
      {myStream && (
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
            url={myStream}
            onReady={() => detectPose()}
          />
          <canvas ref={canvasRef} className="overlay-canvas"></canvas>
        </div>
        </>
      )}
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

export default RoomPage;
