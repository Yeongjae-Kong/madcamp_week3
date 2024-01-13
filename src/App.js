import React, {useRef} from 'react';
import './App.css';
import Webcam from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
// import '@tensorflow/tfjs-backend-wasm';


function App() {
  const webcamRef = useRef(null); 
  const canvasRef = useRef(null);
  
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
