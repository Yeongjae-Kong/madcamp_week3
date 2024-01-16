// const tf = require('@tensorflow/tfjs');
// const poseDetection = require('@tensorflow-models/pose-detection');
// const fs = require('fs');
// const util = require('util');
// const readFile = util.promisify(fs.readFile);

// // 데이터 전처리 및 세부 조정
// const BATCH_SIZE = 6;
// const EPOCH = 700;
// const NUM_LAYER = 1;
// const n_CONFIDENCE = 0.3;
// const FRAME_LENGTH = 30;
// const videoPath = './video'; // 학습 비디오 데이터 경로

// const attention_dot = [];
// for (let n = 0; n < 17; n++) {
//     attention_dot.push(n);
// }

// const draw_line = [
//     [1, 0], [2, 0], [1, 3], [2, 4], [0, 5], [0, 6],
//     [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11],
//     [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
// ];

// async function detectPoseInFrames(frames) {
//     console.log("detect pose start");
//     const model = poseDetection.SupportedModels.MoveNet;
//     const detector = await poseDetection.createDetector(model);

//     const poses = [];
//     console.log("determine model done");
//     for (let frame of frames) {
//         console.log(frame);
//         try {
//             const pose = await detector.estimatePoses(frame);
//             if (pose.length > 0) {
//                 const keypoints = pose[0].keypoints;
//                 const keypointsFlipped = keypoints.map(keypoint => ({
//                     ...keypoint,
//                     x: frame.width - keypoint.x 
//                 }));
//                 poses.push({ keypoints, keypointsFlipped });
//             } else {
//                 console.log('No pose detected in this frame');
//             }
//         } catch (error) {
//             console.error('Error estimating pose:', error);
//         }
//     }
//     console.log('frame pose estimation done');
//     while (poses.length < FRAME_LENGTH) {
//         poses.push(poses[poses.length - 1]);
//     }
//     return poses;
// }

// async function get_skeleton(videoPath) {
//     console.log("get skeleton start");
//     const framesDir = './frames';
//     const frames = [];
//     // 1부터 30까지만, 추후 for (let file of files)로 변경
//     for (let i = 1; i <= 1; i++) {
//         const fileName = `frame-1-${i}.jpg`;
//         const filePath = `${framesDir}/${fileName}`;
//         const fileData = fs.readFileSync(filePath); // 파일 내용 읽기
//         const imageTensor = tf.node.decodeImage(fileData, 3);
//         frames.push(imageTensor);
//       }
//     const poses = await detectPoseInFrames(frames);

//     if (poses.length > 0) {
//         console.log('Poses extracted successfully:', poses.length, 'frames');
//     } else {
//         console.log('No poses extracted.');
//     }
//     return poses;
// }

// const raw_data = [];

// async function processVideoData(videoPath) {
//   console.log("processVideoData start", videoPath);
//       try {
//         const folds = fs.readdirSync(videoPath);
//         console.log("success open videopath");

//         for (let fold of folds) {
//             const videos = fs.readdirSync(`${videoPath}/${fold}`);

//             for (let videoName of videos) {
//                 if (parseInt(videoName.split('_')[3].substring(0, 2)) >= 30) {
//                     const label = videoName.split('_')[2] === 'normal' ? 0 : 1;
//                     console.log('before get raw_data, videopath = ', `${videoPath}/${fold}/${videoName}`);
//                     const poses = await get_skeleton(`${videoPath}/${fold}/${videoName}`);
                    
//                     if (poses.length > 0) {
//                         let seqListN = poses.slice(0, FRAME_LENGTH).map(pose => pose.keypoints);
//                         let seqListF = poses.slice(0, FRAME_LENGTH).map(pose => pose.keypointsFlipped);
//                         raw_data.push({ key: label, value: seqListN });
//                         raw_data.push({ key: label, value: seqListF });
//                         console.log('raw_data =', raw_data);
//                     } else {
//                         console.log('No poses found for video:', `${videoPath}/${fold}/${videoName}`);
//                     }
//                 }
//             }
//         }
//     } catch (error) {
//         console.error("An error occurred:", error);
//     }
//   }
  
//   // 데이터 처리 시작
// processVideoData(videoPath);

// // 모델링
// // 데이터를 텐서로 변환하는 함수
// function convertToTensor(data) {
//   const inputs = data.map(d => d.value);
//   const labels = data.map(d => d.key);
//   if (inputs.length > 0) {
//     const inputTensor = tf.tensor3d(inputs, [inputs.length, inputs[0].length, inputs[0][0].length]);
//     const labelTensor = tf.oneHot(tf.tensor1d(labels).toInt(), 2);
//     return { inputs: inputTensor, labels: labelTensor };
//   } else {
//     console.error('Input data is empty.');
//     return null; // 유효한 데이터가 없을 경우 null 반환
//   }
// }

// // 데이터 분할
// function splitData(data, splitRatio) {
//   const idx = Math.floor(data.length * splitRatio[0]);
//   const trainData = data.slice(0, idx);
//   const remainingData = data.slice(idx);
//   const valIdx = Math.floor(remainingData.length * splitRatio[1] / (splitRatio[1] + splitRatio[2]));
//   const valData = remainingData.slice(0, valIdx);
//   const testData = remainingData.slice(valIdx);
//   return [trainData, valData, testData];
// }

// function createModel(inputSize) {
//     const model = tf.sequential();

//     // 첫 번째 LSTM 레이어
//     model.add(tf.layers.lstm({
//         units: 128,
//         inputShape: [inputSize, 2],
//         returnSequences: true // 다음 레이어가 LSTM일 경우 true
//     }));
//     model.add(tf.layers.dropout(0.1)); // 드롭아웃 추가

//     // 두 번째 LSTM 레이어
//     model.add(tf.layers.lstm({
//         units: 256,
//         returnSequences: true
//     }));
//     model.add(tf.layers.dropout(0.1)); // 드롭아웃 추가

//     // 세 번째 LSTM 레이어
//     model.add(tf.layers.lstm({
//         units: 512,
//         returnSequences: true
//     }));
//     model.add(tf.layers.dropout(0.1)); // 드롭아웃 추가

//     // 네 번째 LSTM 레이어
//     model.add(tf.layers.lstm({
//         units: 256,
//         returnSequences: true
//     }));
//     model.add(tf.layers.dropout(0.1)); // 드롭아웃 추가

//     // 다섯 번째 LSTM 레이어
//     model.add(tf.layers.lstm({
//         units: 128,
//         returnSequences: true
//     }));
//     model.add(tf.layers.dropout(0.1)); // 드롭아웃 추가

//     // 여섯 번째 LSTM 레이어
//     model.add(tf.layers.lstm({
//         units: 64,
//         returnSequences: false // 마지막 레이어는 false
//     }));
//     model.add(tf.layers.dropout(0.1)); // 드롭아웃 추가

//     // 출력 레이어
//     model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
    
//     return model;
// }

  
// const inputSize = 17 * 2; // attention_dot의 길이 * 2
// const model = createModel(inputSize);
  
// model.compile({
//     optimizer: tf.train.adam(0.0001),
//     loss: 'categoricalCrossentropy',
//     metrics: ['accuracy']
// });

// async function trainModel(model, trainData, valData) {
//     const trainTensors = convertToTensor(trainData);
//     const valTensors = convertToTensor(valData);
//     console.log("trainData", trainData);
  
//     await model.fit(trainTensors.inputs, trainTensors.labels, {
//       epochs: 700,
//       batchSize: 6,
//       validationData: [valTensors.inputs, valTensors.labels],
//       callbacks: {
//         onEpochEnd: (epoch, logs) => {
//           console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`);
//         }
//       }
//     });
  
//     // 텐서 메모리 해제
//     trainTensors.inputs.dispose();
//     trainTensors.labels.dispose();
//     valTensors.inputs.dispose();
//     valTensors.labels.dispose();
//   }
  
// // 데이터 준비 및 모델 학습 시작
// const [trainData, valData, testData] = splitData(raw_data, [0.8, 0.1, 0.1]);
// // 모델 학습 함수 호출
// trainModel(model, trainData, valData);

// // 테스트 데이터를 사용하여 모델 성능 평가
// async function testModel(model, testData) {
//     const testTensors = convertToTensor(testData);
//     const evalOutput = model.evaluate(testTensors.inputs, testTensors.labels);

//     console.log('Test Accuracy:', evalOutput[1].dataSync()[0]);
//     console.log('Test Loss:', evalOutput[0].dataSync()[0]);

//     // 텐서 메모리 해제
//     testTensors.inputs.dispose();
//     testTensors.labels.dispose();
// }

// testModel(model, testData);


import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

export async function loadModel() {
    const model = poseDetection.SupportedModels.MoveNet;
    const detector = await poseDetection.createDetector(model);
    return detector;
}

export async function estimatePoses(detector, imageFrames) {
    const poses = [];
    for (const image of imageFrames) {
        const pose = await detector.estimatePoses(image);
        poses.push(pose);
    }
    return poses;
}