import { loadModel, estimatePoses } from './model.js';
import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { train } from '@tensorflow/tfjs-core';

async function runPoseEstimation() {
  // 모델 로드
    const detector = await loadModel();

    // 이미지 프레임과 라벨을 포함하는 배열
    const imageFrames = [];
    const labels = [];
    const framesDir = './frames';

    for (let j = 1; j <= 2; j++) { //temp, data 수에 따라 변경
        const label = j <= 1 ? 0 : 1; // abnormal: 0, normal: 1
        for (let i = 1; i <= 300; i++) {
            const fileName = `frame-${j}-${i}.jpg`;
            const filePath = `${framesDir}/${fileName}`;
            const fileData = fs.readFileSync(filePath);
            const imageTensor = tf.node.decodeImage(fileData, 3);
            imageFrames.push(imageTensor);
            labels.push(label);
        }
    }

    // 포즈 추정 실행
    const poses = await estimatePoses(detector, imageFrames);
    imageFrames.forEach(tensor => tensor.dispose());

    // 포즈 데이터 처리
    const processedPoses = processPoses(poses);

    // 데이터셋 생성 및 분할
    console.log("make combinedData");
    console.log('processedPoses[0] : ', processedPoses[0]);
    const combinedData = processedPoses.map((pose, index) => ({ x: pose, y: labels[index] }));
    console.log("combinedData[0] = ", combinedData[0]);
    console.log("combinedData[0] = ", combinedData[301]);
    if (combinedData.length === 0) {
      console.error('No data to split');
      return;
    }
    const { trainData, valData, testData } = splitData(combinedData, 0.6, 0.2);

    // LSTM 모델 생성 및 컴파일
    const model = createModel(); // inputSize 수정 필요한 경우 반영
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'sparseCategoricalCrossentropy', // 수정된 손실 함수
        metrics: ['accuracy']
    });

    // 모델 훈련 및 평가
    await trainModel(model, trainData, valData);
    await testModel(model, testData);
}

function processPoses(poses) {
  console.log('len poses', poses.length);
  console.log('len poses[0]', poses[0].length);
  console.log('poses[0][0]', poses[0][0]);
  console.log('poses[0].keypoints', poses[0].keypoints);
  console.log('poses[0][0].keypoints', poses[0][0].keypoints);

  return poses.map(poseArray => {
    const defaultKeypoints = new Array(17).fill([1, 1, 0]);
    // 각 포즈 배열의 첫 번째 요소에서 keypoints를 가져옴
    if (!poseArray[0] || !poseArray[0].keypoints) {
        console.error('Invalid pose data:', poseArray);
        return defaultKeypoints; // 또는 적절한 기본값 반환
    }

    const keypoints = poseArray[0].keypoints;

    // 유효한 keypoints 데이터 처리
    return keypoints.map(keypoint => {
        return [keypoint.x, keypoint.y, keypoint.score];
    });
  });
}

function createModel() {
    const model = tf.sequential();
    // 예시: 첫 번째 LSTM 레이어
    model.add(tf.layers.lstm({
      units: 128,
      inputShape: [17, 3], // 시퀀스 길이는 가변적일 수 있음
      returnSequences: true
    }));

    model.add(tf.layers.lstm({
        units: 256,
        returnSequences: true
      }));

    model.add(tf.layers.dropout(0.1));

    model.add(tf.layers.lstm({
        units: 128,
        returnSequences: true
      }));

    model.add(tf.layers.lstm({
        units: 64,
        returnSequences: true
      }));

    model.add(tf.layers.dropout(0.1));

    model.add(tf.layers.lstm({
        units: 32,
        returnSequences: false
      }));
    // 출력 레이어
    model.add(tf.layers.dense({
        units: 2, 
        activation: 'softmax'
        }));

    return model;
}

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function splitData(data, trainSize, valSize) {
    if (!data || data.length === 0) {
      console.error('No data provided or data is empty');
      return { trainData: [], valData: [], testData: [] };
    }
    console.log('data length', data.length);
    console.log('trainsize', trainSize);

    const shuffledIndices = shuffleArray([...Array(data.length).keys()]);
    const numTrainSamples = Math.floor(trainSize * data.length);
    const numValSamples = Math.floor(valSize * data.length);

    const trainIdx = shuffledIndices.slice(0, numTrainSamples);
    console.log('Shuffled Train Data:', extractData(data, trainIdx));
    const valIdx = shuffledIndices.slice(numTrainSamples, numTrainSamples + numValSamples);
    const testIdx = shuffledIndices.slice(numTrainSamples + numValSamples);

    return {
        trainData: extractData(data, trainIdx),
        valData: extractData(data, valIdx),
        testData: extractData(data, testIdx)
    };
}

function extractData(data, indices) {
    return indices.map(idx => data[idx]);
}

async function trainModel(model, trainData, valData) {
    function augmentData(data) {
        return data.map(item => ({
            x: item.x.map(keypoint => [1080 - keypoint[0], keypoint[1], keypoint[2]]),
            y: item.y
        }));
    }
    
    // 원본 데이터
    const originalTrainData = trainData;
    const originalValData = valData;
    
    // 데이터 증강 수행
    const augmentedTrainData = augmentData(originalTrainData);
    const augmentedValData = augmentData(originalValData);
    
    // 원본 데이터와 증강된 데이터 결합
    const combinedTrainData = [...originalTrainData, ...augmentedTrainData];
    const combinedValData = [...originalValData, ...augmentedValData];
    
    // 새로운 데이터셋으로 값 추출
    const trainXValues = combinedTrainData.map(data => data.x);
    const trainYValues = combinedTrainData.map(data => data.y);
    const valXValues = combinedValData.map(data => data.x);
    const valYValues = combinedValData.map(data => data.y);

    // 추출된 값들을 텐서로 변환
    const trainX = tf.tensor3d(trainXValues, [trainXValues.length, 17, 3]);
    const trainY = tf.tensor1d(trainYValues, 'float32');
    const valX = tf.tensor3d(valXValues, [valXValues.length, 17, 3]);
    const valY = tf.tensor1d(valYValues, 'float32');

    // 모델 훈련
    await model.fit(trainX, trainY, {
        epochs: 60, // 에포크 수
        batchSize: 12, // 배치 크기
        validationData: [valX, valY],
        callbacks: {
          onEpochEnd: (epoch, logs) => {
              console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}, Validation Loss = ${logs.val_loss}, Validation Accuracy = ${logs.val_acc}`);
          },
          earlyStopping: tf.callbacks.earlyStopping({ patience: 10 }) // 조기 종료
      }
    });

    // 텐서 메모리 해제
    trainX.dispose();
    trainY.dispose();
    valX.dispose();
    valY.dispose();

    await model.save('file://./model');
}

async function testModel(model, testData) {
    const testXValues = testData.map(data => data.x);
    const testYValues = testData.map(data => data.y);
    // 테스트 데이터를 텐서로 변환
    const testX = tf.tensor3d(testXValues, [testXValues.length, 17, 3]);
    const testY = tf.tensor1d(testYValues, 'float32');
    console.log('Test Y values:', await testY.array());

    // 모델 평가
    const evalResult = model.evaluate(testX, testY);

    // 평가 결과 출력
    const testLoss = evalResult[0].dataSync()[0];
    const testAccuracy = evalResult[1].dataSync()[0];
    console.log(`Test Loss: ${testLoss}`);
    console.log(`Test Accuracy: ${testAccuracy}`);

    // 텐서 메모리 해제
    testX.dispose();
    testY.dispose();
}

runPoseEstimation();