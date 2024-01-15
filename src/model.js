const frameLength = 3;
let currentFrameNumber = 0;
const fs = require('fs');
const { spawn } = require('child_process');
const tf = require('@tensorflow/tfjs-node');
const poseDetection = require('@tensorflow-models/pose-detection');
const util = require('util');
const path = require('path');
const readFile = util.promisify(fs.readFile);
const { moveNet } = poseDetection.SupportedModels;
const videoPath = './video'; // 비디오 경로
const framesDir = './frames'; // 프레임 경로

// 관심 있는 포즈 keypoints 배열
const attention_dot = [];
for (let n = 0; n < 17; n++) {
    attention_dot.push(n);
}

let raw_data = [];

function checkFileExists(filePath) {
  return fs.existsSync(filePath);
}

function extractFramesFromVideo(videoFilePath, frameCount) {
  return new Promise((resolve, reject) => {
    if (!checkFileExists(videoFilePath)) {
      console.log('File does not exist');
      reject(new Error('Video file does not exist'));
      return;
    }

    currentFrameNumber += 1;

    const ffmpeg = spawn('ffmpeg', [
      '-i', videoFilePath,
      '-vf', `fps=${frameCount}`,
      '-qscale:v', '2',
      path.join(framesDir, `frame-${currentFrameNumber}-%d.jpg`),
    ]);

    ffmpeg.on('error', (err) => {
      console.error('Failed to start ffmpeg:', err);
      reject(err);
    });
    
    ffmpeg.on('exit', (code) => {
      if (code === 0) {
        const files = fs.readdirSync(framesDir).filter(file => {
          const regex = new RegExp(`frame-${currentFrameNumber}-(\\d+).jpg`);
          return file.match(regex);
        });
        resolve(files.length);
      } else {
        console.error(`ffmpeg exited with code: ${code}`);
        reject(new Error(`ffmpeg exited with code: ${code}`));
      }
    });
  });
}
  
async function getSkeleton(videoFilePath) {
  const frameCount = await extractFramesFromVideo(videoFilePath, frameLength);
  const xyListList = [];
  const xyListListFlip = [];

  const detectorConfig = {
    modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
    enableTracking: true,
    trackerType: poseDetection.TrackerType.BoundingBox,
  };

  const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet,detectorConfig);

  for (let i = 1; i <= frameCount; i++) {
    const fileName = `frame-${currentFrameNumber}-${i}.jpg`;
    const filePath = `${framesDir}/${fileName}`;
    const image = await readFile(filePath);
    const imageTensor = tf.node.decodeImage(image, 3);
    const poses = await detector.estimatePoses(imageTensor);

    if (poses.length > 0 && poses[0]) {
      const keypoints = poses[0].keypoints;
      xyListList.push(keypoints);
  
      const keypointsFlipped = keypoints.map(keypoint => ({
        ...keypoint,
        x: imageTensor.shape[1] - keypoint.x 
      }));
      xyListListFlip.push(keypointsFlipped);
    } else {
      console.log(`No poses detected in ${fileName}`);
    }
  }
  return { xyListList, xyListListFlip };
}

function shuffle(array) {
    let currentIndex = array.length, randomIndex, temporaryValue;
  
    while (currentIndex !== 0) {
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      temporaryValue = array[currentIndex];
      array[currentIndex] = array[randomIndex];
      array[randomIndex] = temporaryValue;
    }
  
    return array;
  }
  
  async function collectData() {
    console.log('collect data');
    const videoPaths = ['abnormal', 'normal'];
    for (const folder of videoPaths) {
      const videos = fs.readdirSync(`${videoPath}/${folder}`);
      for (const video of videos) {
        const videoNameSplit = video.split('_');
        const label = videoNameSplit[2] === 'normal' ? 0 : 1;
        const videoFilePath = `${videoPath}/${folder}/${video}`;
        const { xyListList, xyListListFlip } = await getSkeleton(videoFilePath);
        const seqListN = xyListList.slice(0, frameLength);
        const seqListF = xyListListFlip.slice(0, frameLength);

        raw_data.push({ key: label, value: seqListN });
        raw_data.push({ key: label, value: seqListF });
      }
    }
  
    console.log('raw data');
    console.table(raw_data);
    raw_data = shuffle(raw_data);
    // raw_data 배열의 각 요소에 대해 실행
    raw_data.forEach((item, index) => {
      console.log(`Item at index ${index} with key ${item.key}:`);
      item.value.forEach((innerArray, innerIndex) => {
        console.log(`Table for inner array at index ${innerIndex}:`);
        console.table(innerArray);
      });
    });
    

  
    let normalDataCount = 0;
    let abnormalDataCount = 0;
  
    for (const data of raw_data) {
      if (data.key === 0) {
        normalDataCount++;
      } else {
        abnormalDataCount++;
      }
    }
  
    console.log('normal data:', normalDataCount/2, '| abnormal data:', abnormalDataCount/2);
  }
  
  // 데이터 수집 함수 호출
  collectData();
  