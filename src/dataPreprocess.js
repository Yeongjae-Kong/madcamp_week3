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
const attentionDot = [];
for (let n = 0; n < 17; n++) {
    attentionDot.push(n);
}

let raw_data = [];

function checkFileExists(filePath) {
  return fs.existsSync(filePath);
}

function extractFramesFromVideo(videoFilePath, frameCount) {
  return new Promise((resolve, reject) => {
    console.log('extract frames');
    console.log(videoFilePath);

    if (!checkFileExists(videoFilePath)) {
      console.log('File does not exist');
      reject(new Error('Video file does not exist'));
      return;
    }

    // 'frames' 디렉토리 내의 현재 파일들을 모두 삭제 (새로운 프레임 추출 전에)
    /*
    fs.readdirSync(framesDir).forEach((file) => {
      fs.unlinkSync(path.join(framesDir, file));
    });
    */
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
        const files = fs.readdirSync(framesDir).filter(file => file.endsWith('.jpg'));
        resolve(files.length); // 생성된 파일 개수를 반환
      } else {
        console.error(`ffmpeg exited with code: ${code}`);
        reject(new Error(`ffmpeg exited with code: ${code}`));
      }
    });
  });
}


async function loadImage(framePath) {
  console.log('load image');
  const data = await readFile(framePath);
  const imageTensor = tf.node.decodeImage(new Uint8Array(data), 3);
  return imageTensor;
}
  
async function getSkeleton(videoFilePath) {
  console.log('get skeleton');
  console.log(videoFilePath);
  const frameCount=0;
  frameCount == await extractFramesFromVideo(videoFilePath, frameLength);
  const xyListList = [];
  const xyListListFlip = [];

  const detectorConfig = {
    modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
    enableTracking: true,
    trackerType: poseDetection.TrackerType.BoundingBox,
  };

  const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
  for (let i = 1; i < frameCount; i++) {
    const framePath = `frames/frame-${currentFrameNumber}-${i}.jpg`;
    console.log(framePath);
    const image = await loadImage(framePath);
    const poses = await detector.estimatePoses(image);
    console.log('poses : ', poses);

    for (const pose of poses) {
      if (!pose.length) {
        console.log('No poses detected');
        continue;
      }
        
      console.log('pose detected');
      console.log('pose : ', pose);
      const keypoints = pose.keypoints.filter((_, index) => attentionDot.includes(index));
      console.log('keypoints : ', keypoints);
      const xyList = keypoints.map(kp => [kp.x, kp.y]);
      const xyListFlip = keypoints.map(kp => [1 - kp.x, kp.y]);

      xyListList.push(xyList.flat());
      xyListListFlip.push(xyListFlip.flat());
    }

    while (xyListList.length < frameLength){
      xyListList.push(xyListList[xyListList.length - 1]);
      xyListListFlip.push(xyListListFlip[xyListListFlip.length - 1]);
    }
  }
  console.log('xyListList :', xyListList);
  console.log('xyListListFlip :', xyListListFlip);
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
      console.log('folder :', folder);
      const videos = fs.readdirSync(`${videoPath}/${folder}`);
      console.log('videos', videos);
      for (const video of videos) {
        console.log('video : ', video);
        const videoNameSplit = video.split('_');
        console.log(videoNameSplit);
        const label = videoNameSplit[2] === 'normal' ? 0 : 1;
        const videoFilePath = `${videoPath}/${folder}/${video}`;
        console.log(videoFilePath);
        const { xyListList, xyListListFlip } = await getSkeleton(videoFilePath);
        const seqListN = xyListList.slice(0, frameLength);
        const seqListF = xyListListFlip.slice(0, frameLength);
    
        raw_data.push({ key: label, value: seqListN });
        raw_data.push({ key: label, value: seqListF });
      }
    }
  
    raw_data = shuffle(raw_data);
    console.log('raw_data : ', raw_data);
  
    let normalDataCount = 0;
    let abnormalDataCount = 0;
  
    for (const data of raw_data) {
      if (data.key === 0) {
        normalDataCount++;
      } else {
        abnormalDataCount++;
      }
    }
  
    console.log('normal data:', normalDataCount, '| abnormal data:', abnormalDataCount);
  }
  
  // 데이터 수집 함수 호출
  collectData();
  