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