# Project 6: 3D Pose Estimation and Objectron for object detection
### Refer to proj6_code >> proj6.ipynb for details
## Brief
- Sections:
    - Part 1: 3D bounding box detection on 2D images
    - Part 2: Estimation of world coordinates of the camera 
    - Part 3: Human Pose Detection
    - Part 4: Projection of 2D pose estimation to 3D world coordinates 
    - Part 5: Intersection between Pose and Objectron
        

## Overview
The goal of this project is to let you experience working with state-of-the-art libraries and real-time processing of video. We will be working with [MediaPipe](https://google.github.io/mediapipe/), which offers customizable Machine Learning solutions for live media. Specifically, we will use: [pose estimation](https://google.github.io/mediapipe/solutions/pose), which estimates the pose of a person in real time; and [objectron](https://google.github.io/mediapipe/solutions/objectron.html), which is a 3D object detector from 2D images. We will try to identify an object that is being touched by a person with these tools.

## 3D bounding box detection on 2D images

We're going to use mediapipe's [objectron](https://google.github.io/mediapipe/solutions/objectron) library based on this [paper](https://arxiv.org/pdf/2003.03522.pdf) to detect 3D bounding boxes of chairs in images. An example is shown in the figure below.

<img src="https://google.github.io/mediapipe/images/mobile/objectron_chair_android_gpu.gif" width="150"/>
<center>3D Objectron example.</center>



## Detect human pose from images



<img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_android_gpu_small.gif" width="150"/>
<center>Pose detection example.</center>
