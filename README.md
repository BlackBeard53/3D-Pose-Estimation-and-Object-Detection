# Project 6: 3D Pose Estimation and Objectron for object detection

## Brief
- Deliverables:
    - Report: Dec 7th, 11:59 PM.
    - Code: Dec 7th, 11:59 PM.
- Hand-in: through Gradescope
- Sections:
    - Part 1: 3D bounding box detection on 2D images
    - Part 2: Estimation of world coordinates of the camera **(no coding required)**
    - Part 3: Human Pose Detection
    - Part 4: Projection of 2D pose estimation to 3D world coordinates **(no coding required)**
    - Part 5: Intersection between Pose and Objectron
    - Part 6: (All)Extra Credit: your own image
    - Part 7: (Grad) Extra Credit: intersection detection in a video
        

## Overview
The goal of this project is to let you experience working with state-of-the-art libraries and real-time processing of video. We will be working with [MediaPipe](https://google.github.io/mediapipe/), which offers customizable Machine Learning solutions for live media. Specifically, we will use: [pose estimation](https://google.github.io/mediapipe/solutions/pose), which estimates the pose of a person in real time; and [objectron](https://google.github.io/mediapipe/solutions/objectron.html), which is a 3D object detector from 2D images. We will try to identify an object that is being touched by a person with these tools.

## Imports


```python
%reload_ext autoreload
%autoreload 2

import os
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import cv2

from proj6_unit_tests import test_base
from proj6_unit_tests import test_my_objectron
from proj6_unit_tests import test_pose_estimate
from proj6_unit_tests import test_intersection
from proj6_unit_tests import test_utils
from proj6_code import student_code
from proj6_code import utils
```

## Part 1: 3D bounding box detection on 2D images

We're going to use mediapipe's [objectron](https://google.github.io/mediapipe/solutions/objectron) library based on this [paper](https://arxiv.org/pdf/2003.03522.pdf) to detect 3D bounding boxes of chairs in images. An example is shown in the figure below.

<img src="https://google.github.io/mediapipe/images/mobile/objectron_chair_android_gpu.gif" width="150"/>
<center>3D Objectron example.</center>

Basically, there is an encoder and a decoder for the detection part. The encoder takes an image as input, analyzes it, and gives some useful information in a certain form. Then the decoder takes these intermediate information as input, translates them into things we want (i.e., 8 vertices' 2D coordinates).
The encoder is trained as a neural network, whose weight file is provided and can be read by Tensorflow (which is the `inference()` function in `my_objectron.py`). The decoder part is implemented as the `decode()` function in `my_objectron.py`. **You should read the paper and figure out what is the intermediate information to finish the coding work, otherwise you will not understand how to write the code for this section or answer the report reflection questions.**

Most of the code is already implemented for you based on [this](https://github.com/yas-sim/objectron-3d-object-detection-openvino) Github repo.


```python
# Reading test image
student_test_img='../data/10.jpg'
img = cv2.imread(student_test_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
```


    
![png](proj6_files/proj6_6_0.png)
    


**TODO 1**:  finish `detect_3d_box()` in `student_code.py`, then run the following cell.


```python
bounding_boxes_chair_2d, annotated_img = student_code.detect_3d_box(student_test_img)
plt.imshow(annotated_img)
plt.show()
```

    [[1, 3, 640, 480]] [[1, 16, 40, 30], [1, 1, 40, 30]]
    0.9636403322219849
    


    
![png](proj6_files/proj6_8_1.png)
    



```python
print("Testing your objectron detection: ", test_base.verify(test_my_objectron.test_my_objectron))
```

    [[1, 3, 640, 480]] [[1, 16, 40, 30], [1, 1, 40, 30]]
    0.9799638986587524
    Testing your objectron detection:  [32m"Correct"[0m
    

## Part 2: Estimation of world coordinates of the camera

### 2.1 Establish the world coordinate frame


```python
# Read test image
img_index = cv2.imread('../data/world_frame.png')
img_index = cv2.cvtColor(img_index, cv2.COLOR_BGR2RGB)
plt.imshow(img_index)
plt.show()
```


    
![png](proj6_files/proj6_12_0.png)
    


Similar to project 5, we want to estimate the intrinsic camera matrix, but this time we will provide an initial estimation for a cube, giving 8 vertices in 3D coordinate. In this coordinate system, we define 0 as the `origin`, 0-1 as the `x-axis`, 0-4 as the `y-axis`, and 0-2 as the `x-axis`.

Then, we need to estimate the dimensions of the chair to calculate the location of the coordinates in the world frame.

**NOTE**: This part has already been implemented for you in `get_world_vertices()` in `utils.py`, feel free to check it out.


```python
# You need to measure the size of chair at first, and set their values there. 
# For example, the example chair's size is 0.4m * 0.4m *1.0m
size_x = 0.4
size_y = 0.4
size_z = 1.0
vertices_world = student_code.get_world_vertices(size_x, size_y, size_z)

initial_box_points_3d = vertices_world
print("Testing your get_world_vertices: ", test_base.verify(test_utils.test_get_world_vertices))
```

    Testing your get_world_vertices:  [32m"Correct"[0m
    

### 2.2 Estimate intrinsic matrix of your camera

Different from project 5, we can use a convenient function from [OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a) to help calibrate our camera, by taking pictures of a chessboard from various angles (better > 10). You can refer to the pictures in the *../data/cali2/example* folder.

If you want to experiment with your own camera/images, feel free to play around with the `Checkerboard-A4-35mm-7x4.pdf` in the `data` folder. There are also two video demoes about how to do camera calibration: [Example1-Camera Calibration](https://www.youtube.com/watch?v=HNfPbw-1e_w), this is for explaining the checkerboard, but you don't need to measure the length of grid on the checkboard. [Example2-Camera Calibration](https://www.youtube.com/watch?v=v7jutAmWJVQ), you need to take pictures as 1:30-1:50 in the video, you can either move the checkboard like in the video or move your camera.

The `calibrate()` function in `utils.py` requires the number of grids. If you use the checkerboard provided in the `data` folder, you can simply use the default parameters. If you want to try your own checkerboard, please note that the part used for calibration is two subgrids smaller than the actual in both x and y direction. For example, the example picture, the checkerboard has 5 x 8 subgrids, then the part used for calibration should be 3 x 6, thus the number of vertices should be 4 x 7 (corresponding to m and n in `calibrate` function).

If you are interested in what is the mechanism in calibrating a camera with checkerboard, you can refer to [WiKi- Camera Calibration](https://en.wikipedia.org/wiki/Chessboard_detection#:~:text=Chessboard%20camera%20calibration,-A%20classical%20problem&text=Chessboards%20are%20often%20used%20during,interest%20points%20in%20an%20image.) and [OpenCV Doc](https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html)

Important tips:
- Take pictures from different angles, and avoid very similar pictures
- Use the pictures of the same resolution for all the tasks; Usually, phone camera uses different zoom for taking pictures and record videos. 
- By default, phone cameras usually adjust focal length automatically. You need to find a way to fix that.

**NOTE**: `calibrate()` is implemented for you in `utils.py`, feel free to check it out.


```python
from proj6_code.utils import calibrate

path2 = '../data/cali2/example2/' # update the path to where you save the pictures
m = 5 # m is the vertice number in x direction
n = 7 # n is the vertice number in y direction

K = calibrate(path2)
```

    ../data/cali2/example2/*.jpg
    [[1.11573975e+03 0.00000000e+00 7.22275004e+02]
     [0.00000000e+00 1.06794751e+03 5.31184727e+02]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    

### 2.3 Estimate camera pose

After you get the 8 vertices of the box in 2D coordinate and the camera intrinsic matrix, we can recover the projection matrix by method called perspective-n-point method. Perspective-n-Point is a typical problem in 3D geometry estimating the pose of a calibrated camera given `n` 3D points in world and their corresponding 2D projected points on image frame. There are several way to get the projection matrix given the 2D-3D correspondence.

* **Direct Linear Transform (DLT)**, we use this method in project 5 to solve the projection matrix. Remember in previous project, we mentioned P has 11 DoF, thus we need at least 6 correspondencse to get the P. But this method has limitation on accuracy because it uses SVD to solve the P as a 12x1 vector ignoring the inner connection of all 12 parameters.

* **Perspective-3-Point (P3P)**, is the minimal form of PnP problem. This method uses the law of cosines to give extra constraints of the 6 triangles (3 similar triagle pairs) formed by the 3 correspondence and camera center. This method has disadvantages of limitation on information given only by three correspondence. Besides, when there are noises in the three correspondence, it is not robust.

* **Perspective-n-Point (PnP)**, is a often used pose estimation method which is more robust. Use the projection model, we can define the reprojection error, which is the difference between the projected 3D points and their matched 2D points, here the intrinsic matrix K should be known. Then given an initial estimation of camera pose, you can solve the problem by using nonlinear optimization. Give some perturbation to the intial estimation and then calculate the jacobian and hessian matrix, then you can get a update on your intial estimation. Repeat the process untill the error converges, you will get a well optimized camera pose. You can also refer to the [Wiki-Perspective-n-Point](https://en.wikipedia.org/wiki/Perspective-n-Point) to know more.

**Note:** The rotation and translation returned by cv2.solvePnP is different with what we deal with in project4. In project4, we use the notations below:

- ${}^wR_c$: for rotation of the camera in the world coordinate frame
- ${}^wt_c$: for translation of the camera in the world coordinate frame, which is the **camera_center**
- ${}^cR_w$: ${}^cR_w = {{}^wR_c}^T$

In project 5, The projection matrix **P**

\begin{align}
\mathbf
{P}=  \mathbf{K} \: {}^w \mathbf{R}_c^\top [ \mathbf{I}\;|\; -{}^w \mathbf{t}_c ] =
\begin{bmatrix}
    \alpha & s & u_0 \\
    0 & \beta & v_0 \\
    0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
r_{11} & r_{21} & r_{31} \\
r_{12} & r_{22} & r_{32} \\
r_{13} & r_{23} & r_{33}
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & -t_x \\
0 & 1 & 0 & -t_y \\
0 & 0 & 1 & -t_z
\end{bmatrix}
\end{align}

And this transform from world coordinate to camera coordinate:

$$\begin{bmatrix}X_c\\Y_c\\Z_c\\1\end{bmatrix} = \begin{bmatrix}\mathbf{{}^wR_c^T} & -\mathbf{{}^wR_c^T}\ {}_w\mathbf{t}_c\\0^\top  & 1\\\end{bmatrix}\begin{bmatrix}X_w\\Y_w\\Z_w\\1\end{bmatrix}$$

**In opencv**, The translation $tvec$ and rotation part $rvec$ returned by `cv2.solvePnP()` is 

\begin{align}
tvec = \begin{bmatrix} t_1 & t_2 & t_3 \end{bmatrix} ^T
\end{align}

\begin{align}
\begin{bmatrix}
r_{11} & r_{21} & r_{31} \\
r_{12} & r_{22} & r_{32} \\
r_{13} & r_{23} & r_{33}
\end{bmatrix} = Rodrigues Transform (rvec)
\end{align}

Another form to perform the transformation from world coordinate to camera coordinate is use the ${}^c{T}_w$ matrix,

$$\begin{bmatrix}X_c\\Y_c\\Z_c\\1\end{bmatrix} = {}^c{T}_w \begin{bmatrix}X_w\\Y_w\\Z_w\\1\end{bmatrix}$$

\begin{align}
\mathbf
{}^c{T}_w=
\begin{bmatrix}
r_{11} & r_{21} & r_{31} & t_1 \\
r_{12} & r_{22} & r_{32} & t_2 \\
r_{13} & r_{23} & r_{33} & t_3 \\
0 & 0 & 0 & 1
\end{bmatrix} = \begin{bmatrix}\mathbf{{}^wR_c^T} & -\mathbf{{}^wR_c^T}\ {}^w\mathbf{t}_c\\0^\top  & 1\\\end{bmatrix}
\end{align}

Therefore, to get the ${}^wR_c$, projection matrix $\textbf{P}$ and camera center ${}^wt_c$ from `tvec` and `rvec`, you need to transform it by:

$$ {{}^wR_c}^T = Rodrigues Transform (rvec) $$

$$ P = K * \begin{bmatrix} {{}^wR_c}^T & tvec \end{bmatrix} $$

$$ {}^wt_c = - {{}^wR_c} * tvec $$

In this project, you do not need to implement the PnP method yourself, you can call the function from `cv2.solvePnP`. After you get the tvec and rvec from cv2.solvePnP, you need to return the ${}^w \mathbf{R}_c^\top$, the camera center in world coordinate and the projection matrix $\textbf{P}$.


**NOTE**: `perspective_n_points()` is implemented for you in `utils.py`, feel free to check it out.


```python
from proj6_code.utils import perspective_n_points

bounding_boxes = bounding_boxes_chair_2d
height = annotated_img.shape[0]
width = annotated_img.shape[1]

box_points_2d = np.array(bounding_boxes)
box_points_2d[:, 0] *= width
box_points_2d[:, 1] *= height

wRc_T, camera_center, P = perspective_n_points(initial_box_points_3d, box_points_2d, K)
```

Then you can visualize the world coordinate of the box object and camera pose to **check whether the results gotten by the PnP method** and **your P, ${}^wt_c$** is correct.


```python
from proj6_code.utils import plot_box_and_camera
plot_box_and_camera(initial_box_points_3d, camera_center, wRc_T.T)
```

    The camera center is at: 
     [[1.91410572]
     [0.44133695]
     [1.40106129]]
    


    
![png](proj6_files/proj6_26_1.png)
    


## Part 3: Human Pose Estimation
In this part, you will use [mediapipe](https://google.github.io/mediapipe/) to do pose estimate from videos. As an important task of computer vision, human Pose Estimation is defined as the problem of localization of human joints (also known as keypoints - elbows, wrists, etc) in images or videos. It has great application in many real-life situations, such as action recognition, AR/VR, animation, gaming, etc.

The official [document](https://google.github.io/mediapipe/solutions/pose#overview) on pose estimation also provides some more detailed explanation, examples, and tutorial about how to use the python version of pose estimation can be found [there](https://google.github.io/mediapipe/solutions/pose#python).


### Detect human pose from images

**TO-DO 2**:  finish `hand_pose_img()` in `student_code.py`


<img src="https://google.github.io/mediapipe/images/mobile/pose_tracking_android_gpu_small.gif" width="150"/>
<center>Pose detection example.</center>


```python
player_test_img='../data/player1.jpg'

player_land_mark, annotated_image = student_code.hand_pose_img(player_test_img)
print('Detected landmark numbers: ', len(player_land_mark))
utils.imshow1(annotated_image)
```

    Detected landmark numbers:  33
    


    
![png](proj6_files/proj6_29_1.png)
    



```python
# Although you may say this is actually the right toe physically, 
# we can call it left toe for now in this notebook, to be consistent with 
# the name convention used in mediapipe's pose estimation.

left_toe = player_land_mark[32]
print(left_toe)
```

    [119.43694353 627.38676071]
    


```python
print("Testing your pose estimation: ", test_base.verify(test_pose_estimate.test_pose_estimate))

```

    Testing your pose estimation:  [32m"Correct"[0m
    

## Part 4: Projection of 2D pose estimation to 3D world coordinates
In this section, you're going to project the 2D human pose estimation onto the 3D coordinates in the world frame (or to be more specific, the chair-center frame). (to do this we will have to tell how far the person is from the camera)

The projection function is pretty similar to the one you implement in project 5, except that we project from 2D to 3D instead of 3D to 2D. We use camera matrix $P \in R^{3×4}$ as a projective mapping from world (3D) to pixel (2D) coordinates defined up to a scale.

\begin{align}
z\begin{bmatrix}
x_p\\
y_p\\
1
\end{bmatrix}
=
\mathbf
{P}
\begin{bmatrix}
x_w\\
y_w\\
z_w\\
1
\end{bmatrix}
\end{align}

The camera matrix can also be decomposed into intrinsic parameters K and extrinsic parameters ${}^w \mathbf{R}_c, \: {}^w \mathbf{t}_c$.

$\mathbf{P} = \mathbf{K} \: {}^w \mathbf{R}_c^\top [\mathbf{I}\;|\; -{}^w \mathbf{t}_c].$

\begin{align}
\mathbf
{P}=
\begin{bmatrix}
    \alpha & s & u_0 \\
    0 & \beta & v_0 \\
    0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
r_{11} & r_{21} & r_{31} \\
r_{12} & r_{22} & r_{32} \\
r_{13} & r_{23} & r_{33}
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & -t_x \\
0 & 1 & 0 & -t_y \\
0 & 0 & 1 & -t_z
\end{bmatrix}
\end{align}

Generally, we cannot fully recover the 3D coordinates from a 2D feature points because the depth information is lost. As such, you're asked to provide the depth information (i.e., $z$ in the projection equation above.) The physical meaning of depth is the physical distance between human in the picture and the camera in meters. In the test files, the depth information used is provided. In the next sections, you need to measure the depth information on your own to make your customized videos work.


**NOTE**: `projection_2d_to_3d()` is implemented for you in `utils.py`, feel free to check it out.

To make things clear, in this project, you have to work with three coordinate frames in total. The world frame (which is defined as the chair frame, 3D), the camera-centric frame (3D), and the image frame (2D). In this sectrion, you will project feature points in 2D image frame to the 3D chair frame.


```python
P = K.dot(wRc_T.dot( np.hstack((np.eye(3),-1.0*camera_center)) ))
```


```python
depth = 1.91

student_land_mark, _ = student_code.hand_pose_img(student_test_img)
pose3d_landmark = student_code.projection_2d_to_3d(P, depth, student_land_mark)

left_hand = pose3d_landmark[22]
print("Your 3D pose landmark of the left hand is ", left_hand)

print("Testing your 3d human pose estimate: ", test_base.verify(test_pose_estimate.test_projection_2d_to_3d))
```

    Your 3D pose landmark of the left hand is  [0.01800057 0.25497359 0.78408101]
    Testing your 3d human pose estimate:  [32m"Correct"[0m
    


```python
# Test if you can project the 0 vertice back correctly
P.dot(np.array([0,0,0,1]).T)/P.dot(np.array([0,0,0,1]).T)[2]
```




    array([3.26594876e+02, 1.16785333e+03, 1.00000000e+00])



## Part 5: Intersection between Pose and Objectron



After detecting both the 3D coordinates of human pose and the 3D coordinates of 8 vertices of the chair, we can now finally apply our trick now. We will detect whether the hand is in the bounding box of the chair in this section (And you can try to detect other parts too). If so, then we would change the color of the bounding box to show it. (Black if no detection, a bight color if detection) Such kind of detection would be pretty useful for situations like collision check and obstacles avoidance.


**TO-DO 3**: finish `check_hand_inside_bounding_box()` in `student_code.py`


```python
annotated_img = student_code.draw_box_intersection(img, left_hand, vertices_world, bounding_boxes_chair_2d)

plt.imshow(annotated_img)
plt.show()

```

    [0. 0. 0. 1.]
    [0.4 0.4 1.  1. ]
    Check succeed!
    (1440, 1080, 3)
    


    
![png](proj6_files/proj6_40_1.png)
    



    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_1788/3609213677.py in <module>
          6 
          7 from proj6_code import utils
    ----> 8 from utils import check_hand_inside_bounding_box
    

    ImportError: cannot import name 'check_hand_inside_bounding_box' from 'utils' (C:\Users\kbipi\proj6_release\proj6_release\proj6_code\utils.py)



```python
print('Test for intersection checking:', test_base.verify(test_intersection.test_check_hand_inside_bounding_box))
```

    [0. 0. 0.]
    [3. 2. 1.]
    [0. 0. 0.]
    [3. 2. 1.]
    [0. 0. 0.]
    [3. 2. 1.]
    Test for intersection checking: [32m"Correct"[0m
    

## (All) Extra Credit: Do it with your own image!

This project is specifically designed to make you interact with your own image and accumulate more real-world CV project experience. Therefore, you're encouraged to take your own pictures, do the calibration task on your own, run detection and estimation with your code. The result you get will be written in the report. To make things easier, we summarize several important tips for you:

(calibration is the heavy work here)

- Take pictures for both before (hand is far from chair) and after interaction, and show it in the report
- Use the same camera and focal length for both calibration pictures and interaction pictures
- Use the pictures of the same resolution for all the tasks; Usually, phone camera use different zoom for taking pictures and record videos. As such, in the extra credit part, you can consider making a video which combines both camera calibration and human pose interaction. After it, extracting frames of the calibration part and use these pictures for calibration.
- `Mediapipe` and `Objectron` do not work for all the situations. So, if you find some pictures do not work in the detection part, firstly, try to change the subject and pose, secondly, try to change the environment, lights, background, etc. You should still be able to find many working situations
- The detection may not be able to work very precisely, and so you're allowed to slightly change a few real-world parameters, such as the size of chair, the depth of pose (the real-world distance between the people in the picture and the camera)


## (Grad) Extra Credit: Intersection Detection in a Video

Congratulations! You just constructed a working pipeline for a single image. At this point, we would like to process a video and look into more than one object.
- Video processing: The video proessing logic is given to you. Try to understand the logic and feel free to refer to openCV's documentation for video processing related functions.
- Pipeline building: Fill out the code blanks with calls to existing functions in order to complete the pipeline. There are detailed steps given in the notebook for you to follow.
- Video submission: Include a link to your video output in your report, make sure the link is playable (Youtube or Google drive or other playable links). You could check the output video in the data/extra_credit_video folder.

This cell should take around eight minutes two complete.

In the report, you have to answer all the questions to receive full credit for extra credit part.


```python
video_path = '../data/raw.mp4'
student_code.process_video(video_path)
```

    ../data/cali3/*.jpg
    [[1.16683850e+03 0.00000000e+00 3.13822971e+02]
     [0.00000000e+00 1.18801562e+03 5.82186193e+02]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    [[1, 3, 640, 480]] [[1, 16, 40, 30], [1, 1, 40, 30]]
    0.9694143533706665
    

    C:\Users\kbipi\miniconda3\envs\cv_proj6\lib\site-packages\numpy\core\shape_base.py:65: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      ary = asanyarray(ary)
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_1788/1056712609.py in <module>
          1 video_path = '../data/raw.mp4'
    ----> 2 student_code.process_video(video_path)
    

    c:\users\kbipi\proj6_release\proj6_release\proj6_code\student_code.py in process_video(path)
        273         sec = round(sec, 2)
        274         #print(sec)
    --> 275         success = processFrame(sec)
        276 
        277     pathIn= '../data/video/'
    

    c:\users\kbipi\proj6_release\proj6_release\proj6_code\student_code.py in processFrame(sec)
        250 
        251             # Step-3: Project 2d landmakrs to 3d using given projection_depth and project matrix from step-3
    --> 252             threeD_landmarks = projection_2d_to_3d(projection_matrix, depth, landmarks)
        253 
        254             # Step-4: Get hand coordinate from the index 22 (0-indexed) of 3d landmarks
    

    c:\users\kbipi\proj6_release\proj6_release\proj6_code\utils.py in projection_2d_to_3d(P, depth, pose2d)
        379 
        380     n=len(pose2d)
    --> 381     pose2d_h = np.hstack((pose2d, np.ones((n,1))))*depth
        382     p1=P[:3,:3]
        383     p2=P[:3,3]
    

    <__array_function__ internals> in hstack(*args, **kwargs)
    

    ~\miniconda3\envs\cv_proj6\lib\site-packages\numpy\core\shape_base.py in hstack(tup)
        341     # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
        342     if arrs and arrs[0].ndim == 1:
    --> 343         return _nx.concatenate(arrs, 0)
        344     else:
        345         return _nx.concatenate(arrs, 1)
    

    <__array_function__ internals> in concatenate(*args, **kwargs)
    

    ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)



```python
config_file = (r'D:\Hacktoberfest2021-2\Python\Object Detection\Coco\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
frozen_model = (r'D:\Hacktoberfest2021-2\Python\Object Detection\Coco\frozen_inference_graph.pb')
out = cv.VideoWriter('Result.mp4', cv.VideoWriter_fourcc(*'m', 'p', '4', 'v'), 20, (700, 500))


model = cv.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = (r'D:\Hacktoberfest2021-2\Python\Object Detection\Labels.txt')
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# print(classLabels)

capture = cv.VideoCapture(0)#r'D:\3D-Object-Detection\Photos\street2.wmv'
while True:
    isTrue, img = capture.read()
    width, height = 700, 500
    img = cv.resize(img, (width, height))

    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd <= 80):
         
                cv.rectangle(img, boxes, (255, 0, 0), 2)
                cv.putText(img, classLabels[int(ClassInd - 1)], (boxes[0] + 10, boxes[1] + 40), cv.FONT_HERSHEY_PLAIN,
                           3, color=(255, 0, 0), thickness=3)

        out.write(img)
        cv.imshow('Video', img)
        
        if cv.waitKey(1) & 0xFF == 27:
            break
capture.release()
out.release()
capture.destroAllWindows
```

## Code testing
We have provided a set of tests for you to evaluate your implementation. We have included tests inside ```proj6.ipynb``` so you can check your progress as you implement each section. At the end, you should call the tests from the terminal using the command ```pytest proj6_unit_tests/```

## Submission

This is very important as you will lose 5 points for every time you do not follow the instructions.

Do not install any additional packages inside the conda environment. The TAs will use the same environment as defined in the config files we provide you, so anything that's not in there by default will probably cause your code to break during grading. Do not use absolute paths in your code or your code will break. Use relative paths like the starter code already does. Failure to follow any of these instructions will lead to point deductions. Create the zip file using ```python zip_submission.py --gt_username <your_gt_username>``` (it will zip up the appropriate directories/files for you!) and hand it through Gradescope. Remember to submit your report as a PDF to Gradescope as well.

## Rubric
The overall rubric division for this part is:

| Submission Type | Credit Type | Individual |
| --------------- | ----------- | ---------- |
| Code            | Mandatory   | 30         |
| Report          | Mandatory   | 70         |
| Report          | EC          | 7          |
| Total           |             | 107        | 


