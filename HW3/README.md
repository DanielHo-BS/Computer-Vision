# HW3

## [Part 1: Homography Estimation](./src/part1.py)

* Familiar **DLT** estimation method
* Warping
  * forward
  * backward

### **Homography**

1. Matrix form

<img src=./resource/Matrix_Homography.png width="400">

2. Multiply by denominator

<img src=./resource/Matrix_Homography_2.png width="400" >

3. Linear system

<img src=./resource/Matrix_Homography_3.png width="400" >

    * $𝐴ℎ = 0$
    * $SVD of A = U \sum V^T$
    * Let ℎ be the last column of 𝑉.

### Forward Warping

<img src=./resource/times.jpg width="400" > <img src=./src/output1_1.png width="400" >

## [Part 2: Marker-Based Planar AR](./src/part2.py)

* Process the given video and ``backward warp`` the
template image per frame.
  * Since we are using backward warping, there should be ``no holes``.
* The output video should contain the warped template
image as if it were there.

### Backward Warping

<img src=./resource/times.jpg width="400" > <img src=./src/output1_2.png width="400" >

## [Part 3: Unwarp the Secret](./src/part3.py)

* Unwarp the QR code with backward warping.

<img src=./resource/BL_secret1.png width="400" > <img src=./resource/BL_secret2.png width="400" >

## [Part 4: Panorama](./src/part4.py)

* Implement the function panorama( ).
  * Estimate the homography between 3 images.
  * Using feature matching, RANSAC to find correct transform.
* ``Stitch`` 3 images using ``backward warping``.
* Feature detection & matching
  * Use opencv built-in ORB detector for keypoint detection.
    * ORB_create( ), detectAndCompute( )
  * Use opencv brute force matcher for feature matching.
    * cv.BFMatcher( ), match( )

<img src=./src/output4.png width="800" >

## Runs

In the terminal run the ``bash file`` to get results of 4 parts:

```bash
./src/hw3.sh
```
