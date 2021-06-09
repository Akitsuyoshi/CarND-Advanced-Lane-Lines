## Advanced Lane Finding

This is my second project of [Self-Driving Car Engineer nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) in udacity.

You can see the first project at [this link](https://github.com/Akitsuyoshi/CarND-LaneLines-P1).

---

## Table of Contents

- [Advanced Lane Finding](#advanced-lane-finding)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Pipeline Steps](#pipeline-steps)
  - [0 Setting](#0-setting)
  - [1 Camera calibration](#1-camera-calibration)
  - [2 Distortion correction](#2-distortion-correction)
  - [3 Color/gradient threshold](#3-colorgradient-threshold)
  - [4 Perspective transform](#4-perspective-transform)
  - [5 Detect lane lines](#5-detect-lane-lines)
  - [6 Determine lane curvature](#6-determine-lane-curvature)
  - [7 Warplane boundaries](#7-warplane-boundaries)
  - [Pipeline conclusion](#pipeline-conclusion)
- [Pipeline Output](#pipeline-output)
  - [Image result](#image-result)
  - [Video result](#video-result)
- [Discussion](#discussion)
  - [Problem during my implementation](#problem-during-my-implementation)
  - [Improvements to pipeline](#improvements-to-pipeline)
  - [Future Feature](#future-feature)
- [Author](#author)

---

## Overview

Here is my goal for this project:

- Make a pipeline to identify the lane boundaries in a video from a front-facing camera on a car
- Reflect on my work

[//]: # (Image References)

[image1]: ./output_images/original_chess.jpg "Original"
[image2]: ./output_images/undist_chess.jpg "Undistorted"
[image3]: ./output_images/test1.jpg "Test1"
[image4]: ./output_images/test2.jpg "Test2"
[image5]: ./output_images/test3.jpg "Test3"
[image6]: ./output_images/undist_test1.jpg "Test1"
[image7]: ./output_images/undist_test2.jpg "Test2"
[image8]: ./output_images/undist_test3.jpg "Test3"
[image9]: ./output_images/binary_images0.jpg "Binary1"
[image10]: ./output_images/binary_images1.jpg "Binary2"
[image11]: ./output_images/binary_images2.jpg "Binary3"
[image11]: ./output_images/binary_images2.jpg "Binary3"
[image12]: ./output_images/warped_test1.jpg "Warped1"
[image13]: ./output_images/warped_test2.jpg "Warped2"
[image14]: ./output_images/warped_test3.jpg "Warped3"
[image15]: ./output_images/red_lined0.jpg "ledline1"
[image16]: ./output_images/red_lined1.jpg "ledline2"
[image17]: ./output_images/red_lined2.jpg "ledline3"
[image18]: ./output_images/red_lined3.jpg "ledline4"
[image19]: ./output_images/red_lined4.jpg "ledline5"
[image20]: ./output_images/red_lined5.jpg "ledline6"
[image21]: ./output_images/binary_warped_images0.jpg "Binary warped1"
[image22]: ./output_images/binary_warped_images1.jpg "Binary warped2"
[image23]: ./output_images/binary_warped_images2.jpg "Binary warped3"
[image24]: ./examples/color_fit_lines.jpg "Color Fit"
[image25]: ./sliding_widnow.png "Sliding window"

[output1]: ./output_images/test_output0.jpg
[output2]: ./output_images/test_output1.jpg
[output3]: ./output_images/test_output2.jpg
[output4]: ./output_images/test_output3.jpg
[output5]: ./output_images/test_output4.jpg
[output6]: ./output_images/test_output5.jpg
[output7]: ./output_images/test_output6.jpg
[output8]: ./output_images/test_output7.jpg
[output9]: ./project_output.gif

The input image(or video frame) and expected output from a pipeline look like this:

|Input         |Expected output       |
|:---------------------:|:---------------------:|
|![alt text][image3]  |![alt text][output1]    |

---

## Pipeline Steps

The steps for making that pipeline are the following:

- Apply Gaussian blur to reduce noises of image 
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit them to find the lane boundary.
- Determine the curvature of the lane and vehicle position to the center of the road.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### 0 Setting

To set it up to run this script at first, I followed this [starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with docker. If you installed docker successfully, you can run jupyter notebook on your local machine by the command below.

```sh
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

### 1 Camera calibration

Here is the code that I made in this part, each function is located in [helper.py](./helper.py).

*All of my functions are located in [helper.py](./helper.py). And my pipeline are in [P2.ipynb](./P2.ipynb)*

|Output         |Functions      |
|:--------------|:--------------|
|Ojbect and image ponits|`get_obj_image_points`|
|Calibrate camera|`calibrate_camera`(`cv2.calibrateCamera`)|
|Undistort image|`undistort_image`(`cv2.undistort`)|

I started to see the chessboard images, in [camera_cal/](./camera_cal) folder. And then I write the `get_obj_image_points` function to get "object points" and "image points". Object points represent three coordinates, like (x, y, z) of the chessboard corners in each image. I assumed the chessboard is fixed on the wall, which means `z = 0`.

After I got object points and image points from `get_obj_image_points`, I compute camera calibration and distortion coefficients using `calibrate_camera`.

Then I got matrix and distortion coefficients value from `calibrate_camere`, I undistort original chessboard images by `undistort_image`.

|Original Image         |Unistorted Image       |
|:---------------------:|:---------------------:|
|![alt text][image1]  |![alt text][image2]    |

### 2 Distortion correction

|Output         |Functions      |
|:--------------|:--------------|
|Undistort Image|`undistort_image`(`cv2.undistort`)|

`undistort_image`'s parameters, object points, and image points are got from the camera calibration step.

I applied `undistort_image` to some of my test images. The output, undistorted images look like this.

|Original Image         |Unistorted Image       |
|:---------------------:|:---------------------:|
|![alt text][image3]  |![alt text][image6]    |
|![alt text][image4]  |![alt text][image7]    |
|![alt text][image5]  |![alt text][image8]    |

### 3 Color/gradient threshold

|Output         |Functions      |
|:--------------|:--------------|
|Magnitude threshold|`mag_thresh`|
|Direction of Gradient|`dir_thresh`|
|Color threshold|`col_thresh`|
|Sobel threshold on x axis|`sobel_x_thresh`|

I used a combination of color and gradient thresholds to extract binary images from the undistorted image.

Examples of binary images resulted from the combination threshold are the following.

|Original Image         |Binary Image       |
|:---------------------:|:---------------------:|
|![alt text][image3]  |![alt text][image9]    |
|![alt text][image4]  |![alt text][image10]    |
|![alt text][image5]  |![alt text][image11]    |

### 4 Perspective transform

|Output         |Functions      |
|:--------------|:--------------|
|Src points|`get_src_points`|
|dst points|`get_dst_points`|
|Warp Image|`warpe_image`(`cv2.warpPerspective`)|

I chose a hard code way to get src and dst points for perspective. Warped images may seem unprecise, but they can mostly work for the test video.
Src and dst points are defined in `get_src_points` and `get_dst_points` respectively.

With src and dst points, `warpe_image` returns warped images like this one. `warpe_image` function uses `cv2.warpPerspective` internally to get the images.

|Original Image         |Warped Image       |
|:---------------------:|:---------------------:|
|![alt text][image3]  |![alt text][image12]    |
|![alt text][image4]  |![alt text][image13]    |
|![alt text][image5]  |![alt text][image14]    |

To make sure that my transform above works correctly, I compare each src points in the original image with dst points in the warped image.
The results of that are the following.

|Src points drawn in original         |Dst points drawn in warped       |
|:---------------------:|:---------------------:|
|![alt text][image15]  |![alt text][image18]    |
|![alt text][image16]  |![alt text][image19]    |
|![alt text][image17]  |![alt text][image20]    |

I checked by applying a perspective transform to binary images as well because later I use `warpe_image` to a binary image in my pipeline.

|Binary Image         |Binary Warped Image      |
|:---------------------:|:---------------------:|
|![alt text][image9]  |![alt text][image21]    |
|![alt text][image10]  |![alt text][image22]    |
|![alt text][image11]  |![alt text][image23]    |

### 5 Detect lane lines

|Output         |Functions      |
|:--------------|:--------------|
|Lane pixels for left and right each|`find_lane_pixels`|
|Poloynomial for left and right line|`fit_polynomial`|
|Lane pixels from previous line polynomial value|`search_around_poly`|

I locate lane lines from the threshold warped image. To do that, I implement `find_lane_pixels`. This function returns the line's x points corresponding to y points for both left and right lane lines. It starts to separate the image by some certain parts and get each point on the image step by step to get pixels values for both lines.
Those values are used in `fit_polynomial` to get the line's polynomial value.

![alt text][image25]

Lines in threshold image can be represented as `y = ax**2 + bx + c`, and `fit_polynomial` returns list, like `[a, b, c]`.

The following image is a good example of that.

![alt text][image24]

Once I got the polynomial value for both lines, I can use this to get lines in the next video frame, using `search_around_poly`.
If the current pipeline failed to detect lines, it starts to search by `find_lane_pixels`. The `search_around_poly` functions are kinda optimization steps because it reduces to compute the finding algorithm more than `find_lane_pixels`.

### 6 Determine lane curvature

|Output         |Functions      |
|:--------------|:--------------|
|Vehicle position from lane center|`measure_pos_from_center`|
|Curvature of radious|`measure_curvature`|
|Changed pixel value in the image to meter|`xy_merter_per_pix`|

Both, `measure_pos_from_center` and `measure_curvature` returns meter, not pixel on the image.

### 7 Warplane boundaries

The output examples are shown in the image result part below.

|Output         |Functions      |
|:--------------|:--------------|
|Get reverse warped image|`reverse_colored_warp_image`(`cv2.warpPerspective`)|
|Combined image with undistort and reverse warped|`cv2.addWeighted`|

I implemented `reverse_colored_warp_image` to make a warped image back to the original image. Before doing so, I add color to the warped image to classify which area is an interesting one, in this project, lane in the image. And I use `cv2.addWeighted` to put the undistorted image and reverse-colored warped image together.

### Pipeline conclusion

I summarize the steps here. The code for pipeline can be found in [P2.ipynb](./P2.ipynb)

- Calibrate Camera to get object and image points
- Undistort RGB image, convert image to grayscale, apply gaussian blur to reduce noises of image
- Get combined threshold binary image(each threshold are following)
  - magnitude threshold binary
  - direction of gradient binary
  - sobel binary on the x-axis
  - color binary
- Get src and dst points for perspective transform
- Make a warped image
- Extract pixel value on lanes in the warped image
- Calculate position and curvature
- Reverse warp image with colored lane
- Put colored lane image and undistort RGB image together
- Draw text on output image to show position and curvature

Image pipeline and video pipeline works almost in the same way, but video considers previous video frame and previous found lane in its implementation.

---

## Pipeline Output

### Image result

The output from `image_pipeline` looks like this.

||||
|:---------------------:|:---------------------:|:---------------------:|
|![alt text][output1] |![alt text][output2]|![alt text][output3]|
|![alt text][output4] |![alt text][output5]|![alt text][output6]|
|![alt text][output7] |![alt text][output8]|

### Video result

Here's a [link to my video result](./project_output.mp4).

![alt text][output9]

---

## Discussion

### Problem during my implementation

When I first saved the output image from each function, like undistortion, I resized its shape. However, I realized that the input image, processing image, and output image need to be the same shape. I got some errors from my unneccessary resizing. In reality, I also need to check the original test image shape before any process maybe.

It's a small pisky bug that I faced, but cv2.imread is a BGR image, and sometimes other functions process RGB image. I got some color problems due to that difference.

When implementing the video pipeline, I noticed that some of my src and dst points definitions didn't work as expected. Especially, curve or shadow time, it likely to fail. I change src accordingly, but it is still written in a hard code way. I couldn't figure it out to get those points correctly for now.

In the part of getting curvature, I still get confused about the way to calculate it. I mostly write my code from udacity lesson code, but not sure of its mathematical formula.

Since it's kinda new for me to write code in python, I always put `print()` statement for debugging code. Might be another better way than that.

### Improvements to pipeline

My current pipeline likely to fail to track when a snow or strong light appears in an image. To deal with that, One possible solution would change the color threshold.

Current src and dst points are written in a hardcode way. Especially src points, it's better to adjust to the image, not only by its shape but also something like lane condition.

### Future Feature

Will try to test my pipeline in two challenge videos. I only test for [project_video.mp4](./project_video.mp4) for now.

---

## Author

- [Tsuyoshi Akiyama](https://github.com/Akitsuyoshi)
