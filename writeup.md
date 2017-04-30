##Udacity Nanodegree Program Self-Driving Car
###Term 1 Project 4: Advanced Lane Finding

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Distorted"
[image2]: ./output_images/calibration2_undist.jpg "Undistorted"
[image3]: ./output_images/test_undist4.jpg "Road Transformed"
[image4]: ./output_images/test_undist_binary4.jpg "Binary Example"
[image5]: ./output_images/test_warped_binary4.jpg "Warp Example"
[image6]: ./output_images/test_map_out_lane_lines_warped_binary4.jpg "Fit Visual"
[image7]: ./polynomial_equation.jpg "Polynomial Equation"
[image8]: ./radius_equation.jpg "Radius Equation"
[image9]: ./output_images/test_project_lines_img4.jpg "Output"
[video1]: ./output_images/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate_camera()` function lines 7 through 33 of the file called `term1_project4.py`.  
I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world (3D). Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners with `cv2.findChessboardCorners()` in a calibration image during iteration through the calibration images.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane (2D) with each successful chessboard detection.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

After calling this `calibrate_camera()` function in the main program (line 302), in lines 304 through 307 I applied the obtained camera calibration and distortion coefficients to a (distorted) calibration image using the `cv2.undistort()` function and obtained this result: 

Distorted:
![alt text][image1]

Undistorted:
![alt text][image2]

###Pipeline (test images)

####1. Provide an example of a distortion-corrected image.

In lines 319 through 323 of `term1_project4.py` a pipeline is applied to each test image by calling the `process_image()` function.  
This function takes, among others, the image `img` as a parameter and also the camera calibration matrix `mtx` and distortion coefficients `dist`, which were calculated via camera calibration. In line 235 of this `process_image()` function the distortion correction `cv2.undistort()` is applied with those parameters.  
Here is an example of a distortion-corrected test image:

![alt text][image3]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In line 240 of `term1_project4.py` the `process_image()` pipeline function calls the `generate_color_gradient_thresholded_binary()` function. This function (line 37) takes the undistorted image (`img`), HLS saturation channel (`s_thresh`) and x gradient (`sx_thresh`) thresholds to generate a binary image.  
First I converted the image from RGB to HLS color space and separated the L and S channels (lines 39 through 42). Then I chose a combination of binary thresholding the result of applying the Sobel operator in the x direction (lines 43 through 49) and binary thresholding the S channel (lines 50 through 52). The gradient/derivative in the x direction emphasizes edges closer to vertical like lane lines. The S channel of the HLS space on the other hand, is doing a fairly robust job of picking up the lines under very different color and contrast conditions (different colors of lane lines under varying degrees of daylight and shadow), better than the H or L channel. Finally I created a binary image to map out where either these color or gradient thresholds were met (lines 53 through 56).  
Here's an example of my output for this step from the distortion-corrected test image above:

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In line 245 of `term1_project4.py` the `process_image()` pipeline function calls the `warp()` function. This function (line 60) takes the undistorted thresholded binary image from the previous step and applies a perspective transform ("birds-eye view").  
I chose to hardcode the source (`src`) and destination (`dst`) points in the following manner:  

For the source points I picked four points of the input image in a trapezoidal shape that would represent a rectangle when looking down on the road from above (line 63).
```
src = np.float32([[580, 460], [700, 460], [1020, 665], [290, 665]])
```
For the destination points, I arbitrarily chose four corners of the rectangle to be a nice fit for displaying the warped (perspective transformed) result (lines 66 through 71).
```
img_size = (img.shape[1], img.shape[0])
offset = img_size[0]*.25
dst = np.float32([[offset, 0],
                  [img_size[0]-offset, 0],
                  [img_size[0]-offset, img_size[1]],
                  [offset, img_size[1]]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 700, 460      | 960, 0        |
| 1020, 665     | 960, 720      |
| 290, 665      | 320, 720      |

Given these src and dst points, I calculated the perspective transform matrix `M` (line 73). I also computed the inverse perspective transform matrix `Minv` (line 75) and returned it for projecting the lane lines onto the original image later on. Finally I warped (perspective transformed) the image (line 77) and returned it. I used OpenCV functions for these steps.  
Here's an example of my output for this step from the undistorted thresholded binary image above to verify that the lines appear parallel in the warped image:

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In line 250 of `term1_project4.py` the `process_image()` pipeline function calls the `map_out_lane_lines()` function. This function (line 82) takes a warped binary image, left and right Line class instances and detects left and right lane pixels to find the lane boundaries and fits a second order polynomial to each.  

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#####Radius of curvature of the lane:  
In line 257 of `term1_project4.py` the `process_image()` pipeline function calls the `measure_curvature()` function. This function (line 202) takes the x and y coordinates (arrays of pixel values `left_fitx`, `right_fitx` and `ploty`) of the left and right lane lines.  
Because pixel space is not the same as real world space I have to convert these x and y values to real world space. I'll assume that if a section of lane is projected similar to the warped images above, the lane is about 30 meters (720 pixels) long and 3.7 meters (700 pixels) wide. 
So in lines 212 and 213 I have to fit new second order polynomials (`left_fit_cr` and `right_fit_cr`) in real world space to those pixel positions (`left_fitx`, `right_fitx` and `ploty`).  
For a second order polynomial curve: ![alt text][image7] the equation for radius of curvature becomes: ![alt text][image8]  
In this radius of curvature equation I'll choose the maximum y-value of `ploty`, corresponding to the bottom of the image as y coordinate (`y_eval`) of each point on the left and right lane lines to evaluate the radii of (lines 215 and 216).

#####Position of the vehicle with respect to the center of the lane:  
In line 259 of `term1_project4.py` the `process_image()` pipeline function calls the `measure_offset()` function. This function (line 222) takes the y coordinates (array of pixel values `ploty`) and the second order polynomials (`left_fit`, `right_fit`) of the left and right lane lines and the image width.  
I'll choose the maximum y-value of `ploty`, corresponding to the bottom of the image as y coordinate of a point on the center of the lane (`y_eval`). To find the corresponding x coordinate of this lane midpoint, I first have to find the x coordinates (`left_fitx_bottom`, `right_fitx_bottom`) of `y_eval` on the left and right lane lines. So I'll fill in this `y_eval` into the equation of both the left and right second order polynomial: ![alt text][image7] (line 224 and 225). By subtracting (line 226) the obtained x coordinates from each other and dividing the result by 2, I'll get the horizontal offset of each lane line from the midpoint of the lane at `y_eval`. By adding this offset to the x coordinate (`left_fitx_bottom`) of the left lane line at `y_eval`, the x position of the midpoint of the lane at the bottom of the image (`x_lane_center_bottom`) is obtained (line 226). Finally I have subtracted this x position from the x position of the vehicle (`x_vehicle_center`, the x position of a point at the centre of the image) and multiplied this result by the meters-per-pixel-in-x-dimension factor (`xm_per_pix` ) to obtain the horizontal offset in meters of the vehicle from the center of the lane.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In line 255 of `term1_project4.py` the `process_image()` pipeline function calls the `project_lane_lines()` function. This function (line 181) takes the warped binary image (`warped_binary`), the undistorted image (`undist`), x and y coordinates (arrays of pixel values `left_fitx`, `right_fitx` and `ploty`) of the two lane lines and the inverse perspective matrix (`Minv`) as parameters to project lane boundaries onto the original (undistorted) image.
Here is an example of my result on a test image:

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

In lines 329 through 332 of `term1_project4.py` the pipeline is applied to process project_video.mp4 by calling the `process_video` function (line 331).  
This function (line 270) calls the `process_image()` function (line 272) which in turn has already been used as pipeline for processing the test images (line 323).  
Here's a [link to my video result project_video_output.mp4](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

GBE: assumed conversions in x and y from pixels space to meters