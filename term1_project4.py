import numpy as np
import glob
import cv2
import math
from moviepy.editor import VideoFileClip

# Function that takes path to calibration/chessboard images, number of inside corners in x and y points
# and computes/returns the camera calibration matrix and distortion coefficients
def calibrate_camera(calibration_images_path='./camera_cal/calibration*.jpg', nx=9, ny=6):
    # Prepare objects points, like (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates
    # Read in and make a list of calibration images
    images = glob.glob(calibration_images_path)
    # Arrays to store object points and image points from all the images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    for idx, fname in enumerate(images):
        # Read in each image
        img = cv2.imread(fname)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw the corners and store the result to verify
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imwrite('./output_images/corners' + str(idx) + '.jpg', img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

# Function that takes the undistorted image, HLS saturation channel and x gradient thresholds
# and generates/returns a binary image
def generate_color_gradient_thresholded_binary(img, s_thresh=(120, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x direction
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255
    return combined_binary

# Function that takes an image, applies a perspective transform ("birds-eye view")
# and returns a warped image and an inverse perspective transform matrix
def warp(img):
    # Pick four points in a trapezoidal shape that would represent
    # a rectangle when looking down on the road from above
    src = np.float32([[580, 460], [700, 460], [1020, 665], [290, 665]])
    # For destination points, I'm arbitrarily choosing four corners of the rectangle
    # to be a nice fit for displaying the warped result
    img_size = (img.shape[1], img.shape[0])
    offset = img_size[0]*.25 # offset for dst points
    dst = np.float32([[offset, 0],
                      [img_size[0]-offset, 0],
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute also the inverse perspective transform and return it for projecting the lane lines
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, Minv

# Function that takes a warped binary image, left and right Line class instances
# and detects left and right lane pixels to find the lane boundaries and fits a second order polynomial to each
def map_out_lane_lines(binary_warped, left_line, right_line):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the width of the windows +/- margin
    margin = 100
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    if left_line.detected is False or right_line.detected is False:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    left_line.detected = True
    left_line.current_fit = left_fit
    right_fit = np.polyfit(righty, rightx, 2)
    right_line.detected = True
    right_line.current_fit = right_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Color in left (blue) and right (red) line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Recast the x and y points into usable format for cv2.polylines()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # Draw polynomial fits onto the image in green
    cv2.polylines(out_img, np.int_([pts_left]), False, (0,255, 0), 3)
    cv2.polylines(out_img, np.int_([pts_right]), False, (0,255, 0), 3)
    
    return ploty, left_fitx, right_fitx, left_fit, right_fit, out_img

# Function that takes the warped image, the undistorted image, the x and y coordinates
# (arrays of pixel values left_fitx, right_fitx and ploty) of the left and right lane lines,
# inverse perspective transform matrix and computes/returns an undistorted image with projected lane lines
def project_lane_lines(warped, undist, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

# Function that takes the x and y coordinates (arrays of pixel values left_fitx, right_fitx and ploty) of the left and right lane lines
# and computes/returns radii of curvature of these lines at the bottom of the image (y_eval) in real world space
def measure_curvature(ploty, left_fitx, right_fitx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return math.ceil(left_curverad*100)/100, math.ceil(right_curverad*100)/100

# Function that takes the y coordinates (array of pixel values ploty) and the second order polynomials of the left and right lane lines
# and the image width and computes/returns vehicle position (horizontal distance) with respect to center of the lane
def measure_offset(ploty, left_fit, right_fit, img_width):
    y_eval = np.max(ploty) # the maximum y-value, corresponding to the bottom of the image
    left_fitx_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2] # x position of point on left lane line at the bottom of the image
    right_fitx_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2] # x position of point on right lane line at the bottom of the image
    x_lane_center_bottom = left_fitx_bottom + (right_fitx_bottom-left_fitx_bottom)/2 # x position of point on lane center at the bottom of the image
    x_vehicle_center = img_width/2 # x position of center of the vehicle  (x position of point at the center of the image)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    offset = (x_vehicle_center-x_lane_center_bottom)*xm_per_pix # horizontal offset in meters of the vehicle center from the lane center (+ = right offset, - = left offset)
    return math.ceil(offset*100)/100

# Function that combines all the functions above to project lane lines and measurements on the original image after camera calibration
def process_image(img, idx, mtx, dist, left_line, right_line, name):
    # Remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Store the result to verify
    if name is not None:
        cv2.imwrite('./output_images/' + name + '_undist' + str(idx) + '.jpg', undist)
    # Generate binary
    undist_binary = generate_color_gradient_thresholded_binary(undist)
    # Store the result to verify
    if name is not None:
        cv2.imwrite('./output_images/' + name + '_undist_binary' + str(idx) + '.jpg', undist_binary)
    # Warp
    warped_binary, Minv = warp(undist_binary)
    # Store the result to verify
    if name is not None:
        cv2.imwrite('./output_images/' + name + '_warped_binary' + str(idx) + '.jpg', warped_binary)
    # Detect and map out lane lines on warped binary
    ploty, left_fitx, right_fitx, left_fit, right_fit, out_img = map_out_lane_lines(warped_binary, left_line, right_line)
    # Store the result to verify
    if name is not None:
        cv2.imwrite('./output_images/' + name + '_map_out_lane_lines_warped_binary' + str(idx) + '.jpg', out_img)
    # Project lane lines on undistorted image
    result = project_lane_lines(warped_binary, undist, left_fitx, right_fitx, ploty, Minv)
    # Compute radii of curvature of the lane lines
    left_curverad, right_curverad = measure_curvature(ploty, left_fitx, right_fitx)
    # Compute vehicle position with respect to center of the lane
    center_offset = measure_offset(ploty, left_fit, right_fit, img.shape[1])
    # Write measurements on the resulted image
    cv2.putText(result, 'radius: ' + str(left_curverad) + 'm', (50, img.shape[0]-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'radius: ' + str(right_curverad) + 'm', (img.shape[1]-300, img.shape[0]-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'offset: ' + str(center_offset) + 'm', (550, img.shape[0]-45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Store the result to verify
    if name is not None:
        cv2.imwrite('./output_images/' + name + '_project_lines_img' + str(idx) + '.jpg', result)
    return result

# Function that uses the process_image() function to process a video with projected lane lines and measurements
def process_video(clip, idx, mtx, dist, left_line, right_line, name):
    def new_process_image(img):
	    return process_image(img, idx, mtx, dist, left_line, right_line, name)
    return clip.fl_image(new_process_image)

# Define a class to receive the characteristics of each line detection
class Line():
	def __init__(self):
	    # was the line detected in the last iteration?
	    self.detected = False
	    # x values of the last n fits of the line
	    self.recent_xfitted = []
	    #average x values of the fitted line over the last n iterations
	    self.bestx = None
	    #polynomial coefficients averaged over the last n iterations
	    self.best_fit = None
	    #polynomial coefficients for the most recent fit
	    self.current_fit = [np.array([False])]
	    #radius of curvature of the line in some units
	    self.radius_of_curvature = None
	    #distance in meters of vehicle center from the line
	    self.line_base_pos = None
	    #difference in fit coefficients between last and new fits
	    self.diffs = np.array([0,0,0], dtype='float')
	    #x values for detected line pixels
	    self.allx = None
	    #y values for detected line pixels
	    self.ally = None

### Main program ###

# Calibrate the camera
mtx, dist = calibrate_camera()

# Remove distortion from one of the calibration images and store the result to verify
img = cv2.imread('./camera_cal/calibration2.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('./output_images/calibration2_undist.jpg', undist)

# Line class instantiations
left_line = Line()
right_line = Line()

# Process straight_lines images
images = glob.glob('./test_images/straight_lines*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    process_image(img, idx, mtx, dist, left_line, right_line, 'straight_lines')

# Process test images
images = glob.glob('./test_images/test*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    process_image(img, idx, mtx, dist, left_line, right_line, 'test')

# Line class (re-)instantiations
left_line = Line()
right_line = Line()

# Process video
clip = VideoFileClip('./project_video.mp4')
output_clip = clip.fx(process_video, None, mtx, dist, left_line, right_line, None)
output_clip.write_videofile('./project_video_output.mp4', audio=False)