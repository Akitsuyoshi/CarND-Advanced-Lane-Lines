import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_two_image(image_1, image_2, title_1, title_2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image_1)
    ax1.set_title(title_1, fontsize=50)
    ax2.imshow(image_2)
    ax2.set_title(title_2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def plot_images(images):
    plt.figure(figsize=(25, 10))
    columns = 3
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)


def save_rgb_image(img, img_name):
    img = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(250, 145))
    img_path = 'output_images/' + img_name + '.jpg'
    cv2.imwrite(img_path, img)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Get obj and image points from given image
def get_obj_image_points(fx, fy, image_path_list=[]):
    # Prepare object points, like (0, 0, 0), (1, 0, 0) .....(6, 5, 0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # list to store object and image points respectively.
    # object points for 3d real world space, whreas image points for 2d
    objpoints = []
    imgpoints = []

    # Iterate throuch chessboard to get its corners
    for f_name in image_path_list:
        img = cv2.imread(f_name)
        # COLOR_BGR2GRAY for cv2.imgread
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corners
        ret, corners = cv2.findChessboardCorners(img, (fx, fy), None)

        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints


def calibrate_camera(obj_p, img_p, img_size):
    # ret, mtx, dist, _, _
    return cv2.calibrateCamera(obj_p, img_p, img_size, None, None)


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = gaussian_blur(img, kernel_size=9)

    # Gradient in x and y
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # Calculate magnitude
    abs_sobel = np.sqrt(sobel_x**2, sobel_y**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0])
           & (scaled_sobel <= thresh[1])] = 1

    return binary


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = gaussian_blur(img, kernel_size=9)

    # Gradient in x and y
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_grad_dir = np.arctan2(np.absolute(sobel_x), np.absolute(sobel_y))

    binary = np.zeros_like(abs_grad_dir)
    binary[(abs_grad_dir >= thresh[0])
           & (abs_grad_dir <= thresh[1])] = 1
    return binary


def col_thresh(img, thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img = gaussian_blur(img, kernel_size=11)
    # Get S channel
    s = img[:, :, 2]

    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1

    return binary


def sobel_x_thresh(img, thresh=(0, 255)):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = gaussian_blur(img, kernel_size=9)

    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def get_src_points(img_size):
    return np.float32(
            [[(img_size[0] / 2) - 20, img_size[1] / 2 + 90],
             [((img_size[0] / 6) + 20), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 60), img_size[1] / 2 + 90]])


def get_dst_points(img_size):
    return np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])


def warpe_image(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


def find_lane_pixels(binary_warped):
    # Take the histogram of bottom half of binary
    hist = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # output_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Get start point xpoint for left and right line
    mid_p = np.int(hist.shape[0] // 2)
    left_x_current = np.argmax(hist[:mid_p])
    right_x_current = np.argmax(hist[mid_p:]) + mid_p

    # HYPER PARAMS
    n_windows = 8
    margin = 100  # width of windows, +/- margin
    minpi_x = 50

    windowns_h = np.int(binary_warped.shape[0] // n_windows)

    # Identify all no zero pixels on x and y axis
    nonzero = binary_warped.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    left_lane_inds = []
    right_lane_inds = []

    # Iterate through windos
    for window in range(n_windows):
        # Get window boundaries
        win_y_low = binary_warped.shape[0] - (window+1)*windowns_h
        win_y_high = binary_warped.shape[0] - window*windowns_h
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        # For Visualization
        # Draw windows on image
        # cv2.rectangle(output_img,
        #               (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high),
        #               (0, 255, 0), 3)

        # cv2.rectangle(output_img,
        #               (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high),
        #               (0, 255, 0), 3)

        # Get nonzero pixels within windows
        good_left_inds = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) &
                          (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) &
                           (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) &
                           (nonzero_x < win_xright_high)).nonzero()[0]

        # Store each pixels
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter next window, if certen pixels are found
        if len(good_left_inds) > minpi_x:
            left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpi_x:
            right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # After iteration above, concatenate to get 1D list
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    # Extract left and right lane line pixel position
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    # output_img is for visualization
    return left_x, left_y, right_x, right_y


def fit_polynomial(binary_warped, left_x, left_y, right_x, right_y):
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generates x and y values for plottig
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    try:
        left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print("Failed to fit a lane")
        left_fit_x = ploty ** 2 + ploty
        right_fit_x = ploty ** 2 + ploty

    # For Visualization
    # Plots left and right line
    # Colors in the left and right regions
    # output_img[left_y, left_x] = [255, 0, 0]
    # output_img[right_y, right_x] = [0, 0, 255]
    # plt.plot(left_fit_x, ploty, color='yellow')
    # plt.plot(right_fit_x, ploty, color='yellow')

    return ploty, left_fit_x, right_fit_x, left_fit, right_fit


def reverse_colored_warp_image(binary_warped, left_fit_x, right_fit_x, ploty,
                               src, dst):
    # Get warped image with line
    zero_binary_warped = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((zero_binary_warped,
                            zero_binary_warped,
                            zero_binary_warped))

    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw line onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    img_size = (binary_warped.shape[1], binary_warped.shape[0])
    # Warp the blank back to original image space using inverse perspective
    # Here, getPerspectiveTransform params order, dst to src
    return cv2.warpPerspective(color_warp,
                               cv2.getPerspectiveTransform(dst, src),
                               img_size)


def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 150  # Margin(+/-) around previous polynomial

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_polynomial = left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2]
    left_lane_inds = ((nonzero_x > (left_polynomial - margin)) &
                      (nonzero_x < (left_polynomial + margin)))

    right_polynomial = right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2]
    right_lane_inds = ((nonzero_x > (right_polynomial - margin)) &
                       (nonzero_x < (right_polynomial + margin)))

    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    return left_x, left_y, right_x, right_y


def find_lane_pixels_by_line(binary_warped, line):
    """Requied line param, line is Line class instance"""
    if (line.is_detected is False):
        # Extract each pixels on lane
        left_x, left_y, right_x, right_y = find_lane_pixels(binary_warped)
    else:
        left_x, left_y, right_x, right_y = search_around_poly(binary_warped,
                                                              line.recent_left_fit,
                                                              line.recent_right_fit)
    return left_x, left_y, right_x, right_y


def xy_merter_per_pix():
    """ Return: meter on x and y axis per pixel """
    return 3.7/700, 30/720


def measure_pos_from_center(img_size, left_x, right_x):
    xm_per_pix, _ = xy_merter_per_pix()
    # Center of found lane in image
    lane_center = left_x[-1] + ((right_x[-1] - left_x[-1]) / 2)
    # Diff between center of lane - position of vehicle
    offset = img_size[0] / 2 - lane_center

    return offset * xm_per_pix


def measure_curvature(ploty, left_fit, right_fit):
    xm_per_pix, ym_per_pix = xy_merter_per_pix()
    y_eval = np.max(ploty)

    left_curved = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])

    right_curved = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return np.mean([left_curved, right_curved])


def draw_two_text(img, position, curvature):
    text_1 = 'Radious of Curvature = ' + str(round(curvature, 1)) + '(m)'
    if (position < 0):
        text_2 = 'Vehicle is ' + str(round(-position, 2)) + '(m)' + ' left of center'
    else:
        text_2 = 'Vehicle is ' + str(round(position, 2)) + '(m)' + ' right of center'

    cv2.putText(img, text_1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text_2, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3, cv2.LINE_AA)
