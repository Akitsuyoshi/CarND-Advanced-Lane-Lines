import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


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


def get_image(image_path):
    return cv2.imread(image_path)


def get_rgb_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image_path, image):
    cv2.imwrite(image_path, image)


# def get_resize_image(image, resized_shape=(250, 145)):
#     return cv2.resize(image, dsize=resized_shape)


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
        img = get_image(f_name)
        # COLOR_BGR2GRAY for cv2.imgread
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the corners
        ret, corners = cv2.findChessboardCorners(gray, (fx, fy), None)

        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints


# Save resulted image from passed func, by 250 * 145 size
def save_images(folder_name, image_path_list=[]):
    for img_path in image_path_list:
        img = get_image(img_path)
        res = cv2.resize(img, dsize=(250, 145))

        img_path = 'output_images/'+folder_name+'/'+os.path.basename(img_path)
        cv2.imwrite(img_path, res)


def undistort_image(img, obj_p, img_p):
    img_shape = (img.shape[1], img.shape[0])
    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_p, img_p, img_shape,
                                               None, None)

    return cv2.undistort(img, mtx, dist, None, mtx)


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient in x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Calculate magnitude
    abs_sobel = np.sqrt(sobel_x**2, sobel_y**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0])
           & (scaled_sobel <= thresh[1])] = 1

    return binary


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient in x and y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_grad_dir = np.arctan2(np.absolute(sobel_x), np.absolute(sobel_y))

    binary = np.zeros_like(abs_grad_dir)
    binary[(abs_grad_dir >= thresh[0])
           & (abs_grad_dir <= thresh[1])] = 1
    return binary


def col_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Get s channel
    s = hls[:, :, 2]

    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1

    return binary


def sobel_x_thresh(img, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def combined_with_fixed_thresh_range(img):
    mag_binary = mag_thresh(img, sobel_kernel=15, thresh=(80/3, 80))
    dir_binary = dir_thresh(img, sobel_kernel=15, thresh=(0.6, 1.3))
    col_binary = col_thresh(img, thresh=(135, 255))
    sobel_x_binary = sobel_x_thresh(img, thresh=(30, 110))

    combined_binary = np.zeros_like(col_binary)
    combined_binary[(col_binary == 1) |
                    (sobel_x_binary == 1) |
                    (mag_binary == 1) & (dir_binary == 1)] = 1
    return combined_binary


def get_src_points(img_size):
    return np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) + 20), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])


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
    output_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Get start point xpoint for left and right line
    mid_p = np.int(hist.shape[0] // 2)
    left_x_current = np.argmax(hist[:mid_p])
    right_x_current = np.argmax(hist[mid_p:]) + mid_p

    # HYPER PARAMS
    n_windows = 12
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

        # Draw windows on image
        cv2.rectangle(output_img,
                      (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0), 3)
        
        cv2.rectangle(output_img,
                      (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 255, 0), 3)

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

    return left_x, left_y, right_x, right_y, output_img


def fit_polynomial(binary_warped):
    left_x, left_y, right_x, right_y, output_img = find_lane_pixels(binary_warped)

    # Get y = m * x ** 2 + n * x + b
    # each polyfit returns list, [m, n, b]
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

    # Colors in the left and right regions
    output_img[left_y, left_x] = [255, 0, 0]
    output_img[right_y, right_x] = [0, 0, 255]

    # Plots
    plt.plot(left_fit_x, ploty, color='yellow')
    plt.plot(right_fit_x, ploty, color='yellow')

    return output_img, left_fit, right_fit
