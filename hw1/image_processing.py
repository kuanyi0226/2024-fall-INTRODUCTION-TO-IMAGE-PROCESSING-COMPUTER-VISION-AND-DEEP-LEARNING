import cv2
import numpy as np

#1. Image Processing
#1.1
def color_separation(file_path):
    img = cv2.imread(file_path)
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)
    b_img = cv2.merge([b, zeros, zeros])
    g_img = cv2.merge([zeros, g, zeros])
    r_img = cv2.merge([zeros, zeros, r])

    cv2.imshow("Blue Channel", b_img)
    cv2.imshow("Green Channel", g_img)
    cv2.imshow("Red Channel", r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#1.2
def color_transformation(file_path):
    img = cv2.imread(file_path)
    cv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(img)
    avg_gray = ((b / 3) + (g / 3) + (r / 3)).astype(np.uint8)
    weighted_gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    cv2.imshow("OpenCV Grayscale", cv_gray)
    cv2.imshow("Average Grayscale", avg_gray)
    cv2.imshow("Weighted Grayscale", weighted_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#1.3
def color_extraction(file_path):
    img = cv2.imread(file_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([18, 0, 25])
    upper_bound = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)
    result_img = cv2.bitwise_and(img, img, mask=mask_inv)

    cv2.imshow("Yellow-Green Mask", mask)
    cv2.imshow("Image with Yellow-Green Removed", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2. Image Smoothing
#2.1
def gaussian_blur(file_path):
    img = cv2.imread(file_path)
    blurred_img = cv2.GaussianBlur(img, (11, 11), sigmaX=5, sigmaY=5)
    cv2.imshow("Gaussian Blurred Image", blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#2.2
def bilateral_filter(file_path):
    img = cv2.imread(file_path)
    bilateral_img = cv2.bilateralFilter(img, d=9, sigmaColor=90, sigmaSpace=90)
    cv2.imshow("Bilateral Filtered Image", bilateral_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#2.3
def median_filter(file_path):
    img = cv2.imread(file_path)
    median_img = cv2.medianBlur(img, ksize=5)
    cv2.imshow("Median Filtered Image", median_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#3. Edge Dectection
#3.1
def sobel_x(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    cv2.imshow("Sobel X", sobelx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#3.2
def sobel_y(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    cv2.imshow("Sobel Y", sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#3.3 Combination and Threshold
def combination_threshold(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    combined = cv2.magnitude(sobelx, sobely)
    _, thresholded = cv2.threshold(combined, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow("Combination and Threshold", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#3.4
def gradient_angle(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.arctan2(sobely, sobelx)
    cv2.imshow("Gradient Angle", gradient)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#4. Transforms
def apply_transform(file_path, angle, scale, tx, ty):
    img = cv2.imread(file_path)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M_rotate = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_scaled_img = cv2.warpAffine(img, M_rotate, (w, h))
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    transformed_img = cv2.warpAffine(rotated_scaled_img, M_translate, (w, h))
    cv2.imshow("Transformed Image", transformed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

