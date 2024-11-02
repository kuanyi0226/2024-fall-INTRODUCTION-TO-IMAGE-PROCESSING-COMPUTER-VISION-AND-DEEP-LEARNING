import cv2
import numpy as np

#1. Image Processing
#1.1
def color_separation(file_path, showImages=True):
    if(file_path != None):
        img = cv2.imread(file_path)
        #Use cv2.split() to get R G B gray scale images
        b, g, r = cv2.split(img)
        if showImages:
            zeros = np.zeros_like(b) #2D array filled with zeros
            #Use cv2.merge() to turn each gray scale image back to BGR image, individually
            b_img = cv2.merge([b, zeros, zeros])
            g_img = cv2.merge([zeros, g, zeros])
            r_img = cv2.merge([zeros, zeros, r])

            cv2.imshow("Blue Channel", b_img)
            cv2.imshow("Green Channel", g_img)
            cv2.imshow("Red Channel", r_img)
        else:
            return b,g,r

#1.2
def color_transformation(file_path):
    if(file_path != None):
        img = cv2.imread(file_path)
        #Q1:Call OpenCV function  cv2.cvtColor
        cv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Q2:Merge BGR separated channel images from problem 1.1 
        b, g, r = color_separation(file_path,False) #dont show images
        avg_gray = ((b / 3) + (g / 3) + (r / 3)).astype(np.uint8) #Convert to an 8-bit unsigned integer format (values between 0 and 255)

        cv2.imshow("cv_gray", cv_gray)
        cv2.imshow("avg_gray", avg_gray)

#1.3
def color_extraction(file_path):
    if(file_path != None):
        img = cv2.imread(file_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #Yellow-Green HSV range- Hue(H):18-85, Saturation(S):0-255, Value(V):25-255
        lower_bound = np.array([18, 0, 25])
        upper_bound = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
        #Invert the mask
        mask_inverse = cv2.bitwise_not(mask)
        #Remove Yellow and Green color in the image 
        extracted_image = cv2.bitwise_and(img, img, mask=mask_inverse)

        cv2.imshow("Yellow-Green Mask", mask)
        cv2.imshow("Extracted Image", extracted_image)

#2. Image Smoothing
#2.1
def gaussian_blur(file_path):
    if(file_path != None):
        img = cv2.imread(file_path)
        blurred_img = cv2.GaussianBlur(img, (11, 11), sigmaX=5, sigmaY=5)
        cv2.imshow("Gaussian Blurred Image", blurred_img)

#2.2
def bilateral_filter(file_path):
    if(file_path != None):
        img = cv2.imread(file_path)
        bilateral_img = cv2.bilateralFilter(img, d=9, sigmaColor=90, sigmaSpace=90)
        cv2.imshow("Bilateral Filtered Image", bilateral_img)

#2.3
def median_filter(file_path):
    if(file_path != None):
        img = cv2.imread(file_path)
        median_img = cv2.medianBlur(img, ksize=5)
        cv2.imshow("Median Filtered Image", median_img)


#3. Edge Dectection
#3.1
def sobel_x(file_path):
    if(file_path != None):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        cv2.imshow("Sobel X", sobelx)

#3.2
def sobel_y(file_path):
    if(file_path != None):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        cv2.imshow("Sobel Y", sobely)

#3.3 Combination and Threshold
def combination_threshold(file_path):
    if(file_path != None):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        combined = cv2.magnitude(sobelx, sobely)
        _, thresholded = cv2.threshold(combined, 128, 255, cv2.THRESH_BINARY)
        cv2.imshow("Combination and Threshold", thresholded)

#3.4
def gradient_angle(file_path):
    if(file_path != None):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.arctan2(sobely, sobelx)
        cv2.imshow("Gradient Angle", gradient)


#4. Transforms
def apply_transform(file_path, angle, scale, tx, ty):
    if(file_path != None):
        img = cv2.imread(file_path)
        (h, w) = img.shape[:2] #width and height from image(1920x1080)
        #center: based on the position of the burger in the given image
        center = (240,200)
        #rotate and scale
        M = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(img, M, (w, h))
        #translate(move)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(result, M, (w, h))
        cv2.imshow("Transformed Burger", result)


