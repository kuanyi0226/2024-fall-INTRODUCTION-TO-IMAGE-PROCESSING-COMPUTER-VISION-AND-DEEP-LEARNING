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
def apply_gaussian_blur(m):
    # Calculate the kernel size of of gaussian filter:(2m + 1) x (2m + 1)
    kernel_size = 2 * m + 1
    # Gaussian Blur
    blur = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
    # Show image
    cv2.imshow("Gaussian Blur", blur)

def gaussian_blur(file_path):
    global original_image
    if(file_path != None):
        original_image = cv2.imread(file_path)

        cv2.namedWindow("Gaussian Blur")
        cv2.createTrackbar("m", "Gaussian Blur", 1, 5, apply_gaussian_blur)
        # init with m = 1
        apply_gaussian_blur(1)

#2.2
def apply_bilateral_filter(m):
    # Calculate d
    d = 2 * m + 1
    # Bilateral Filter
    bilateral = cv2.bilateralFilter(original_image, d=d, sigmaColor=90, sigmaSpace=90)
    # Show image 
    cv2.imshow("Bilateral Filter", bilateral)

def bilateral_filter(file_path):
    global original_image
    if(file_path != None):
        original_image = cv2.imread(file_path)

        cv2.namedWindow("Bilateral Filter")
        cv2.createTrackbar("m", "Bilateral Filter", 1, 5, apply_bilateral_filter)
        # init with m = 1
        apply_bilateral_filter(1)

#2.3
def apply_median_filter(m):
    # Calculate kernal size
    kernel_size = 2 * m + 1
    # Bilateral Filter
    median = cv2.medianBlur(original_image, ksize=kernel_size)
    # Show image 
    cv2.imshow("Median Filter", median)

def median_filter(file_path):
    global original_image
    if(file_path != None):
        original_image = cv2.imread(file_path)

        cv2.namedWindow("Median Filter")
        cv2.createTrackbar("m", "Median Filter", 1, 5, apply_median_filter)
        # init with m = 1
        apply_median_filter(1)

#3. Edge Dectection
#3.1
def sobel_x(file_path, showImages=True):
    if file_path is not None:
        image = cv2.imread(file_path)
        #Convert the RGB image into a grayscale image 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Smooth grayscale image with Gaussian smoothing filter
        m = 1
        kernel_size = 2 * m + 1  # Kernel size of Gaussian filter: (2m+1) x (2m+1)
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX =0, sigmaY=0)

        # Sobel X filter
        sobel_x_filter = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])

        # init result
        sobel_x_result = np.zeros_like(blur, dtype=np.float32) #float32: prevent overflow
        # convolution 
        for i in range(1, blur.shape[0] - 1):
            for j in range(1, blur.shape[1] - 1):
                # Extract the 3x3 region from blur
                region = blur[i - 1:i + 2, j - 1:j + 2]
                # Apply the Sobel kernel
                sobel_value = np.sum(region * sobel_x_filter)
                sobel_x_result[i, j] = sobel_value
        
        
        # Take the absolute value and scale to 0-255 for display
        sobel_x_result = np.abs(sobel_x_result)  # Take absolute values
        sobel_x_result = np.clip(sobel_x_result,0,255) 
        sobel_x_result = sobel_x_result.astype(np.uint8)  # Convert to uint8, to make it compatible with cv2.imshow
        
        if showImages:
            # show image
            cv2.imshow("Sobel X", sobel_x_result)
        else:
            # use the result in 3.3 & 3.4
            return sobel_x_result

#3.2
def sobel_y(file_path, showImages=True):
    if file_path is not None:
        image = cv2.imread(file_path)
        #Convert the RGB image into a grayscale image 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Smooth grayscale image with Gaussian smoothing filter
        m = 1
        kernel_size = 2 * m + 1  # Kernel size of Gaussian filter: (2m+1) x (2m+1)
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigmaX =0, sigmaY=0)

        # Sobel Y filter
        sobel_y_filter = np.array([[-1,-2,-1],
                                   [ 0, 0, 0],
                                   [ 1, 2, 1]])

        # init result
        sobel_y_result = np.zeros_like(blur, dtype=np.float32) #float32: prevent overflow
        # convolution 
        for i in range(1, blur.shape[0] - 1):
            for j in range(1, blur.shape[1] - 1):
                # Extract the 3x3 region from blur
                region = blur[i - 1:i + 2, j - 1:j + 2]
                # Apply the Sobel kernel
                sobel_value = np.sum(region * sobel_y_filter)
                sobel_y_result[i, j] = sobel_value
        
        
        # Take the absolute value and scale to 0-255 for display
        sobel_y_result = np.abs(sobel_y_result)  # Take absolute values
        sobel_y_result = np.clip(sobel_y_result,0,255) 
        sobel_y_result = sobel_y_result.astype(np.uint8)  # Convert to uint8, to make it compatible with cv2.imshow
        
        if showImages:
            # show image
            cv2.imshow("Sobel Y", sobel_y_result)
        else:
            # use the result in 3.3 & 3.4
            return sobel_y_result
        
#3.3 Combination and Threshold
def combination_threshold(file_path):
    if file_path is not None:
        # Use the rsult from Sobel X and Sobel Y
        sobel_x_result = sobel_x(file_path, showImages=False)
        sobel_y_result = sobel_y(file_path, showImages=False)

        # Combine Sobel X and Sobel Y (need to use astype(np.float32), or there will be an error)
        combination = np.sqrt(sobel_x_result.astype(np.float32)**2 + sobel_y_result.astype(np.float32)**2)

        # Normalize the combination result to the range [0, 255]
        normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        # Apply thresholds: (1) 128, (2) 28
        _, result1 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        _, result2 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

        # Show both combination and threshold results
        cv2.imshow('Combination Result', normalized)
        cv2.imshow('Threshold Result (128)', result1)
        cv2.imshow('Threshold Result (28)', result2)

#3.4
def gradient_angle(file_path):
    if file_path is not None:
        # Compute Sobel X and Sobel Y
        sobel_x_result = sobel_x(file_path, showImages=False)
        sobel_y_result = sobel_y(file_path, showImages=False)

        # Calculate the gradient angle θ in degrees
        gradient_angle = np.arctan2(sobel_y_result, sobel_x_result) * (180.0 / np.pi)
        gradient_angle[gradient_angle < 0] += 360  # Normalize angles to [0, 360)

        # Create masks for the specified angle ranges
        mask1 = np.zeros_like(gradient_angle, dtype=np.uint8)
        mask2 = np.zeros_like(gradient_angle, dtype=np.uint8)

        # Mask for angle range 170˚ to 190˚
        mask1[(gradient_angle >= 170) & (gradient_angle <= 190)] = 255
        # Mask for angle range 260˚ to 280˚
        mask2[(gradient_angle >= 260) & (gradient_angle <= 280)] = 255

        # Apply bitwise AND to the Sobel results with the masks
        result1 = cv2.bitwise_and(sobel_x_result.astype(np.uint8), mask1)
        result2 = cv2.bitwise_and(sobel_y_result.astype(np.uint8), mask2)

        # Show both results
        cv2.imshow('Result for 170˚ to 190˚', result1)
        cv2.imshow('Result for 260˚ to 280˚', result2)

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


