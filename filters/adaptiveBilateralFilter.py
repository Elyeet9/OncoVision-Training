import numpy as np
import cv2

def adaptiveBilateralFilter(img, window_size=7, sigma_space=50, sigma_color_factor=0.25):
    '''Adaptive Bilateral Filter to reduce image noise while preserving edges.

    img : numpy.array
        Input image (grayscale or color).
    window_size : int
        Size of the filter window (must be odd).
    sigma_space : float
        Standard deviation for the spatial Gaussian kernel.
    sigma_color_factor : float
        Factor to adaptively calculate the color standard deviation based on local pixel intensity variation.
    '''

    # Window size must be odd
    if window_size % 2 == 0:
        window_size += 1

    # Convert to grayscale if the image is in color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Create a copy for the result
    result = np.zeros_like(img)

    # Padding to handle the borders
    pad = window_size // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # For each pixel in the image
    for i in range(pad, padded_img.shape[0] - pad):
        for j in range(pad, padded_img.shape[1] - pad):
            # Get the local window
            window = padded_img[i-pad:i+pad+1, j-pad:j+pad+1]

            # Get the standard deviation of the local window
            local_std = np.std(window)
            sigma_color = max(10, local_std * sigma_color_factor)

            # Apply bilateral filter to the local window
            filtered_pixel = cv2.bilateralFilter(
                padded_img[i-pad:i+pad+1, j-pad:j+pad+1],
                d=window_size,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )[pad, pad]

            # Save the result
            result[i-pad, j-pad] = filtered_pixel

    return result
