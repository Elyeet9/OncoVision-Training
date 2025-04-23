import cv2
import numpy as np


def adaptiveBilateralFilter(image, window_size=7, sigma_d=1.0):
    ''' Function that returns an image filtered with an adaptive bilateral filter.
    Parameters
    ___
    image: numpy array
        The input image to be filtered.
    window_size: int
        The size of the window used for filtering. It should be an odd number.
    sigma_d: float
        The constant used to calculate the domain filter. It controls the spatial extent of the filter.
    Returns
    ___
    filtered_image: numpy array
        The filtered image.
    '''
    image = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    pad = window_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    filtered_image = np.zeros_like(image)

    # Create the domain filter
    x, y = np.meshgrid(np.arange(-pad, pad + 1), np.arange(-pad, pad + 1))
    domain_filter = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))

    h, w = image.shape
    for i in range(h):
        for j in range(w):
            # Get the local window
            i1, j1 = i + pad, j + pad
            region = padded_image[i1 - pad : i1 + pad + 1, j1 - pad : j1 + pad + 1]
            center = padded_image[i1, j1]

            # ABF adaptive offset
            mean = np.mean(region)
            delta = center - mean
            if delta > 0:
                zeta = np.max(region) - center
            elif delta < 0:
                zeta = np.min(region) - center
            else:
                zeta = 0

            # Sigma for range filter using local standard deviation
            # Add a small value to avoid division by zero
            sigma_r = np.std(region) + 1e-6

            # Calculate the range filter
            diff = region - center - zeta
            range_filter = np.exp(-(diff ** 2) / (2 * sigma_r**2))

            # Combine the domain and range filters
            combined_filter = domain_filter * range_filter
            combined_filter /= np.sum(combined_filter)

            # Filter the region
            filtered_image[i, j] = np.sum(region * combined_filter)
    
    return (filtered_image * 255).astype(np.uint8)
