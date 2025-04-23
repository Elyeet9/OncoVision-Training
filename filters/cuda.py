import cupy as cp

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
    pad = window_size // 2
    img = cp.asarray(image, dtype=cp.float32) / 255.0
    img = cp.pad(img, pad, mode='reflect')
    H, W = image.shape
    k = window_size

    # Create strided sliding window view
    shape = (H, W, k, k)
    strides = img.strides * 2
    patches = cp.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    # Domain filter
    y, x = cp.meshgrid(cp.arange(-pad, pad + 1), cp.arange(-pad, pad + 1))
    domain_filter = cp.exp(-(x**2 + y**2) / (2 * sigma_d**2))
    domain_filter = domain_filter[None, None, :, :]

    # Central pixels
    center = patches[:, :, pad, pad][:, :, None, None]

    # Local stats
    local_mean = cp.mean(patches, axis=(2, 3), keepdims=True)
    local_min = cp.min(patches, axis=(2, 3), keepdims=True)
    local_max = cp.max(patches, axis=(2, 3), keepdims=True)
    delta = center - local_mean

    # ζ adaptive offset
    zeta = cp.where(delta > 0, local_max - center,
           cp.where(delta < 0, local_min - center, 0.0))

    # σr: adaptive std dev
    sigma_r = cp.std(patches, axis=(2, 3), keepdims=True) + 1e-5

    # Range filter
    diff = patches - center - zeta
    range_filter = cp.exp(-(diff**2) / (2 * sigma_r**2))

    # Combined kernel
    kernel = domain_filter * range_filter
    kernel /= cp.sum(kernel, axis=(2, 3), keepdims=True)

    # Apply to image
    result = cp.sum(kernel * patches, axis=(2, 3))
    return cp.asnumpy((result * 255).clip(0, 255).astype(cp.uint8))
