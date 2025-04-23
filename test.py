from filters.adaptiveBilateralFilter import adaptiveBilateralFilter
import cv2
import time

# Read the image
img = cv2.imread('test-input.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply the adaptive bilateral filter
print('Applying adaptive bilateral filter...')
start_time = time.time()
filtered_img = adaptiveBilateralFilter(img)
end_time = time.time() - start_time
print(f'Filter applied in {end_time:.2f} seconds.')

cv2.imwrite('test-output.png', filtered_img)
