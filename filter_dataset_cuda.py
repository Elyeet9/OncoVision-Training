import os
from tqdm import tqdm
from filters.cuda import adaptiveBilateralFilter
import cv2

# Get an array of every item inside dataset/augmented/train
train_files = os.listdir('dataset/augmented/train')
print("Total number of training files: ", len(train_files))

# Set a progress bar
train_bar = tqdm(train_files)
for file in train_bar:
    # Set the description of the progress bar
    train_bar.set_description(f"Processing {file}")

    # Read the image
    img = cv2.imread(f'dataset/augmented/train/{file}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the filter and save the image
    filtered_img = adaptiveBilateralFilter(img, window_size=5)
    cv2.imwrite(f'dataset/filtered/train/{file}', filtered_img)

# Now the same but for validation set
val_files = os.listdir('dataset/augmented/val')
print("Total number of validation files: ", len(val_files))
val_bar = tqdm(val_files)
for file in val_bar:
    # Set the description of the progress bar
    val_bar.set_description(f"Processing {file}")

    # Read the image
    img = cv2.imread(f'dataset/augmented/val/{file}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the filter and save the image
    filtered_img = adaptiveBilateralFilter(img, window_size=5)
    cv2.imwrite(f'dataset/filtered/val/{file}', filtered_img)
