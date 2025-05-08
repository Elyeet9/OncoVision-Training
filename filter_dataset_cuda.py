import os
from tqdm import tqdm
from filters.cuda import adaptiveBilateralFilter
import cv2

# Get an array of every item inside dataset/augmented/train
train_path = 'dataset/data_v2/images/train'
filtered_train_path = 'dataset/filtered/images/train'
train_files = os.listdir(train_path)
print("Total number of training files: ", len(train_files))

# Set a progress bar
train_bar = tqdm(train_files)
for file in train_bar:
    # Set the description of the progress bar
    train_bar.set_description(f"Processing {file}")

    # Read the image
    img = cv2.imread(f'{train_path}/{file}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the filter and save the image
    filtered_img = adaptiveBilateralFilter(img, window_size=5)
    cv2.imwrite(f'{filtered_train_path}/{file}', filtered_img)

# Now the same but for validation set
val_path = 'dataset/data_v2/images/val'
filtered_val_path = 'dataset/filtered/images/val'
val_files = os.listdir(val_path)
print("Total number of validation files: ", len(val_files))
val_bar = tqdm(val_files)
for file in val_bar:
    # Set the description of the progress bar
    val_bar.set_description(f"Processing {file}")

    # Read the image
    img = cv2.imread(f'{val_path}/{file}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the filter and save the image
    filtered_img = adaptiveBilateralFilter(img, window_size=5)
    cv2.imwrite(f'{filtered_val_path}/{file}', filtered_img)
