import os

images_dir = 'dataset/data_v2/images'
labels_dir = 'dataset/data_v2/labels'

count_removed = 0
for folder in os.listdir(images_dir):
    for file in os.listdir(os.path.join(images_dir, folder)):
        # check if labels_dir/folder/file.txt exists for images_dir/folder/file.png
        if not os.path.exists(os.path.join(labels_dir, folder, file.replace('.png', '.txt'))):
            # remove images_dir/folder/file.png
            os.remove(os.path.join(images_dir, folder, file))
            count_removed += 1

print(f"Removed {count_removed} unlabeled images.")
