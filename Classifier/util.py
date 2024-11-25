import os 

def count_images_recursive(directory):
    image_extensions = ".png"
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(image_extensions):
                count += 1
    return count

dataset_path = "Classifier\data"
class_counts = {}

for cls in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, cls)
    if os.path.isdir(class_path):
        class_counts[cls] = count_images_recursive(class_path)

for cls, count in class_counts.items():
    print(f"{cls}: {count} images")