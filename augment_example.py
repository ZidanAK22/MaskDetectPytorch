import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class PascalVOCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = self.load_annotations()

    def load_annotations(self):
        annotations = []
        images_dir = os.path.join(self.data_dir, 'images')
        annotations_dir = os.path.join(self.data_dir, 'annotations')

        for xml_file in os.listdir(annotations_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotations_dir, xml_file)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Extract image information
                image_path = os.path.join(images_dir, root.find('filename').text)
                image_size = (
                    int(root.find('size/width').text),
                    int(root.find('size/height').text),
                    int(root.find('size/depth').text)
                )

                # Extract object information (bounding boxes and class labels)
                objects = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bbox = [
                        int(obj.find('bndbox/xmin').text),
                        int(obj.find('bndbox/ymin').text),
                        int(obj.find('bndbox/xmax').text),
                        int(obj.find('bndbox/ymax').text),
                    ]
                    objects.append({
                        'name': name,
                        'bbox': bbox,
                    })

                annotations.append({
                    'image_path': image_path,
                    'image_size': image_size,
                    'objects': objects,
                })

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image = Image.open(annotation['image_path']).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, annotation

# Define data directory and instantiate the dataset
data_dir = 'D:/Ngodink/Python/Pytorchic/MasksDetect/ds_masks'  # Update with your dataset path
transform = transforms.Compose([
    transforms.ToTensor(),  # Move this transform here
    transforms.RandomHorizontalFlip(),
])
dataset = PascalVOCDataset(data_dir=data_dir, transform=transform)

# Display a sample of the original and augmented images
sample_idx = 0
original_image, _ = dataset[sample_idx]
augmented_image, _ = dataset[sample_idx]

# Convert tensor to NumPy array for visualization
original_image_np = original_image.permute(1, 2, 0).numpy()
augmented_image_np = augmented_image.permute(1, 2, 0).numpy()

plt.figure(figsize=(4, 4))
plt.title('Original Image')
plt.imshow(original_image_np)
plt.axis('off')
plt.show()

plt.figure(figsize=(4, 4))
plt.title('Augmented Image')
plt.imshow(augmented_image_np)
plt.axis('off')
plt.show()
