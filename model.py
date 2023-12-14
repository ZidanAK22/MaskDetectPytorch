import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation, color
from skimage.feature import local_binary_pattern

class PascalVOCDataset(Dataset):
    def __init__(self, data_dir, transform=None, segmentation=False, feature_extraction=False, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.segmentation = segmentation
        self.target_size = target_size
        self.feature_extraction = feature_extraction
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

        if self.target_size:
            image = transforms.Resize(self.target_size)(image)

        # Apply transformations if specified
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = image

        # Image Segmentation
        if self.segmentation:
            # Felzenszwalb's segmentation
            seg_mask = self.felzenszwalb_segmentation(image)
            seg_mask_transformed = self.felzenszwalb_segmentation(image_transformed)

            # Note the addition of class labels here
            return image_transformed, seg_mask_transformed, annotation['objects']

        # Feature Extraction
        elif self.feature_extraction:
            # Extract features using LBP
            lbp_features = self.extract_lbp_features(image_transformed)

            # Note the addition of class labels here
            return image_transformed, lbp_features, annotation['objects']

        # Default behavior (no segmentation or feature extraction)
        else:
            # Note the addition of class labels here
            return image_transformed, annotation['objects']

        
    def felzenszwalb_segmentation(self, image):
        
        # Convert the image to numpy array
        image_array = np.array(image)

        # Ensure correct channel order (transpose from CHW to HWC)
        image_array = image_array.transpose((1, 2, 0))

        # Check if the image is grayscale (single channel)
        if len(image_array.shape) == 2:
            # Add a third channel to make it compatible with rgb2lab
            image_array = np.expand_dims(image_array, axis=-1)

        # Convert RGB to LAB color space (recommended for Felzenszwalb)
        lab_image = color.rgb2lab(image_array)

        # Perform Felzenszwalb segmentation
        seg_mask = segmentation.felzenszwalb(lab_image, scale=100, sigma=0.5, min_size=50)

        return seg_mask

    def extract_lbp_features(self, image):
        # Convert the image to numpy array
        image_array = np.array(image)

        # Ensure correct channel order (transpose from CHW to HWC)
        image_array = image_array.transpose((1, 2, 0))

        # Check if the image is grayscale (single channel)
        if len(image_array.shape) == 2:
            # Add a third channel to make it compatible with rgb2gray
            image_array = np.expand_dims(image_array, axis=-1)

        # Convert RGB to grayscale
        gray_image = color.rgb2gray(image_array)

        # Compute Local Binary Pattern (LBP)
        lbp_radius = 1
        lbp_points = 8 * lbp_radius
        lbp = local_binary_pattern(gray_image, lbp_points, lbp_radius, method='uniform')

        # Compute histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp.max() + 1), density=True)

        return hist       
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
    
def split_dataset(dataset, split_ratio=0.8):
    # Calculate the size of training and validation sets
    num_samples = len(dataset)
    train_size = int(split_ratio * num_samples)
    val_size = num_samples - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset   

def collate_fn(batch):
    return tuple(zip(*batch)) 

# Use the new path for your dataset
data_dir = r'D:\Ngodink\Python\Pytorchic\Mask\ds_masks'

# Create an instance of the dataset with the updated path
dataset = PascalVOCDataset(data_dir, transform=transforms.ToTensor(), segmentation=True, target_size=(224, 224))

# Define classification model and criterion
num_classes = 3  # Adjust based on your dataset's number of classes
classification_model = SimpleCNN(num_classes)
classification_criterion = nn.CrossEntropyLoss()
classification_optimizer = optim.Adam(classification_model.parameters(), lr=0.001)

# Combine datasets for classification training
full_dataset = PascalVOCDataset(data_dir=data_dir, transform=T.ToTensor(), feature_extraction=True)
train_size = int(0.8 * len(full_dataset))
train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

batch_size = 32

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop for classification
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    for images, features, labels in train_loader:
        # Zero the gradients
        classification_optimizer.zero_grad()

        # Forward pass
        outputs = classification_model(images)

        # Compute the loss
        loss = classification_criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        classification_optimizer.step()

    # Validation after each epoch
    classification_model.eval()
    with torch.no_grad():
        for val_images, val_features, val_labels in val_loader:
            val_outputs = classification_model(val_images)
            val_loss = classification_criterion(val_outputs, val_labels)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Save the trained model if needed
torch.save(classification_model.state_dict(), 'classification_model.pth')



# data_dir = 'D:/Ngodink/Python/Pytorchic/MasksDetect/ds_masks'  # Update with your dataset path

# # Define a placeholder dataset for accessing the felzenszwalb_segmentation method
# placeholder_dataset = PascalVOCDataset(data_dir=data_dir)

# # Transformation for Felzenszwalb segmentation
# felzenszwalb_transform = transforms.Compose([
#     transforms.Lambda(lambda x: (transforms.ToTensor()(x),)),
#     transforms.Lambda(lambda x: (x[0], placeholder_dataset.felzenszwalb_segmentation(x[0]))),
# ])

# # Transformation for LBP feature extraction
# lbp_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
# ])

# # Create datasets with different transformations
# dataset_felzenszwalb = PascalVOCDataset(data_dir=data_dir, transform=felzenszwalb_transform)
# dataset_lbp = PascalVOCDataset(data_dir=data_dir, transform=lbp_transform, feature_extraction=True)

# # Access the Felzenszwalb segmentation mask for a specific image
# sample_idx_felzenszwalb = 0
# image_felzenszwalb, annotation_felzenszwalb = dataset_felzenszwalb[sample_idx_felzenszwalb]
# image_felzenszwalb, seg_mask_felzenszwalb = image_felzenszwalb

# # Plot the original image and Felzenszwalb segmentation mask
# plt.subplot(1, 2, 1)
# plt.imshow(np.array(image_felzenszwalb.permute(1, 2, 0)))
# plt.title("Original Image")

# plt.subplot(1, 2, 2)
# plt.imshow(seg_mask_felzenszwalb, cmap='viridis')
# plt.title("Felzenszwalb Segmentation Mask")

# plt.show()

# # Access the LBP features for a specific image
# sample_idx_lbp = 0
# image_lbp, lbp_features, annotation_lbp = dataset_lbp[sample_idx_lbp]

# # Plot the original image and display LBP features
# plt.subplot(1, 2, 1)
# plt.imshow(np.array(image_lbp.permute(1, 2, 0)))
# plt.title("Original Image")

# plt.subplot(1, 2, 2)
# plt.bar(range(len(lbp_features)), lbp_features)
# plt.title("LBP Features")

# plt.show()
