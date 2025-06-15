import torch
import torch.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

dataset = ImageFolder("images")

n_total = len(dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_val - n_train
train_indices, val_indices, test_indices = torch.utils.data.random_split(
    range(n_total), [n_train, n_val, n_test]
)

# apply augmentation for train_dataset
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0, scale=(0.9, 1.1), translate=(0.1, 0.1), shear=5
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ]
)
# test and val dataset
val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


train_dataset = TransformSubset(
    torch.utils.data.Subset(dataset, train_indices), train_transform
)
val_dataset = TransformSubset(
    torch.utils.data.Subset(dataset, train_indices), val_transforms
)
test_dataset = TransformSubset(
    torch.utils.data.Subset(dataset, test_indices), val_transforms
)
