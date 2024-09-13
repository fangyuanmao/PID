import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.CenterCrop(512),           # Center crop to 512x512
    transforms.Resize(512),               # Resize to 512x512
    transforms.RandomHorizontalFlip(),    # Random horizontal flip
    transforms.RandomCrop(512),           # Random crop of 512x512
    transforms.ToTensor(),                # Convert image to tensor
])

eval_transforms = transforms.Compose([
    transforms.CenterCrop(512),           # Center crop to 512x512
    transforms.Resize(512),               # Resize to 512x512
    transforms.ToTensor(),                # Convert image to tensor
])

class TrainDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(TrainDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform if transform is not None else train_transforms

        # List all image files in the directories
        self.images = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('jpg', 'png'))])

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load images
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor and normalization
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, image

    def __len__(self):
        return len(self.images)


class EvalDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(EvalDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform if transform is not None else eval_transforms

        # List all image files in the directories
        self.images = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('jpg', 'png'))])

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load images
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor and normalization
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, image

    def __len__(self):
        return len(self.images)
