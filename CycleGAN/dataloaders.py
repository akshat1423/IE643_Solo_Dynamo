# Dataset Class
class RESIDE_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, paired=True, num_paired=25, max_unclean=500, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.paired = paired
        self.num_paired = num_paired
        self.max_unclean = max_unclean
        self.paired_paths = self._load_image_paths() if paired else None
        self.unclean_images = self._load_unclean_images() if not paired else None

    def _load_image_paths(self):
        hazy_dir = os.path.join(self.root_dir, 'hazy')
        clear_dir = os.path.join(self.root_dir, 'GT')

        # Load paired images, limit to num_paired
        hazy_paths = sorted(os.listdir(hazy_dir))[:self.num_paired]
        clean_paths = sorted(os.listdir(clear_dir))[:self.num_paired]
        
        paired_paths = []
        for hazy_name, clean_name in zip(hazy_paths, clean_paths):
            hazy_path = os.path.join(hazy_dir, hazy_name)
            clean_path = os.path.join(clear_dir, clean_name)
            paired_paths.append((hazy_path, clean_path))

        return paired_paths

    def _load_unclean_images(self):
        hazy_dir = os.path.join(self.root_dir, 'hazy')
        hazy_paths = sorted(os.listdir(hazy_dir))

        # Randomly sample 500 unclean images from the dataset
        sampled_hazy_paths = random.sample(hazy_paths, min(self.max_unclean, len(hazy_paths)))
        return [os.path.join(hazy_dir, hazy_name) for hazy_name in sampled_hazy_paths]

    def __len__(self):
        return len(self.paired_paths) if self.paired else len(self.unclean_images)

    def __getitem__(self, idx):
        if self.paired:
            hazy_img_path, clean_img_path = self.paired_paths[idx]
            hazy_img = Image.open(hazy_img_path).convert('RGB')
            clean_img = Image.open(clean_img_path).convert('RGB')
            if self.transform:
                hazy_img = self.transform(hazy_img)
                clean_img = self.transform(clean_img)
            return hazy_img, clean_img
        else:
            hazy_img_path = self.unclean_images[idx]
            hazy_img = Image.open(hazy_img_path).convert('RGB')
            if self.transform:
                hazy_img = self.transform(hazy_img)
            return hazy_img

# Transformation for resizing and converting images to tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Initialize datasets and loaders with pin_memory=True
paired_dataset = RESIDE_Dataset(root_dir='../input/hazing-images-dataset-cvpr-2019', paired=True, num_paired=25, transform=transform)
paired_loader = DataLoader(paired_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

# unpaired_dataset = RESIDE_Dataset(root_dir='../input/hazing-images-dataset-cvpr-2019', paired=False, max_unclean=50, transform=transform)
unpaired_dataset_unclean_its = RESIDE_Dataset(root_dir='../input/indoor-training-set-its-residestandard', paired=False, max_unclean=1000, transform=transform)
unpaired_dataset_unclean_ots = RESIDE_Dataset(root_dir='../input/synthetic-objective-testing-set-sots-reside', paired=False, max_unclean=1000, transform=transform)
unpaired_dataset_clean_its= RESIDE_Dataset(root_dir='../input/indoor-training-set-its-residestandard', paired=False, max_unclean=1000, transform=transform)
unpaired_dataset_clean_ots = RESIDE_Dataset(root_dir='../input/synthetic-objective-testing-set-sots-reside', paired=False, max_unclean=1000, transform=transform)

unpaired_loader = DataLoader(unpaired_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)  # Reduced batch size for unpaired
