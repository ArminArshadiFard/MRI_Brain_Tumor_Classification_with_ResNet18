import os
import sys
import kagglehub
from torchvision import datasets, transforms
from utils import setup_logger

logger = setup_logger("data", "logs/data.log")

def download_dataset(dataset_name: str, force_download: bool = False):
    try:
        logger.info(f" Downloading dataset: {dataset_name}")
        if force_download:
            import shutil
            cache_dir = os.path.expanduser("~/.cache/kagglehub")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logger.info("Cleared kagglehub cache")

        path = kagglehub.dataset_download(dataset_name)
        logger.info(f" Dataset downloaded to: {path}")
        return path

    except Exception as e:
        logger.error(f" Failed to download dataset: {str(e)}")
        sys.exit(1)

def get_data_transforms(input_size=224):
    return {
        'Training': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def load_datasets(config):
    os.makedirs("logs", exist_ok=True)

    data_dir = download_dataset(config['data']['dataset_name'])
    transforms = get_data_transforms(config['data']['input_size'])

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transforms[x])
        for x in ['Training', 'Testing']
    }

    logger.info(f" Loaded datasets:")
    logger.info(f"   Training: {len(image_datasets['Training'])} samples")
    logger.info(f"   Testing:  {len(image_datasets['Testing'])} samples")
    logger.info(f"   Classes:  {image_datasets['Training'].classes}")

    return image_datasets