import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from data import load_datasets
from model import get_model
import yaml
import os


def predict_with_tta(model, pil_image, device, n_augmentations=4):
    """Predict with Test-Time Augmentation using PIL Image"""
    model.eval()

    # Base transform (expects PIL Image)
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Augmentation transforms
    augmentations = [
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]

    predictions = []

    # Original image
    img_tensor = base_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        predictions.append(torch.softmax(output, dim=1))

    # Augmented images
    for i, aug in enumerate(augmentations):
        if i >= n_augmentations - 1:
            break
        try:
            img_tensor = aug(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                predictions.append(torch.softmax(output, dim=1))
        except Exception as e:
            continue  # Skip failed augmentations

    # Average predictions
    if predictions:
        avg_pred = torch.mean(torch.stack(predictions), dim=0)
        return avg_pred
    else:
        # Fallback to original
        return torch.softmax(model(base_transform(pil_image).unsqueeze(0).to(device)), dim=1)


def evaluate_with_tta():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw dataset WITHOUT transforms for TTA
    data_dir = os.path.join(os.path.dirname(__file__), '..', '.cache', 'kagglehub', 'datasets',
                            'sartajbhuvaji', 'brain-tumor-classification-mri', 'versions')

    # Find the dataset directory
    dataset_path = None
    for root, dirs, files in os.walk(os.path.expanduser("~/.cache/kagglehub")):
        if "Testing" in dirs and "Training" in dirs:
            dataset_path = root
            break

    if dataset_path is None:
        raise FileNotFoundError("Could not find dataset directory")

    # Load raw test dataset (no transforms)
    from torchvision import datasets
    test_dataset_raw = datasets.ImageFolder(
        os.path.join(dataset_path, "Testing")
    )

    class_names = test_dataset_raw.classes

    # Load best model
    model = get_model(
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model.load_state_dict(torch.load(config['training']['save_path'], weights_only=True))
    model = model.to(device)

    # Evaluate with TTA
    correct = 0
    total = 0

    print("Evaluating with Test-Time Augmentation...")
    for i, (pil_image, label) in enumerate(test_dataset_raw):
        if i % 50 == 0:
            print(f"Processed {i}/{len(test_dataset_raw)} samples")

        tta_pred = predict_with_tta(model, pil_image, device, n_augmentations=4)
        predicted = torch.argmax(tta_pred, dim=1).item()

        if predicted == label:
            correct += 1
        total += 1

    tta_accuracy = correct / total
    print(f"\n🎯 TTA Accuracy: {tta_accuracy:.4f} ({tta_accuracy * 100:.2f}%)")

    return tta_accuracy


if __name__ == "__main__":
    evaluate_with_tta()