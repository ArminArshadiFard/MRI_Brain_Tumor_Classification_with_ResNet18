import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np


def get_model(arch="resnet18", num_classes=4, pretrained=True):
    weights = "IMAGENET1K_V1" if pretrained else None

    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        if x.ndim != 4:
            raise ValueError("Input tensor must be 4D (batch, channels, height, width)")
        self.model.eval()
        x.requires_grad_(True)
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer.")
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam
        return cam