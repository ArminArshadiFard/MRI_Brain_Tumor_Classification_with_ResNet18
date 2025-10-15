import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from src.model import get_model, GradCAM

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading our trained model
trained_model = get_model("resnet18", 4)
trained_model.load_state_dict(torch.load("../src/checkpoints/best-model-checkpoint.pt", map_location=device))
trained_model = trained_model.to(device).eval()

# the same ones used while training
tumor_classes = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


def classify_mri_scan(input_image):
    # Preprocess image so model accepts them
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_img = preprocess(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = trained_model(tensor_img)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Generate Grad-CAM explanation
    gradcam = GradCAM(trained_model, trained_model.layer4[-1])
    cam_map = gradcam(tensor_img)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    resized_original = np.array(input_image.resize((224, 224)))
    blended = cv2.addWeighted(heatmap, 0.6, resized_original, 0.4, 0)

    blended_rgb = blended[:, :, ::-1]

    predictions = {tumor_classes[i]: float(probabilities[i]) for i in range(4)}

    return predictions, blended_rgb


if __name__ == "__main__":
    demo = gr.Interface(
        fn=classify_mri_scan,
        inputs=gr.Image(type="pil", label="Upload a brain scan"),
        outputs=[
            gr.Label(num_top_classes=4, label="Diagnosis"),
            gr.Image(label="Model Attention (Grad-CAM)")
        ],
        title="testing the trained model manually",
        description=(
            "Upload a picture of a brain MRI scan to start."
        ),
        examples=[
            ["glioma.jpg"],
            ["meningioma.jpg"],
            ["healthy.jpg"],
            ["pituitary.jpg"]
        ],
        cache_examples=False
    )
    demo.launch()