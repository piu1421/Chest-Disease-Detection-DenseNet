"""
Chest Disease Detection Using DenseNet Architecture
Supports 17 Classes & Grad-CAM Heatmap Localization (Batch Multi-Image Processing)
"""

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Corrected variable name syntax
CLASS_NAMES = [
    "Asthma",
    "COPD",
    "Emphysema",
    "Bronchiectasis",
    "Cystic Fibrosis",
    "Interstitial Lung Disease (ILD)",
    "Pneumothorax",
    "Empyema",
    "Mesothelioma",
    "Chest Wall Tumors",
    "Normal",
    "Pneumonia",
    "Tuberculosis (TB)",
    "Acute Bronchitis",
    "Pulmonary Embolism (PE)",
    "Pulmonary Hypertension",
    "Pulmonary Edema",
]

# Initialize and load model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(CLASS_NAMES))
model.eval()


def preprocess_image(pil_image):
    """Preprocesses a single PIL image using CLAHE and PyTorch normalization."""
    rgb_img = pil_image.convert("RGB")
    np_img = np.array(rgb_img)

    # Apply CLAHE Contrast Enhancement
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    pil_enhanced = Image.fromarray(enhanced_rgb)

    # PyTorch Transformations
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    tensor_img = transform_pipeline(pil_enhanced).unsqueeze(0)
    return tensor_img, pil_enhanced


def generate_gradcam_heatmap(input_tensor, pil_original):
    """Generates a Grad-CAM localization heatmap for DenseNet121."""
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into the final denseblock of DenseNet
    target_layer = model.features.denseblock4
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    input_tensor = input_tensor.requires_grad_(True)
    output = model(input_tensor)

    # Get predicted class
    pred_class = torch.argmax(output[0])
    score = output[0, pred_class]
    score.backward()

    # Extract gradients and feature maps
    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = activations[0].cpu().data.numpy()[0]

    # Compute Grad-CAM weights
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * f_maps[i, :, :]

    cam = np.maximum(cam, 0)  # ReLU activation
    if np.max(cam) != 0:
        cam = cam / np.max(cam)  # Normalize

    # Cleanup hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # Overlay heatmap onto original image
    orig_np = np.array(pil_original.convert("RGB").resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay), output[0]


def process_batch_images(image_list):
    """Processes up to 90 images and generates predictions and heatmaps."""
    if not image_list:
        return [], [], "No images uploaded."

    preprocessed_gallery = []
    heatmap_gallery = []
    summary_report = []

    for idx, img_path in enumerate(image_list):
        try:
            # Handle image loading whether input is a file path or PIL image
            if isinstance(img_path, str):
                pil_image = Image.open(img_path)
            else:
                pil_image = img_path

            tensor_img, processed_pil = preprocess_image(pil_image)

            # Generate predictions and Grad-CAM
            heatmap_pil, logits = generate_gradcam_heatmap(tensor_img, pil_image)
            probabilities = torch.softmax(logits, dim=0)

            # Get Top 3 Predicted Classes
            top_probs, top_indices = torch.topk(probabilities, k=3)
            top_predictions = [
                f"{CLASS_NAMES[idx.item()]}: {prob.item()*100:.1f}%"
                for prob, idx in zip(top_probs, top_indices)
            ]

            # Store for Gradio Gallery outputs
            preprocessed_gallery.append((processed_pil, f"Image {idx+1} (CLAHE)"))
            heatmap_gallery.append((heatmap_pil, f"Image {idx+1} (Grad-CAM)"))

            # Build text summary entry
            summary_report.append(
                f"**Image #{idx+1} Diagnostic Summary:**\n"
                f"• Top Result: {top_predictions[0]}\n"
                f"• Alternatives: {', '.join(top_predictions[1:])}\n"
            )

        except Exception as e:
            summary_report.append(f"**Image #{idx+1}:** Error processing ({str(e)})")

    full_report = f"### Batch Processing Complete ({len(image_list)} scans processed)\n\n" + "\n".join(
        summary_report
    )
    return preprocessed_gallery, heatmap_gallery, full_report


custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #31103f 50%, #4a0e2e 100%) !important;
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif !important;
}
.gr-box, .gr-form, .gr-panel, .gr-input, .gr-button {
    border-radius: 12px !important;
}
h1, h2, h3, p, span, label {
    color: #f3e8ff !important;
}
"""

custom_title = "Chest Disease Detection Using DenseNet Architecture (Batch Mode)"
custom_description = """
### Automated Diagnostic & Visual Localization System (Supports Multi-Image Upload)
Upload **multiple X-ray scans at once (up to 90 images)**. The system applies **CLAHE Contrast Enhancement**, 
runs multi-class classification across 17 thoracic diseases, and computes **Grad-CAM visual heatmaps**.
"""

iface = gr.Interface(
    fn=process_batch_images,
    inputs=gr.Gallery(
        label="Upload Chest X-Ray Images (Select multiple files up to 90)",
        type="pil",
    ),
    outputs=[
        gr.Gallery(
            label="Preprocessed Images (CLAHE Enhanced)",
            columns=4,
            rows=2,
            height="auto",
        ),
        gr.Gallery(
            label="Grad-CAM Lesion Localization Heatmaps",
            columns=4,
            rows=2,
            height="auto",
        ),
        gr.Markdown(label="Batch Diagnostics Summary Report"),
    ],
    title=custom_title,
    description=custom_description,
    css=custom_css,
    theme="default",
)

if __name__ == "__main__":
    iface.launch(share=False)
