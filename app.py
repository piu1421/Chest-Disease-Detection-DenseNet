"""
Chest Disease Detection Using DenseNet Architecture

Institution: Amity University, Noida
"""

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ==========================================
# 1. PREPROCESSING PIPELINE
# ==========================================


def preprocess_image(pil_image):
    open_cv_image = np.array(pil_image.convert("RGB"))
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Convert back to 3-channel RGB image for DenseNet input
    rgb_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
    pil_enhanced = Image.fromarray(rgb_enhanced)

    # Resizing & Tensor Transformation
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    tensor_image = transform_pipeline(pil_enhanced).unsqueeze(0)
    return tensor_image, pil_enhanced


# ==========================================
# 2. MODEL INITIALIZATION (DenseNet121)
# ==========================================


def load_densenet_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3),
    )
    model.eval()
    return model


model = load_densenet_model()
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================


def predict_chest_disease(input_image):
    if input_image is None:
        return "Please upload a valid chest X-ray image.", None

    input_tensor, processed_pil = preprocess_image(input_image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidences = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }
    return confidences, processed_pil


# ==========================================
# 4. GRADIO USER INTERFACE
# ==========================================

custom_title = "Chest Disease Detection Using DenseNet Architecture"
custom_description = """
### Automated Diagnostic Support System
This system utilizes **DenseNet Deep Learning Architecture** combined with image preprocessing techniques 
(Histogram Equalization/CLAHE, Resizing, and Normalization) to detect thoracic conditions from chest X-ray images.

**Target Classes:**
1. **Normal**
2. **Pneumonia**
3. **Tuberculosis**
"""

custom_article = """
<div style='text-align: center;'>
    <p><b>Developed by:</b> SAMPURNA SINHA</p>
    <p><b>Guided by:</b> SWETA SRINIVASTAVA</p>
    <p><i>Amity University, Noida</i></p>
</div>
"""

iface = gr.Interface(
    fn=predict_chest_disease,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray Image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Diagnostic Prediction & Confidence"),
        gr.Image(label="Preprocessed Image Output (CLAHE Enhanced)"),
    ],
    title=custom_title,
    description=custom_description,
    article=custom_article,
    theme="default",
)

if __name__ == "__main__":
    iface.launch(share=False)


if __name__ == "__main__":
    iface.launch(share=False)
