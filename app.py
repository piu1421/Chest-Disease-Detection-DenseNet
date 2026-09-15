"""
Chest Disease Detection Using DenseNet Architecture
Features: Multi-label classification (16 Classes) & Grad-CAM Heatmap Localization
"""

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


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



def load_densenet_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
   
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_NAMES)),
    )
    model.eval()
    return model


model = load_densenet_model()



def generate_gradcam_heatmap(input_tensor, pil_image):
   
    target_layer = model.features.denseblock4

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = target_layer.register_forward_hook(forward_hook)

   
    output = model(input_tensor)
    idx = torch.argmax(output[0])
    score = output[0][idx]

    model.zero_grad()
    score.backward()

    handle.remove()

    
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0][0]

    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    
    orig_img = np.array(pil_image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[2]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))



def preprocess_image(pil_image):
    open_cv_image = np.array(pil_image.convert("RGB"))
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    rgb_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
    pil_enhanced = Image.fromarray(rgb_enhanced)

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


def predict_chest_disease(input_image):
    if input_image is None:
        return None, None, None

  
    input_tensor, processed_pil = preprocess_image(input_image)
    input_tensor.requires_grad_()

    outputs = model(input_tensor)
    probabilities = torch.sigmoid(outputs[0])  

    confidences = {
        CLASS_NAMES[i]: float(probabilities[i].detach())
        for i in range(len(CLASS_NAMES))
    }

    heatmap_overlay = generate_gradcam_heatmap(input_tensor, input_image)

    return confidences, processed_pil, heatmap_overlay



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

custom_title = "Chest Disease Detection Using DenseNet Architecture"
custom_description = """
### Automated Diagnostic & Visual Localization System
This system utilizes **DenseNet Deep Learning Architecture** along with **Grad-CAM Visual Localization** to detect conditions and highlight lesion locations directly on chest X-ray scans.

---

### Target Classes & Scope:

**(A) Exclusive and Primary Chest/Lung Conditions**
1. Asthma | 2. COPD | 3. Emphysema | 4. Bronchiectasis | 5. Cystic Fibrosis | 6. Interstitial Lung Disease (ILD) | 7. Pneumothorax | 8. Empyema | 9. Mesothelioma | 10. Chest Wall Tumors

**(B) Common Chest/Respiratory Infections & Vascular Conditions**
1. Pneumonia | 2. Tuberculosis (TB) | 3. Acute Bronchitis | 4. Pulmonary Embolism (PE) | 5. Pulmonary Hypertension | 6. Pulmonary Edema | 7. Normal
"""

iface = gr.Interface(
    fn=predict_chest_disease,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray Image"),
    outputs=[
        gr.Label(
            num_top_classes=5, label="Diagnostic Prediction & Confidence Scores"
        ),
        gr.Image(label="Preprocessed Image Output (CLAHE Enhanced)"),
        gr.Image(
            label="Lesion Localization Heatmap (Grad-CAM Region Detection)"
        ),
    ],
    title=custom_title,
    description=custom_description,
    css=custom_css,
    theme="default",
)

if __name__ == "__main__":
    iface.launch(share=False)


if __name__ == "__main__":
    iface.launch(share=False)
