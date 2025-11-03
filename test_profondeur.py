from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Charger le modèle et le processeur
processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf")

# 2️⃣ Charger l'image
image = Image.open("data/images/salon.png").convert("RGB")

# 3️⃣ Préparer les entrées
inputs = processor(images=image, return_tensors="pt")

# 4️⃣ Prédire la carte de profondeur
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# 5️⃣ Redimensionner la carte au format de l’image originale
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
).squeeze().cpu().numpy()

# 6️⃣ Normaliser pour affichage
depth_min, depth_max = prediction.min(), prediction.max()
normalized_depth = (prediction - depth_min) / (depth_max - depth_min)

# 7️⃣ Afficher le résultat
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(normalized_depth, cmap="plasma")
plt.title("Carte de profondeur estimée")
plt.axis("off")

plt.show()