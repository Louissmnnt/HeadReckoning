# ============================================
# 🔍 Test du modèle DETR (facebook/detr-resnet-50)
# ============================================

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Chargement du processeur et du modèle pré-entraîné ---
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# --- Exemple d'image à tester ---
# 📁 Ouvre une image locale (remplace le chemin par le tien)
image = Image.open("data/images/salon.png")


# --- Préparation de l'image ---
inputs = processor(images=image, return_tensors="pt")

# --- Inférence du modèle ---
with torch.no_grad():
    outputs = model(**inputs)

# --- Post-traitement des résultats ---
target_sizes = torch.tensor([image.size[::-1]])  # (hauteur, largeur)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# --- Affichage des détections ---
plt.figure(figsize=(10, 8))
plt.imshow(image)
ax = plt.gca()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.8:  # Seulement les objets avec une confiance > 80 %
        xmin, ymin, xmax, ymax = box.tolist()
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        text = f"{model.config.id2label[label.item()]}: {score:.2f}"
        plt.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

plt.axis("off")
plt.show()