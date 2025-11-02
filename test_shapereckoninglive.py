# ============================================
# 🔍 Test du modèle DETR (facebook/detr-resnet-50)
# ============================================

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import requests
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Chargement du processeur et du modèle pré-entraîné ---
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Initialisation de la caméra ---
cam = cv2.VideoCapture(0)  # Ouvre la caméra par défaut (index 0)

# --- Boucle principale ---
while True:
    ret, frame_bgr = cam.read()

    # OpenCV -> BGR ; DETR attend RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Préparation entrée (sur le bon device)
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)

    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-traitement (taille cible = (h, w))
    h, w = frame_rgb.shape[:2]
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Dessin des détections avec OpenCV
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = float(score)
        if score < 0.8:
            continue

        x_min, y_min, x_max, y_max = box.to("cpu").tolist()
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cls = model.config.id2label[int(label)]
        cv2.putText(frame_bgr, f"{cls}: {score:.2f}",
                    (x_min, max(0, y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Affichage temps réel
    cv2.imshow("Video", frame_bgr)
    
    if cv2.waitKey(1) == ord('q'):  # Si l’utilisateur appuie sur la touche 'q'
        break  # Sort de la boucle

# --- Nettoyage ---
cam.release()  # Libère la caméra (important pour éviter de la bloquer)
cv2.destroyAllWindows()  # Ferme toutes les fenêtres ouvertes par OpenCV
