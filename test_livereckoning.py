import cv2  # Importation de la bibliothèque OpenCV pour le traitement d'images et de vidéos

# --- Initialisation de la caméra ---
cam = cv2.VideoCapture(0)  # Ouvre la caméra par défaut (index 0)

# --- Chargement du classificateur de visages ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# On charge un modèle pré-entraîné (fichier XML) pour détecter les visages dans une image

# --- Boucle principale ---
while True:
    ret, frame = cam.read()  # Capture une image (frame) depuis la caméra
    # 'ret' indique si la capture s’est bien déroulée (True ou False)
    # 'frame' contient l’image capturée

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion de l’image couleur en niveaux de gris
    # La détection de visage est plus rapide et plus fiable sur une image en niveaux de gris

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Détection des visages dans l’image
    # Paramètres :
    #   1.3 → facteur d’échelle (réduction de l’image à chaque étape de la recherche)
    #   5   → nombre minimal de voisins (filtre les détections trop faibles)

    for (x, y, w, h) in faces:  # Pour chaque visage détecté :
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Dessine un rectangle bleu autour du visage détecté
        # (x, y) = coin supérieur gauche, (x + w, y + h) = coin inférieur droit
        # (255, 0, 0) = couleur (bleu en BGR), 2 = épaisseur du trait

    cv2.imshow("Video", frame)  # Affiche l’image (avec rectangles) dans une fenêtre nommée "Video"

    if cv2.waitKey(1) == ord('q'):  # Si l’utilisateur appuie sur la touche 'q'
        break  # Sort de la boucle

# --- Nettoyage ---
cam.release()  # Libère la caméra (important pour éviter de la bloquer)
cv2.destroyAllWindows()  # Ferme toutes les fenêtres ouvertes par OpenCV
