import cv2
import numpy as np

# Charger l'image
image = cv2.imread("data/images/visage_feminin.jpg")

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binariser (seuil automatique)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Trouver les contours extérieurs
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer une image vide pour dessiner les contours
contour_image = np.zeros_like(image)

# Dessiner les contours en vert
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Afficher les résultats
cv2.imshow("Original", image)
cv2.imshow("Contours Exterieurs", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
