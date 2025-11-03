import cv2
import numpy as np

# Charger une image
image = cv2.imread("data/images/visage_feminin.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Détection de contours avec Canny
edges = cv2.Canny(blurred, 50, 150)

# Trouver les contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessiner les contours sur une copie de l’image originale
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
