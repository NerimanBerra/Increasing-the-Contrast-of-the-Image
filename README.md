import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread(r"C:\Users\berra\Desktop\RKSoft\4\animals5.png")


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntünün kontrastını artır
lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) uygulayarak kontrast artırma
clahe = cv2.createCLAHE(clipLimit=-3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

img = cv2.merge((cl, a, b))
enhanced_image = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
cv2.imwrite(r"C:\Users\berra\Desktop\RKSoft\5\animals4.png", enhanced_image)
# Sonuçları görselleştir
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Orijinal Görüntü')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Kontrast Arttırılmış Görüntü')
plt.imshow(enhanced_image)
plt.axis('off')

plt.show()
