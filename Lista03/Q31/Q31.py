import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q31\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # --- MÉTODO A: Limiarização de Otsu ---
    # Retorna o limiar calculado e a imagem binária
    limiar_otsu, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- MÉTODO B: Clusterização K-means (K=2 para comparação direta) ---
    pixel_values = np.float32(img.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    img_kmeans = centers[labels.flatten()].reshape(img.shape)

    # --- PLOTAGEM COMPARATIVA ---
    plt.figure(figsize=(18, 6))
    
    titles = [
        f'Original ({file})', 
        f'Limiarizacao Otsu (T={int(limiar_otsu)})', 
        'Clusterizacao K-means (K=2)'
    ]
    images = [img, img_otsu, img_kmeans]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=13, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q31_comparativo_segmentacao_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 31 concluída! Comparativos salvos em {path_output}")