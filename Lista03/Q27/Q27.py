import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q27\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # --- MÉTODO 1: Gradiente Morfológico ---
    # Diferença entre dilatação e erosão
    kernel = np.ones((3, 3), np.uint8)
    grad_morph = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # --- MÉTODO 2: Sobel (Magnitude) ---
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_sobel = cv2.magnitude(sobelx, sobely)
    grad_sobel = np.uint8(cv2.normalize(grad_sobel, None, 0, 255, cv2.NORM_MINMAX))

    # --- MÉTODO 3: Laplaciano ---
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    # --- PLOTAGEM COM LAYOUT SEGURO ---
    plt.figure(figsize=(20, 5))
    
    titles = ['Original', 'Gradiente Morfologico', 'Sobel (Magnitude)', 'Laplaciano']
    images = [img, grad_morph, grad_sobel, laplacian]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    # Ajuste do rect para evitar corte nos títulos
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q27_comparativo_realce_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 27 concluída! Comparativo de realce salvo em {path_output}")