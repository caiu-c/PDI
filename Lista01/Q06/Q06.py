# Questão 6 - Aplicação dos Filtros Sobel e Comparação com Prewitt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q06\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        continue

    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # --- PROCESSAMENTO SOBEL ---
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Cálculo da Magnitude: sqrt(Gx^2 + Gy^2)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag_abs = cv2.convertScaleAbs(sobel_mag)

    # Cálculo da Direção (Ângulo em radianos): atan2(Gy, Gx)
    sobel_dir = np.arctan2(sobel_y, sobel_x)

    # --- PROCESSAMENTO PREWITT (Para Comparação) ---
    kernel_prew_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_prew_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    prew_x = cv2.filter2D(img, cv2.CV_64F, kernel_prew_x)
    prew_y = cv2.filter2D(img, cv2.CV_64F, kernel_prew_y)
    prew_mag = cv2.convertScaleAbs(np.sqrt(prew_x**2 + prew_y**2))

    # 3. Exibição e Comparação
    titles = ['Original', 'Sobel Magnitude', 'Sobel Direção (Fase)', 'Prewitt Magnitude']
    images = [img, sobel_mag_abs, sobel_dir, prew_mag]

    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(titles[0])
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(sobel_mag_abs, cmap='gray')
    plt.title(titles[1])
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(sobel_dir, cmap='jet')
    plt.title(titles[2])
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 4)
    plt.imshow(prew_mag, cmap='gray')
    plt.title(titles[3])
    plt.axis('off')

    plt.tight_layout()

    # 4. Salvar resultado
    nome_saida = f"q6_sobel_vs_prewitt_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 6 concluída! Magnitude e Direção calculadas.")
