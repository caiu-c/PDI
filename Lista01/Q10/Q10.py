# Questão 10 - Multilimiarização para Segmentação em 3 Regiões de Intensidade
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q10\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    # 2. Carregar imagem em tons de cinza
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Aplicar Multilimiarização
    # Definimos dois limiares para separar a imagem em 3 níveis de intensidade
    T1 = 85
    T2 = 170

    # Criamos uma imagem de saída com zeros
    # Região 1: pixels < T1 -> valor 0 (Preto)
    # Região 2: T1 <= pixels < T2 -> valor 127 (Cinza)
    # Região 3: pixels >= T2 -> valor 255 (Branco)
    
    multithresholded = np.zeros_like(img)
    multithresholded[(img >= T1) & (img < T2)] = 127
    multithresholded[img >= T2] = 255

    # 4. Exibição e Comparação
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Grayscale)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.hist(img.ravel(), 256, [0, 256], color='black')
    plt.axvline(T1, color='r', linestyle='--', label=f'T1={T1}')
    plt.axvline(T2, color='g', linestyle='--', label=f'T2={T2}')
    plt.title('Histograma e Limiares')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.imshow(multithresholded, cmap='gray')
    plt.title('Imagem Multilimiarizada (3 Níveis)')
    plt.axis('off')

    plt.tight_layout()

    # 5. Salvar resultado
    nome_saida = f"q10_multilimiarizacao_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 10 concluída! Segmentação em múltiplas regiões realizada.")
