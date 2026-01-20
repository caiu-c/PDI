import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q25\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Definição dos Elementos Estruturantes (Kernel Size 5x5)
kernels = {
    "Retangular": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    "Cruz": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
    "Eliptico": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
}

# 3. Processamento
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Binarização (Otsu)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for name, k in kernels.items():
        # Aplicação das operações
        img_erosion = cv2.erode(img_bin, k, iterations=1)
        img_dilation = cv2.dilate(img_bin, k, iterations=1)

        # AJUSTE DA FIGURA: Aumentamos a altura (figsize) para acomodar os títulos
        plt.figure(figsize=(16, 6))
        
        titles = ['Original Binaria', f'Erosao ({name})', f'Dilatacao ({name})']
        images = [img_bin, img_erosion, img_dilation]

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(images[i], cmap='gray')
            # Adicionado pad para afastar o título da imagem
            plt.title(titles[i], fontsize=13, pad=15)
            plt.axis('off')

        # O rect=[0, 0.03, 1, 0.95] reserva 5% de espaço no topo para o título
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        nome_saida = f'q25_{name}_{file}.png'
        # bbox_inches='tight' garante que nada seja cortado no salvamento
        plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
        plt.close()

print("Questão 25 concluída! Imagens salvas com títulos preservados.")