import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q38\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Inicialização do detector SIFT
    # Nas versões recentes do OpenCV (4.4+), o SIFT está no pacote principal
    sift = cv2.SIFT_create()

    # Detecção de pontos-chave (keypoints) e cálculo de descritores
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 4. Desenhar os pontos-chave na imagem
    # DRAW_RICH_KEYPOINTS desenha círculos com o tamanho e a orientação do ponto
    img_sift = cv2.drawKeypoints(img, keypoints, None, 
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                 color=(0, 255, 0))

    # --- PLOTAGEM COM LAYOUT PROTEGIDO ---
    plt.figure(figsize=(16, 8))
    
    # Converter para RGB para exibição correta
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_sift = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
    
    titles = [f'Original ({file})', f'SIFT: {len(keypoints)} Pontos Detectados']
    images = [img_rgb, res_sift]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14, pad=30) # Pad maior para evitar corte
        plt.axis('off')

    # Ajuste de margens para preservar o título no topo
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q38_sift_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 38 concluída! Pontos SIFT extraídos para as imagens: {files}")