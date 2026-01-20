import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q30\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

# Valores de K para comparar
valores_k = [2, 3, 5]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # O K-means do OpenCV espera uma matriz de pontos (N, 1) do tipo float32
    pixel_values = img.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Critérios de parada: (Tipo, Iterações Máximas, Precisão Epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Lista para armazenar os resultados das segmentações
    resultados = [img] # Inclui a original para comparação

    for k in valores_k:
        # Aplicação do K-means
        # flags: cv2.KMEANS_RANDOM_CENTERS (centros aleatórios)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Converte os centros de volta para uint8
        centers = np.uint8(centers)
        
        # Mapeia cada pixel para o valor do seu centroide correspondente
        segmented_data = centers[labels.flatten()]
        
        # Redimensiona de volta para o formato original da imagem
        segmented_image = segmented_data.reshape((img.shape))
        resultados.append(segmented_image)

    # --- PLOTAGEM COM LAYOUT SEGURO ---
    plt.figure(figsize=(20, 5))
    
    titles = ['Original', 'K = 2 (Binário)', 'K = 3 (Ternário)', 'K = 5 (Multi-nível)']
    
    for i in range(len(resultados)):
        plt.subplot(1, 4, i+1)
        plt.imshow(resultados[i], cmap='gray')
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q30_kmeans_variacao_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 30 concluída! Segmentações K-means salvas em {path_output}")