import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE AMBIENTE E DIRETÓRIOS
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q30\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO DE AMOSTRAGEM ALEATÓRIA
# ---------------------------------------------------------
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, min(3, len(all_files)))

# Definição dos níveis de clusterização (K-clusters)
valores_k = [2, 3, 5]

# ---------------------------------------------------------
# 3. SEGMENTAÇÃO POR K-MEANS CLUSTERING
# ---------------------------------------------------------
for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: 
        continue

    # Pré-processamento: Conversão para vetor (N, 1) em float32 para o solver K-means
    pixel_values = img.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Critérios de Parada:
    # 1. cv2.TERM_CRITERIA_EPS: Precisão desejada (0.2)
    # 2. cv2.TERM_CRITERIA_MAX_ITER: Máximo de iterações (100)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Armazenamento dos resultados para comparação visual
    resultados = [img] 

    for k in valores_k:
        # Aplicação do algoritmo K-means
        # attempts=10: Executa o algoritmo 10 vezes com sementes diferentes para melhor convergência
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Quantização: Mapeia cada pixel ao valor de intensidade de seu respectivo centroide
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        
        # Reestruturação para as dimensões espaciais originais
        segmented_image = segmented_data.reshape((img.shape))
        resultados.append(segmented_image)

    # ---------------------------------------------------------
    # 4. EXPORTAÇÃO E ANALISE VISUAL
    # ---------------------------------------------------------
    plt.figure(figsize=(20, 5))
    
    titles = ['Original', f'K = 2 (Binário)', f'K = 3 (Ternário)', f'K = 5 (Multi-nível)']
    
    for i in range(len(resultados)):
        plt.subplot(1, 4, i+1)
        plt.imshow(resultados[i], cmap='gray')
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    # Ajuste de layout para garantir integridade dos títulos superiores
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q30_kmeans_variacao_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Resultados de clusterização salvos em: {path_output}")