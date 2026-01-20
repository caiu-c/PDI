import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q31\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO ALEATÓRIA DE AMOSTRAS
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, min(3, len(all_files)))

# ---------------------------------------------------------
# 3. PROCESSAMENTO COMPARATIVO DE SEGMENTAÇÃO
# ---------------------------------------------------------
for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None: 
        continue

    # --- MÉTODO A: Limiarização de Otsu ---
    # Algoritmo de binarização global baseado na minimização da variância intraclasse
    limiar_otsu, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- MÉTODO B: Clusterização K-means (K=2) ---
    # Segmentação baseada em agrupamento de intensidades no espaço de cores
    pixel_values = np.float32(img.reshape((-1, 1)))
    
    # Critérios de parada: 100 iterações ou precisão de 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Execução do K-means com 10 reinicializações para convergência estável
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reconstrução da imagem quantizada com base nos centroides calculados
    centers = np.uint8(centers)
    img_kmeans = centers[labels.flatten()].reshape(img.shape)

    # ---------------------------------------------------------
    # 4. GERAÇÃO DE RESULTADOS E EXPORTAÇÃO
    # ---------------------------------------------------------
    plt.figure(figsize=(18, 6))
    
    titles = [
        f'Original ({file})', 
        f'Limiarização Otsu (T={int(limiar_otsu)})', 
        'Clusterização K-means (K=2)'
    ]
    images = [img, img_otsu, img_kmeans]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=13, pad=25)
        plt.axis('off')

    # Ajuste de layout para preservação de títulos e exportação em alta resolução
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q31_comparativo_segmentacao_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Comparativos salvos em: {path_output}")